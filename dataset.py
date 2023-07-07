from typing import List
import torch
from torch.utils.data.dataset import Dataset, Subset, TensorDataset, ConcatDataset
# from imblearn.over_sampling import SMOTE
import os
import numpy as np
from tqdm import tqdm

from scipy.signal import spectrogram
from wfdb import processing
from torch.utils.data.sampler import Sampler
from torch.utils.data import (
    Dataset,
    ConcatDataset,
)
from collections import Counter
from multiprocessing import Pool
import neurokit2 as nk
import numpy as np
import wfdb
import torch
import warnings
# from scipy.io import loadmat
# import wfdb

warnings.filterwarnings("ignore")


def preTrainDataset(directories, window_size=120, cuda=False):
    # Lead 2 only, for Guoxin's work in 2023 Mar
    part_list = []
    for directory in directories:
        part_list.extend(
            list(filter(lambda x: x.endswith('.pth'), map(lambda x: os.path.join(directory, x), os.listdir(directory)))))
    # part_list=[part_list[0]]

    records = {}
    for part_path in tqdm(part_list, desc='Loading dataset'):
        part = torch.load(part_path)
        for k, record in part.items():
            if k.endswith('_2'):
                records[k] = PreTrainRecord(record, window_size=window_size, cuda=cuda)
    return ConcatDataset(records.values())


def processedPreTrainDataset(directories):
    # Lead 2 only, for Guoxin's work in 2023 Mar
    part_list = []
    for directory in directories:
        part_list.extend(
            list(filter(lambda x: x.endswith('.pth'), map(lambda x: os.path.join(directory, x), os.listdir(directory)))))
    # part_list=[part_list[0]]

    records = {}
    for part_path in tqdm(part_list, desc='Loading dataset'):
        part = torch.load(part_path)
        records[part_path] = TensorDataset(part['data'].float())
    return ConcatDataset(records.values())

def preTrainDataset2(file, cuda=False):
    signals = torch.load(file).to('cuda' if cuda else 'cpu').unsqueeze(-1)

    return signals
    # return TensorDataset(signals)


class PreTrainRecord(Dataset):

    def __init__(self, file, window_size=120, cuda=False):
        super().__init__()
        self.window_size = window_size

        signal, annpos, _ = self._load_record(file)
        # print(signal.shape)
        # Remove the beats that do not have window_size/2 on left or right
        lb = 0
        rb = len(annpos) - 1
        while lb < len(annpos) and annpos[lb] < self.window_size * ((self.window_size // 120) - 1) // (self.window_size // 120):
            lb += 1
        while rb > 0 and annpos[rb] > signal.shape[0] - self.window_size + (self.window_size *
                                                                            ((self.window_size // 120) - 1) //
                                                                            (self.window_size // 120)):
            rb -= 1
        if lb <= rb and annpos[lb] >= self.window_size * (
            (self.window_size // 120) - 1) // (self.window_size // 120) and annpos[rb] <= signal.shape[0] - self.window_size + (
                self.window_size * ((self.window_size // 120) - 1) // (self.window_size // 120)):
            self.signal = signal
            self.annpos = annpos[lb:rb + 1]
            if cuda:
                self.signal = torch.tensor(self.signal, device='cuda').float()
            else:
                self.signal = torch.tensor(self.signal, device='cpu').float()
            self.bound_cache = {}
        else:
            self.annpos = []

        self.length = len(self.annpos)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        lb = int(self.annpos[index] - self.window_size * ((self.window_size // 120) - 1) // (self.window_size // 120))
        rb = int(self.annpos[index] + self.window_size - self.window_size * ((self.window_size // 120) - 1) //
                 (self.window_size // 120))
        signal = self.signal[lb:rb, :]
        if signal.shape[0] != self.window_size:
            print(lb, rb, self.annpos[index], signal.shape[0], self.signal.shape)

        signal = (signal - signal.amin(axis=0)) / (signal.amax(axis=0) - signal.amin(axis=0) + 1e-30)
        return signal

    def _load_record(self, record):
        signal = record['signal']
        annpos = record['annpos']
        anntype = record['ann']
        # print(signal.shape,annpos.shape)
        return signal, annpos, anntype

class LabelledToUnablled(Dataset):
    def __init__(self, ds) -> None:
        super().__init__()
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x, _ = self.ds[index]
        return x



class ECGDB_UL(Dataset):

    def __init__(
        self,
        files: list,
        channel_names: list,
        fs: float = 480,
        expansion: int = 1,
        cuda: bool = False,
        transform: object = None,
    ):
        self.transform = transform
        self.expansion = expansion
        self.signals = []
        pool = Pool()
        pooltemp = []
        self.get_unlabeled_signals(files[1], fs, channel_names)
        for file in files:
            res = pool.apply_async(
                self.get_unlabeled_signals,
                args=(
                    file,
                    fs,
                    channel_names,
                ),
            )
            pooltemp.append(res)
        pool.close()
        pool.join()
        for temp in pooltemp:
            signal = temp.get()
            self.signals.extend(signal)
        self.signals = np.array(self.signals)
        self.signals = torch.tensor(self.signals, device="cuda" if cuda else "cpu",dtype=float)
        torch.save(self.signals, f'./shaoxing_resampled_len{fs}')
        print('saved')

    def __len__(self):
        return len(self.signals) * self.expansion

    def __getitem__(self, index: int):
        index = index // self.expansion
        signal = self.signals[index]
        # if self.transform:
        #     signal = [self.transform(signal), self.transform(signal)]
        # else:
        #     signal = [signal, signal]
        # signal[0] = signal[0].unsqueeze(0)
        # signal[1] = signal[1].unsqueeze(0)
        return signal

    def get_unlabeled_signals(self, path: str, fs: float, channel_names: list = None):
        signal = []
        try:
            record = wfdb.rdrecord(path, channel_names=channel_names, physical=False, return_res=16)
            for channel in range(record.d_signal.shape[1]):
                d_signal = record.d_signal[:, channel]
                d_signal = nk.ecg_clean(d_signal, sampling_rate=record.__dict__["fs"])
                _, rpeaks = nk.ecg_peaks(d_signal, sampling_rate=record.__dict__["fs"])
                _, waves_peak = nk.ecg_delineate(d_signal, rpeaks, sampling_rate=record.__dict__["fs"])
                for idx in range(len(waves_peak["ECG_P_Peaks"])):
                    start_idx = waves_peak["ECG_P_Peaks"][idx]
                    end_idx = waves_peak["ECG_T_Peaks"][idx]
                    if np.isnan(start_idx) or np.isnan(end_idx):
                        continue
                    sig = d_signal[start_idx:end_idx]
                    sig = processing.normalize_bound(sig, 0, 1)
                    sig = processing.resample_sig(sig, len(sig), fs)[0]
                    signal.append(sig)
        except:
            print("Unvailable Record: {}".format(path))
        return signal
