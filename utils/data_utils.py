"""
Author: Guoxin Wang
Date: 2022-04-29 14:54:52
LastEditors: Guoxin Wang
LastEditTime: 2024-04-23 08:58:36
FilePath: /maecg/utils/data_utils.py
Description: 

Copyright (c) 2022 by Guoxin Wang, All Rights Reserved. 
"""

import os
import warnings
from collections import Counter
from multiprocessing import Pool

import neurokit2 as nk
import numpy as np
import torch
import wfdb
from sklearn import preprocessing
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from wfdb import processing

warnings.filterwarnings("ignore")

BEAT_LABEL_5 = {
    # Ntype
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    # Stype
    "A": 1,
    "S": 1,
    "a": 1,
    "J": 1,
    # Vtype
    "V": 2,
    "E": 2,
    # Ftype
    "F": 3,
    # Qtype
    "f": 4,
    "/": 4,
    "Q": 4,
}
BEAT_LABEL_4 = {
    # Ntype
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    # Stype
    "A": 1,
    "S": 1,
    "a": 1,
    "J": 1,
    # Vtype
    "V": 2,
    "E": 2,
    # Ftype
    "F": 3,
}
BEAT_LABEL_2 = {
    # Ntype
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    # Stype
    "A": 1,
    "S": 1,
    "a": 1,
    "J": 1,
    # Vtype
    "V": 1,
    "E": 1,
    # Ftype
    "F": 1,
    # Qtype
    "f": 1,
    "/": 1,
    "Q": 1,
}


class RandomCrop(object):
    def __init__(self, output_size: int = 64, p: float = 0.5) -> None:
        self.output_size = output_size
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        size = signal.size()
        if torch.rand(1) < self.p:
            rand = torch.randint(0, int(len(signal) - self.output_size), (1,)).item()
            signal = signal[rand : int(rand + self.output_size)].view(size)
        return signal


class RandomScale(object):
    def __init__(self, scale_size: int = 1.5, p: float = 0.5) -> None:
        self.scale_size = scale_size
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            signal = signal * self.scale_size
        return signal


class RandomFlip(object):
    def __init__(self, p: float = 0.5, dim: int = 0) -> None:
        self.p = p
        self.dim = dim

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            if self.dim == 0:
                signal = -signal
            else:
                signal = torch.flip(signal, dims=(0,))
        return signal


class RandomCutout(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            signal = (signal + torch.abs(signal)) / 2
        return signal


class RandomShift(object):
    def __init__(self, shift_size: int = 64, p: float = 0.5) -> None:
        self.shift_size = shift_size
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if self.shift_size:
            if torch.rand(1) < self.p:
                signal = torch.roll(signal, self.shift_size)
        return signal


class RandomSine(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            signal = torch.sin(signal)
        return signal


class RandomSquare(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            signal = torch.square(signal)
        return signal


class RandomNoise(object):
    def __init__(self, p: float = 1, mean: float = 0, std: float = 0.05) -> None:
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            signal = signal + torch.normal(
                mean=self.mean, std=self.std, size=signal.size()
            )
            signal = (signal - signal.min()) / (signal.max() - signal.min()) * 2 - 1
        return signal


class ECG_Beat_UL(Dataset):
    def __init__(
        self,
        files: list,
        width: int,
        channel_names: list,
        expansion: int,
        transform: object = None,
    ) -> None:
        self.transform = transform
        self.expansion = expansion
        self.signals = []
        pool = Pool()
        pooltemp = []
        for file in files:
            res = pool.apply_async(
                self.get_unlabeled_signals,
                args=(
                    file,
                    width,
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
        indices = np.random.permutation(len(self.signals))
        self.signals = torch.tensor(self.signals[indices], dtype=torch.float)

    def __len__(self) -> int:
        return len(self.signals) * self.expansion

    def __getitem__(self, index: int) -> list:
        index = index // self.expansion
        signal = self.signals[index]
        if self.transform:
            signal = self.transform(signal)
        return signal

    def get_unlabeled_signals(self, path: str, width: int, channel_names: list = None):
        signal = []
        try:
            record = wfdb.rdrecord(path, channel_names=channel_names)
            for channel in range(record.p_signal.shape[1]):
                p_signal = record.p_signal[:, channel]
                p_signal = processing.resample_sig(
                    p_signal, record.__dict__["fs"], 360
                )[0]
                p_signal = nk.ecg_clean(p_signal, sampling_rate=360)
                _, rpeaks = nk.ecg_peaks(p_signal, sampling_rate=360)
                for i in range(len(rpeaks["ECG_R_Peaks"])):
                    if (
                        rpeaks["ECG_R_Peaks"][i] - width < 0
                        or rpeaks["ECG_R_Peaks"][i] + width > len(p_signal) - 1
                    ):
                        continue
                    start_idx = rpeaks["ECG_R_Peaks"][i] - width
                    end_idx = rpeaks["ECG_R_Peaks"][i] + width
                    sig = p_signal[start_idx:end_idx]
                    sig = processing.normalize_bound(sig, -1, 1)
                    signal.append(sig)
        except:
            print("Unvailable Record: {}".format(path))
        return signal


class ECG_Beat_AF(Dataset):
    def __init__(
        self,
        files: list,
        width: int,
        channel_names: list,
        expansion: int,
        transform: object = None,
        num_class: int = 5,
    ) -> None:
        self.transform = transform
        self.expansion = expansion
        self.num_class = num_class
        if num_class == 5:
            self.LABEL = BEAT_LABEL_5
        elif num_class == 4:
            self.LABEL = BEAT_LABEL_4
        else:
            self.LABEL = BEAT_LABEL_2
        self.signals = []
        self.labels = []
        pool = Pool()
        pooltemp = []
        for file in files:
            res = pool.apply_async(
                self.get_labeled_signals,
                args=(
                    file,
                    width,
                    channel_names,
                ),
            )
            pooltemp.append(res)
        pool.close()
        pool.join()
        for temp in pooltemp:
            signal, label = temp.get()
            self.signals.extend(signal)
            self.labels.extend(label)
        self.signals = np.array(self.signals)
        self.labels = np.array(self.labels)
        indices = np.random.permutation(len(self.signals))
        self.signals = torch.tensor(self.signals[indices], dtype=torch.float)
        self.labels = torch.tensor(self.labels[indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.signals) * self.expansion

    def __getitem__(self, index: int) -> tuple:
        index = index // self.expansion
        signal = self.signals[index]
        label = self.labels[index]
        if self.transform:
            signal = self.transform(signal)
        return signal, label

    def get_labeled_signals(self, path: str, width, channel_names: list = None):
        record = wfdb.rdrecord(path, channel_names=channel_names)
        ann = wfdb.rdann(path, "atr")
        signal = []
        label = []
        try:
            for channel in range(record.p_signal.shape[1]):
                p_signal = record.p_signal[:, channel]
                p_signal = processing.resample_sig(
                    p_signal, record.__dict__["fs"], 360
                )[0]
                p_signal = nk.ecg_clean(p_signal, sampling_rate=360)
                for i in range(len(ann.sample)):
                    if ann.symbol[i] in self.LABEL.keys():
                        annpos = int(ann.sample[i] * 360 / record.__dict__["fs"])
                        if annpos - width < 0 or annpos + width > len(p_signal) - 1:
                            continue
                        start_idx = annpos - width
                        end_idx = annpos + width
                        label.append(self.LABEL[ann.symbol[i]])
                        sig = p_signal[start_idx:end_idx]
                        sig = processing.normalize_bound(sig, -1, 1)
                        signal.append(sig)
        except:
            print("Unvailable Record: {}".format(path))
        return signal, label

    def get_labels(self) -> list:
        return self.labels.tolist()


class ECG_Beat_AU(Dataset):
    def __init__(
        self,
        folders: list,
        width: int,
        channel_names: list,
        expansion: int,
        transform: object = None,
    ) -> None:
        self.transform = transform
        self.expansion = expansion
        self.num_class = len(folders)
        self.signals = []
        self.labels = []
        pool = Pool()
        pooltemp = []
        for folder in folders:
            res = pool.apply_async(
                self.get_labeled_signals,
                args=(
                    folder,
                    width,
                    channel_names,
                ),
            )
            pooltemp.append(res)
        pool.close()
        pool.join()
        for temp in pooltemp:
            signal, label = temp.get()
            self.signals.extend(signal)
            self.labels.extend(label)
        self.signals = np.array(self.signals)
        self.labels = np.array(self.labels)
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        indices = np.random.permutation(len(self.signals))
        self.signals = torch.tensor(self.signals[indices], dtype=torch.float)
        self.labels = torch.tensor(self.labels[indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.signals) * self.expansion

    def __getitem__(self, index: int) -> tuple:
        index = index // self.expansion
        signal = self.signals[index]
        label = self.labels[index]
        if self.transform:
            signal = self.transform(signal)
        return signal, label

    def get_labeled_signals(self, folder: str, width, channel_names: list = None):
        files = np.array(
            [
                os.path.join(folder, file_name.split(".")[0])
                for file_name in os.listdir(folder)
                if file_name.endswith(".hea")
            ]
        )
        signal = []
        label = []
        for file in files:
            try:
                record = wfdb.rdrecord(file, channel_names=channel_names)
                for channel in range(record.p_signal.shape[1]):
                    p_signal = record.p_signal[:, channel]
                    p_signal = processing.resample_sig(
                        p_signal, record.__dict__["fs"], 360
                    )[0]
                    p_signal = nk.ecg_clean(p_signal, sampling_rate=360)
                    _, rpeaks = nk.ecg_peaks(p_signal, sampling_rate=360)
                    for i in range(len(rpeaks["ECG_R_Peaks"])):
                        if (
                            rpeaks["ECG_R_Peaks"][i] - width < 0
                            or rpeaks["ECG_R_Peaks"][i] + width > len(p_signal) - 1
                        ):
                            continue
                        start_idx = rpeaks["ECG_R_Peaks"][i] - width
                        end_idx = rpeaks["ECG_R_Peaks"][i] + width
                        sig = p_signal[start_idx:end_idx]
                        sig = processing.normalize_bound(sig, -1, 1)
                        signal.append(sig)
                        label.append(folder)
            except:
                print("Unvailable Record: {}".format(folder))
        return signal, label

    def get_labels(self) -> list:
        return self.labels.tolist()


class ECG_Beat_DN(Dataset):
    def __init__(
        self,
        folders: list,
        width: int,
        channel_names_wn: list,
        channel_names_won: list,
        expansion: int,
        transform: object = None,
    ) -> None:
        self.transform = transform
        self.expansion = expansion
        self.num_class = len(folders)
        self.signals_wn = []
        self.signals_won = []
        pool = Pool()
        pooltemp = []
        for folder in folders:
            res = pool.apply_async(
                self.get_signals,
                args=(
                    folder,
                    width,
                    channel_names_wn,
                    channel_names_won,
                ),
            )
            pooltemp.append(res)
        pool.close()
        pool.join()
        for temp in pooltemp:
            signal_wn, signal_won = temp.get()
            self.signals_wn.extend(signal_wn)
            self.signals_won.extend(signal_won)
        self.signals_wn = np.array(self.signals_wn)
        self.signals_won = np.array(self.signals_won)
        indices = np.random.permutation(len(self.signals_wn))
        self.signals_wn = torch.tensor(self.signals_wn[indices], dtype=torch.float)
        self.signals_won = torch.tensor(self.signals_won[indices], dtype=torch.float)

    def __len__(self) -> int:
        return len(self.signals_wn) * self.expansion

    def __getitem__(self, index: int) -> tuple:
        index = index // self.expansion
        signal_wn = self.signals_wn[index]
        signal_won = self.signals_won[index]
        if self.transform:
            signal_wn = self.transform(signal_wn)
        return signal_wn, signal_won

    def get_signals(
        self,
        folder: str,
        width,
        channel_names_wn: list = None,
        channel_names_won: list = None,
    ):
        files = np.array(
            [
                os.path.join(folder, file_name.split(".")[0])
                for file_name in os.listdir(folder)
                if file_name.endswith(".hea")
            ]
        )
        signal_wn = []
        signal_won = []
        for file in files:
            try:
                record_won = wfdb.rdrecord(file, channel_names=channel_names_won)
                record_wn = wfdb.rdrecord(file, channel_names=channel_names_wn)
                for channel in range(record_won.p_signal.shape[1]):
                    p_signal_won = record_won.p_signal[:, channel]
                    p_signal_won = processing.resample_sig(
                        p_signal_won, record_won.__dict__["fs"], 360
                    )[0]
                    p_signal_won = nk.ecg_clean(p_signal_won, sampling_rate=360)
                    p_signal_wn = record_wn.p_signal[:, channel]
                    p_signal_wn = processing.resample_sig(
                        p_signal_wn, record_wn.__dict__["fs"], 360
                    )[0]
                    _, rpeaks = nk.ecg_peaks(p_signal_won, sampling_rate=360)
                    for i in range(len(rpeaks["ECG_R_Peaks"])):
                        if (
                            rpeaks["ECG_R_Peaks"][i] - width < 0
                            or rpeaks["ECG_R_Peaks"][i] + width > len(p_signal_won) - 1
                        ):
                            continue
                        start_idx = rpeaks["ECG_R_Peaks"][i] - width
                        end_idx = rpeaks["ECG_R_Peaks"][i] + width
                        sig_won = p_signal_won[start_idx:end_idx]
                        sig_won = processing.normalize_bound(sig_won, -1, 1)
                        sig_wn = p_signal_wn[start_idx:end_idx]
                        sig_wn = processing.normalize_bound(sig_wn, -1, 1)
                        signal_won.append(sig_won)
                        signal_wn.append(sig_wn)
            except:
                print("Unvailable Record: {}".format(folder))
        return signal_wn, signal_won
