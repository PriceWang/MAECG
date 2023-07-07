"""
Author: Guoxin Wang
Date: 2022-04-29 14:54:52
LastEditors: Guoxin Wang
LastEditTime: 2023-07-03 16:08:08
FilePath: /mae/util/data_utils.py
Description: 

Copyright (c) 2022 by Guoxin Wang, All Rights Reserved. 
"""
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


class Mixup(object):
    def __init__(self, alpha: float = 0.1, device: str = "cpu") -> None:
        self.alpha = alpha
        self.device = device

    def __call__(self, data: torch.Tensor, target: torch.Tensor) -> tuple:
        gamma = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(data.size(0), device=self.device, dtype=torch.long)
        return self.partial_mixup(data, gamma, indices), self.partial_mixup(
            target, gamma, indices
        )

    def partial_mixup(
        self, input: torch.Tensor, gamma: float, indices: torch.Tensor
    ) -> torch.Tensor:
        if input.size(0) != indices.size(0):
            raise RuntimeError("Size mismatch!")
        perm_input = input[indices]
        return input.mul(gamma).add(perm_input, alpha=1 - gamma)


# def random_masking(x, mask_size, mask_ratio):
#     """
#     Perform per-sample random masking by per-sample shuffling.
#     Per-sample shuffling is done by argsort random noise.
#     x: [N, L, D], sequence
#     """
#     L = x.shape[0]  # batch, length, dim
#     len_keep = int(L * (1 - mask_ratio))

#     noise = 0.05 * torch.rand(L, device=x.device)  # noise in [0, 1]

#     # sort noise for each sample
#     ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
#     ids_restore = torch.argsort(ids_shuffle)

#     # keep the first subset
#     ids_keep = ids_shuffle[:len_keep]
#     x_masked = torch.gather(x, dim=0, index=ids_keep.repeat(1))

#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones(L, device=x.device)
#     mask[:len_keep] = 0
#     # unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=0, index=ids_restore)

#     return x_masked, mask, ids_restore


class RandomCrop(object):
    def __init__(self, output_size: int = 64, p: float = 0.5) -> None:
        self.output_size = output_size
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        size = data.size()
        if torch.rand(1) < self.p:
            rand = torch.randint(0, int(len(data) - self.output_size), (1,)).item()
            data = data[rand : int(rand + self.output_size)].view(size)
        return data


class RandomScale(object):
    def __init__(self, scale_size: int = 1.5, p: float = 0.5) -> None:
        self.scale_size = scale_size
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            data = data * self.scale_size
        return data


class RandomFlip(object):
    def __init__(self, p: float = 0.5, dim: int = 0) -> None:
        self.p = p
        self.dim = dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            if self.dim == 0:
                data = -data
            else:
                data = torch.flip(data, dims=(0,))
        return data


class RandomCutout(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            data = (data + torch.abs(data)) / 2
        return data


class RandomShift(object):
    def __init__(self, shift_size: int = 64, p: float = 0.5) -> None:
        self.shift_size = shift_size
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.shift_size:
            if torch.rand(1) < self.p:
                data = torch.roll(data, self.shift_size)
        return data


class RandomSine(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            data = torch.sin(data)
        return data


class RandomSquare(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            data = torch.square(data)
        return data


class RandomNoise(object):
    def __init__(
        self, p: float = 1, mean: float = 0, std: float = 0.05, device: str = "cpu"
    ) -> None:
        self.p = p
        self.mean = mean
        self.std = std
        self.device = device

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            data = data + torch.normal(
                mean=self.mean, std=self.std, size=data.size(), device=self.device
            )
        return data


class ECG_Beat_L(Dataset):
    def __init__(
        self,
        files: list,
        width: int,
        channel_names: list,
        expansion: int,
        transform: object = None,
        numclasses: int = 5,
    ) -> None:
        self.transform = transform
        self.expansion = expansion
        if numclasses == 5:
            self.LABEL = BEAT_LABEL_5
        elif numclasses == 4:
            self.LABEL = BEAT_LABEL_4
        else:
            self.LABEL = BEAT_LABEL_2
        self.datas = []
        self.labels = []
        pool = Pool()
        pooltemp = []
        for file in files:
            res = pool.apply_async(
                self.get_labeled_datas,
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
            data, label = temp.get()
            self.datas.extend(data)
            self.labels.extend(label)
        self.datas = np.array(self.datas)
        self.labels = np.array(self.labels)
        self.datas = torch.tensor(
            self.datas, dtype=torch.float
        )
        self.labels = torch.tensor(
            self.labels, dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.datas) * self.expansion

    def __getitem__(self, index: int) -> tuple:
        index = index // self.expansion
        data = self.datas[index]
        label = self.labels[index]
        if self.transform:
            data = self.transform(data)
        data = data.unsqueeze(0)
        return data, label

    def get_labeled_datas(self, path: str, width, channel_names: list = None):
        record = wfdb.rdrecord(path, channel_names=channel_names)
        ann = wfdb.rdann(path, "atr")
        label = []
        data = []
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
                        data.append(sig)
        except:
            print("Unvailable Record: {}".format(path))
        return data, label

    def get_labels(self) -> list:
        return self.labels.tolist()


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
        self.datas = []
        pool = Pool()
        pooltemp = []
        for file in files:
            res = pool.apply_async(
                self.get_unlabeled_datas,
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
            data = temp.get()
            self.datas.extend(data)
        self.datas = np.array(self.datas)
        self.datas = torch.tensor(
            self.datas, dtype=torch.float
        )

    def __len__(self) -> int:
        return len(self.datas) * self.expansion

    def __getitem__(self, index: int) -> list:
        index = index // self.expansion
        data = self.datas[index]
        if self.transform:
            data = [self.transform(data), self.transform(data)]
        else:
            data = [data, data]
        data[0] = data[0].unsqueeze(0)
        data[1] = data[1].unsqueeze(0)
        return data

    def get_unlabeled_datas(self, path: str, width: int, channel_names: list = None):
        data = []
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
                    data.append(sig)
        except:
            print("Unvailable Record: {}".format(path))
        return data


class ImbalancedSampler(Sampler):
    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
    ) -> None:
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        label = self._get_labels(dataset) if labels is None else labels
        weights = [1.0 / Counter(label)[i] for i in label]
        self.weights = torch.DoubleTensor(weights)

    def _get_labels(self, dataset: Dataset) -> list:
        if isinstance(dataset, ConcatDataset):
            return np.concatenate([ds.get_labels() for ds in dataset.datasets])
        elif isinstance(dataset, Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self) -> list:
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self) -> int:
        return self.num_samples
