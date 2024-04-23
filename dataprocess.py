"""
Author: Guoxin Wang
Date: 2022-10-27 13:45:59
LastEditors: Guoxin Wang
LastEditTime: 2024-04-23 08:34:42
FilePath: /maecg/dataprocess.py
Description: 

Copyright (c) 2022 by Guoxin Wang, All Rights Reserved. 
"""

import argparse
import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch

from utils.data_utils import ECG_Beat_AF, ECG_Beat_AU, ECG_Beat_DN, ECG_Beat_UL

parser = argparse.ArgumentParser(description="Data Processing")
parser.add_argument(
    "--task",
    required=True,
    type=str,
    choices=["ul_beat", "af_beat", "au_beat", "dn_beat"],
    help="target task",
)
parser.add_argument(
    "--data_path",
    default="..",
    type=str,
    help="path of original dataset",
)
parser.add_argument(
    "--prefix",
    default="",
    type=str,
    help="prefix of datapath",
)
parser.add_argument(
    "--output_dir",
    default="..",
    type=str,
    help="path where to save",
)
parser.add_argument(
    "--width",
    default=512,
    type=int,
    help="half width",
)
parser.add_argument(
    "--channel_names",
    default=None,
    action="append",
    help="list of channels to use",
)
parser.add_argument(
    "--channel_names_wn",
    default=None,
    action="append",
    help="list of channels to use (with noise)",
)
parser.add_argument(
    "--num_class",
    default=5,
    type=int,
    help="number of classes",
)
parser.add_argument("--expansion", default=1, type=int, help="expansion factor")
parser.add_argument(
    "--inter", default=False, action="store_true", help="inter-patient for mitdb"
)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def ul_beat() -> None:
    files = np.array(
        [
            os.path.join(path, file_name.split(".")[0])
            for path, _, file_list in os.walk(args.data_path)
            for file_name in file_list
            if file_name.endswith(".hea")
        ]
    )
    unlabeled_set = ECG_Beat_UL(
        files=files,
        width=args.width,
        channel_names=args.channel_names,
        expansion=args.expansion,
    )
    torch.save(
        unlabeled_set,
        "{}/{}.pth".format(args.output_dir, args.task),
    )


def af_beat() -> None:
    # Uses DS1 DS2 as training set and testing set for MITDB
    if args.inter:
        DS1 = [
            101,
            106,
            108,
            109,
            112,
            114,
            115,
            116,
            118,
            119,
            122,
            124,
            201,
            203,
            205,
            207,
            208,
            209,
            215,
            220,
            223,
            230,
        ]
        DS2 = [
            100,
            103,
            105,
            111,
            113,
            117,
            121,
            123,
            200,
            202,
            210,
            212,
            213,
            214,
            219,
            221,
            222,
            228,
            231,
            232,
            233,
            234,
        ]
        train_files = np.array(
            [
                os.path.join(
                    args.data_path,
                    str(file_name),
                )
                for file_name in DS1
            ]
        )
        test_files = np.array(
            [
                os.path.join(
                    args.data_path,
                    str(file_name),
                )
                for file_name in DS2
            ]
        )
        dataset = ECG_Beat_AF(
            files=train_files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            num_class=args.num_class,
        )
        train_size = int(0.9 * len(dataset))
        train_set = copy.deepcopy(dataset)
        valid_set = copy.deepcopy(dataset)
        train_set.signals = dataset.signals[:train_size]
        train_set.labels = dataset.labels[:train_size]
        valid_set.signals = dataset.signals[train_size:]
        valid_set.labels = dataset.labels[train_size:]
        torch.save(
            train_set,
            "{}/{}_{}_train.pth".format(args.output_dir, args.task, args.num_class),
        )
        torch.save(
            valid_set,
            "{}/{}_{}_valid.pth".format(args.output_dir, args.task, args.num_class),
        )
        test_set = ECG_Beat_AF(
            files=test_files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            num_class=args.num_class,
        )
        torch.save(
            test_set,
            "{}/{}_{}_test.pth".format(args.output_dir, args.task, args.num_class),
        )
    else:
        files = np.array(
            [
                os.path.join(args.data_path, file_name.split(".")[0])
                for file_name in os.listdir(args.data_path)
                if file_name.endswith(".hea")
            ]
        )
        dataset = ECG_Beat_AF(
            files=files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            num_class=args.num_class,
        )
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))
        train_set = copy.deepcopy(dataset)
        valid_set = copy.deepcopy(dataset)
        test_set = copy.deepcopy(dataset)
        train_set.signals = dataset.signals[:train_size]
        train_set.labels = dataset.labels[:train_size]
        valid_set.signals = dataset.signals[train_size : train_size + valid_size]
        valid_set.labels = dataset.labels[train_size : train_size + valid_size]
        test_set.signals = dataset.signals[train_size + valid_size :]
        test_set.labels = dataset.labels[train_size + valid_size :]
        torch.save(
            train_set,
            "{}/{}_{}_train.pth".format(args.output_dir, args.task, args.num_class),
        )
        torch.save(
            valid_set,
            "{}/{}_{}_valid.pth".format(args.output_dir, args.task, args.num_class),
        )
        torch.save(
            test_set,
            "{}/{}_{}_test.pth".format(args.output_dir, args.task, args.num_class),
        )


def au_beat() -> None:
    folders = [
        os.path.join(args.data_path, folder)
        for folder in os.listdir(args.data_path)
        if folder.startswith(args.prefix)
    ]
    dataset = ECG_Beat_AU(
        folders=folders,
        width=args.width,
        channel_names=args.channel_names,
        expansion=args.expansion,
    )
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    train_set = copy.deepcopy(dataset)
    valid_set = copy.deepcopy(dataset)
    test_set = copy.deepcopy(dataset)
    train_set.signals = dataset.signals[:train_size]
    train_set.labels = dataset.labels[:train_size]
    valid_set.signals = dataset.signals[train_size : train_size + valid_size]
    valid_set.labels = dataset.labels[train_size : train_size + valid_size]
    test_set.signals = dataset.signals[train_size + valid_size :]
    test_set.labels = dataset.labels[train_size + valid_size :]
    torch.save(
        train_set,
        "{}/{}_train.pth".format(args.output_dir, args.task),
    )
    torch.save(
        valid_set,
        "{}/{}_valid.pth".format(args.output_dir, args.task),
    )
    torch.save(
        test_set,
        "{}/{}_test.pth".format(args.output_dir, args.task),
    )


def dn_beat() -> None:
    folders = [
        os.path.join(args.data_path, folder)
        for folder in os.listdir(args.data_path)
        if folder.startswith(args.prefix)
    ]
    dataset = ECG_Beat_DN(
        folders=folders,
        width=args.width,
        channel_names_wn=args.channel_names_wn,
        channel_names_won=args.channel_names,
        expansion=args.expansion,
    )
    train_size = int(0.7 * len(dataset))
    train_set = copy.deepcopy(dataset)
    test_set = copy.deepcopy(dataset)
    train_set.signals_wn = dataset.signals_wn[:train_size]
    train_set.signals_won = dataset.signals_won[:train_size]
    test_set.signals_wn = dataset.signals_wn[train_size:]
    test_set.signals_won = dataset.signals_won[train_size:]
    torch.save(
        train_set,
        "{}/{}_train.pth".format(args.output_dir, args.task),
    )
    torch.save(
        test_set,
        "{}/{}_test.pth".format(args.output_dir, args.task),
    )


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Data Generating for Task {}".format(args.task))
    if args.task == "ul_beat":
        ul_beat()
    elif args.task == "af_beat":
        af_beat()
    elif args.task == "au_beat":
        au_beat()
    elif args.task == "dn_beat":
        dn_beat()


if __name__ == "__main__":
    main()
