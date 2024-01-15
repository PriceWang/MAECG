"""
Author: Guoxin Wang
Date: 2022-10-27 13:45:59
LastEditors: Guoxin Wang
LastEditTime: 2024-01-15 14:48:07
FilePath: /mae/dataprocess.py
Description: 

Copyright (c) 2022 by Guoxin Wang, All Rights Reserved. 
"""
import torch
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import random_split
from utils.data_utils import *

parser = argparse.ArgumentParser(description="Data Processing")
parser.add_argument(
    "--task",
    required=True,
    type=str,
    choices=["ul_beat", "af_beat", "af_beat_intra", "au_beat", "dn_beat"],
    help="target task",
)
parser.add_argument(
    "--data_path",
    default="../storage/ssd/public/unsupervisedecg/physionet.org/files/mitdb/1.0.0",
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
    default="../storage/ssd/public/guoxin/mitdb",
    type=str,
    help="path where to save",
)
parser.add_argument(
    "--width",
    default=240,
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
    "--numclasses",
    default=5,
    type=int,
    help="number of classes",
)
parser.add_argument("--expansion", default=1, type=int, help="expansion factor")
parser.add_argument("--mitdb", action="store_true", help="special for mitdb")
parser.set_defaults(mitdb=False)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def ul_beat() -> None:
    logger.info("Data Generating for Task {}".format(args.task))
    files = np.array(
        [
            os.path.join(path, file_name.split(".")[0])
            for path, _, file_list in os.walk(args.data_path)
            for file_name in file_list
            if file_name.endswith(".hea")
        ]
    )
    np.random.shuffle(files)
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
    logger.info("Data Generating for Task {}".format(args.task))
    # Uses DS1 DS2 as training set and testing set for MITDB
    if args.mitdb:
        DS1 = [
            101,
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
            numclasses=args.numclasses,
        )
        valid_size = int(0.05 * len(dataset))
        [train_set, valid_set] = random_split(
            dataset, [len(dataset) - valid_size, valid_size]
        )
        torch.save(
            train_set,
            "{}/{}_train.pth".format(args.output_dir, args.task),
        )
        torch.save(
            valid_set,
            "{}/{}_valid.pth".format(args.output_dir, args.task),
        )
        test_set = ECG_Beat_AF(
            files=test_files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            numclasses=args.numclasses,
        )
        torch.save(
            test_set,
            "{}/{}_test.pth".format(args.output_dir, args.task),
        )
    else:
        files = np.array(
            [
                os.path.join(args.data_path, file_name.split(".")[0])
                for file_name in os.listdir(args.data_path)
                if file_name.endswith(".hea")
            ]
        )
        test_set = ECG_Beat_AF(
            files=files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            numclasses=args.numclasses,
        )
        torch.save(
            test_set,
            "{}/{}.pth".format(args.output_dir, args.task),
        )


def af_beat_intra() -> None:
    logger.info("Data Generating for Task {}".format(args.task))
    if args.mitdb:
        DS = [
            101,
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
        files = np.array(
            [
                os.path.join(
                    args.data_path,
                    str(file_name),
                )
                for file_name in DS
            ]
        )
        dataset = ECG_Beat_AF(
            files=files,
            width=args.width,
            channel_names=args.channel_names,
            expansion=args.expansion,
            numclasses=args.numclasses,
        )
        test_size = int(0.3 * len(dataset))
        [train_set, valid_set, test_set] = random_split(
            dataset, [len(dataset) - test_size * 2, test_size, test_size]
        )
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


def au_beat() -> None:
    logger.info("Data Generating for Task {}".format(args.task))
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
    [train_set, valid_set, test_set] = random_split(
        dataset, [train_size, valid_size, len(dataset) - train_size - valid_size]
    )
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
    logger.info("Data Generating for Task {}".format(args.task))
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
    [train_set, test_set] = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    torch.save(
        train_set,
        "{}/{}_train.pth".format(args.output_dir, args.task),
    )
    torch.save(
        test_set,
        "{}/{}_test.pth".format(args.output_dir, args.task),
    )


def main() -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.task == "ul_beat":
        ul_beat()
    elif args.task == "af_beat":
        af_beat()
    elif args.task == "af_beat_intra":
        af_beat_intra()
    elif args.task == "au_beat":
        au_beat()
    elif args.task == "dn_beat":
        dn_beat()


if __name__ == "__main__":
    main()
