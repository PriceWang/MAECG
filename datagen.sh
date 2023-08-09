###
# @Author: Guoxin Wang
# @Date: 2023-07-07 16:52:35
 # @LastEditors: Guoxin Wang
 # @LastEditTime: 2023-07-23 17:39:10
 # @FilePath: /mae/datagen.sh
# @Description:
#
# Copyright (c) 2023 by Guoxin Wang, All Rights Reserved.
###

# python dataprocess.py \
#     --task af_beat \
#     --data_path ../storage/ssd/public/unsupervisedecg/physionet.org/files/mitdb/1.0.0 \
#     --output_dir ../storage/ssd/public/guoxin/mitdb \
#     --width 240 \
#     --channel_names MLII \
#     --numclasses 5 \
#     --expansion 1 \
#     --mitdb

# python dataprocess.py \
#     --task af_beat \
#     --data_path ../storage/ssd/public/unsupervisedecg/physionet.org/files/incartdb/1.0.0 \
#     --output_dir ../storage/ssd/public/guoxin/incartdb \
#     --width 240 \
#     --channel_names II \
#     --numclasses 5 \
#     --expansion 1

# python dataprocess.py \
#     --task ul_beat \
#     --data_path ../storage/ssd/public/unsupervisedecg/physionet.org/files/challenge-2021/1.0.0 \
#     --output_dir ../storage/ssd/public/guoxin/challenge-2021 \
#     --width 240 \
#     --channel_names II \
#     --expansion 1

# python dataprocess.py \
#     --task au_beat \
#     --data_path ../storage/ssd/public/unsupervisedecg/physionet.org/files/ecgiddb/1.0.0 \
#     --prefix Person \
#     --output_dir ../storage/ssd/public/guoxin/ecgiddb \
#     --width 240 \
#     --channel_names "ECG I filtered" \
#     --expansion 1

python dataprocess.py \
    --task dn_beat \
    --data_path ../storage/ssd/public/unsupervisedecg/physionet.org/files/ecgiddb/1.0.0 \
    --prefix Person \
    --output_dir ../storage/ssd/public/guoxin/ecgiddb \
    --width 240 \
    --channel_names "ECG I filtered" \
    --channel_names_wn "ECG I" \
    --expansion 1
