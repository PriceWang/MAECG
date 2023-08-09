###
# @Author: Guoxin Wang
# @Date: 2023-07-01 16:45:15
# @LastEditors: Guoxin Wang
# @LastEditTime: 2023-08-03 15:51:32
# @FilePath: /mae/pretrain.sh
# @Description:
#
# Copyright (c) 2023 by Guoxin Wang, All Rights Reserved.
###

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_pretrain.py \
    --model mae_vit_tiny_patch4 \
    --batch_size 1024 \
    --mask_ratio 0.6 \
    --lr 1e-3 \
    --data_path /home/guoxin/storage/ssd/public/guoxin/challenge-2021 \
    --output_dir ./pretrains/output/mae_vit_tiny_p4_m60_ep200 \
    --log_dir ./pretrains/log/mae_vit_tiny_p4_m60_ep200
