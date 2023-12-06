###
# @Author: Guoxin Wang
# @Date: 2023-07-05 16:39:56
 # @LastEditors: Guoxin Wang
 # @LastEditTime: 2023-11-15 11:48:00
 # @FilePath: /mae/evaluation.sh
# @Description:
#
# Copyright (c) 2023 by Guoxin Wang, All Rights Reserved.
###

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
#     --model vit_small_p32 \
#     --batch_size 1024 \
#     --resume /home/guoxin/mae/finetunes/af_beat/output/vit_small_p32_m50_ep200_bs1024/checkpoint-max_acc.pth \
#     --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_test.pth \
#     --num_workers 20 \
#     --eval

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
#     --model vit_small_p32 \
#     --batch_size 1024 \
#     --resume /home/guoxin/mae/finetunes/af_beat_intra/output/vit_small_p32_m50_ep200_bs1024/checkpoint-max_acc.pth \
#     --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_intra_test.pth \
#     --num_workers 20 \
#     --eval

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
#     --model vit_small_p32 \
#     --batch_size 1024 \
#     --resume /home/guoxin/mae/finetunes/au_beat/output/vit_small_p32_m50_ep200_bs32/checkpoint-max_acc.pth \
#     --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_test.pth \
#     --num_workers 20 \
#     --eval

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_small_p32 \
    --batch_size 1024 \
    --resume /home/guoxin/mae/finetunes/af_beat/output/vit_small_p32_m50_ep200_bs1024/checkpoint-max_acc.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_test.pth \
    --num_workers 20 \
    --eval

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_small_p32 \
    --batch_size 1024 \
    --resume /home/guoxin/mae/finetunes/af_beat_intra/output/vit_small_p32_m50_ep200_bs1024/checkpoint-max_acc.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_intra_test.pth \
    --num_workers 20 \
    --eval

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_small_p32 \
    --batch_size 1024 \
    --resume /home/guoxin/mae/finetunes/au_beat/output/vit_small_p32_m50_ep200_bs32/checkpoint-max_acc.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_test.pth \
    --num_workers 20 \
    --eval
