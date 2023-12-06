###
# @Author: Guoxin Wang
# @Date: 2023-07-05 16:33:12
# @LastEditors: Guoxin Wang
# @LastEditTime: 2023-12-06 00:38:04
# @FilePath: /mae/finetune.sh
# @Description:
#
# Copyright (c) 2023 by Guoxin Wang, All Rights Reserved.
###

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
#     --model vit_base_p2 \
#     --batch_size 1024 \
#     --lr 3e-4 \
#     --epochs 200 \
#     --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p2_m50_ep500/checkpoint-299.pth \
#     --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
#     --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
#     --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
#     --mixup 0.5 \
#     --output_dir ./finetunes/af_beat/output/vit_base_p2_m50_ep200_bs1024 \
#     --log_dir ./finetunes/af_beat/log/vit_base_p2_m50_ep200_bs1024

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
#     --model vit_base_p2 \
#     --batch_size 1024 \
#     --lr 3e-4 \
#     --epochs 200 \
#     --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p2_m50_ep500/checkpoint-299.pth \
#     --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_intra_train.pth \
#     --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_intra_valid.pth \
#     --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
#     --mixup 0.5 \
#     --output_dir ./finetunes/af_beat_intra/output/vit_base_p2_m50_ep200_bs1024 \
#     --log_dir ./finetunes/af_beat_intra/log/vit_base_p2_m50_ep200_bs1024

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
#     --model vit_base_p2 \
#     --batch_size 32 \
#     --lr 1e-3 \
#     --epochs 200 \
#     --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p2_m50_ep500/checkpoint-299.pth \
#     --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_train.pth \
#     --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_valid.pth \
#     --mixup 0.5 \
#     --output_dir ./finetunes/au_beat/output/vit_base_p2_m50_ep200_bs32 \
#     --log_dir ./finetunes/au_beat/log/vit_base_p2_m50_ep200_bs32

# OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune_dn.py \
#     --model mae_vit_base_patch2 \
#     --batch_size 32 \
#     --lr 1e-3 \
#     --epochs 200 \
#     --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p2_m50_ep500/checkpoint-299.pth \
#     --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/dn_beat_train.pth \
#     --output_dir ./finetunes/dn_beat/output/mae_vit_base_p2_m50_ep200_bs32 \
#     --log_dir ./finetunes/dn_beat/log/mae_vit_base_p2_m50_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 1024 \
    --lr 3e-4 \
    --epochs 200 \
    --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p32_m50_ep500/checkpoint-499.pth \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --mixup 0.5 \
    --output_dir ./finetunes/test/output/wmix_woextra \
    --log_dir ./finetunes/test/log/wmix_woextra

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 1024 \
    --lr 3e-4 \
    --epochs 200 \
    --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p32_m50_ep500/checkpoint-499.pth \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
    --output_dir ./finetunes/test/output/womix_wextra \
    --log_dir ./finetunes/test/log/womix_wextra

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 1024 \
    --lr 3e-4 \
    --epochs 200 \
    --finetune /home/guoxin/mae/pretrains/output/mae_vit_base_p32_m50_ep500/checkpoint-499.pth \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --output_dir ./finetunes/test/output/womix_woextra \
    --log_dir ./finetunes/test/log/womix_woextra
