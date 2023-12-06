###
# @Author: Guoxin Wang
# @Date: 2023-07-05 16:33:12
 # @LastEditors: Guoxin Wang
 # @LastEditTime: 2023-09-05 16:14:54
 # @FilePath: /mae/supervised.sh
# @Description:
#
# Copyright (c) 2023 by Guoxin Wang, All Rights Reserved.
###

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p4 \
    --batch_size 256 \
    --accum_iter 4\
    --lr 3e-4 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/af_beat/output/vit_base_p4_ep200_bs1024 \
    --log_dir ./superviseds/af_beat/log/vit_base_p4_ep200_bs1024

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p8 \
    --batch_size 256 \
    --accum_iter 4\
    --lr 3e-4 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/af_beat/output/vit_base_p8_ep200_bs1024 \
    --log_dir ./superviseds/af_beat/log/vit_base_p8_ep200_bs1024

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p16 \
    --batch_size 256 \
    --accum_iter 4\
    --lr 3e-4 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/af_beat/output/vit_base_p16_ep200_bs1024 \
    --log_dir ./superviseds/af_beat/log/vit_base_p16_ep200_bs1024

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 256 \
    --accum_iter 4\
    --lr 3e-4 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/mitdb/af_beat_valid.pth \
    --extra_train /home/guoxin/storage/ssd/public/guoxin/incartdb/af_beat.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/af_beat/output/vit_base_p32_ep200_bs1024 \
    --log_dir ./superviseds/af_beat/log/vit_base_p32_ep200_bs1024

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p4 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_valid.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/au_beat/output/vit_base_p4_ep200_bs32 \
    --log_dir ./superviseds/au_beat/log/vit_base_p4_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p8 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_valid.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/au_beat/output/vit_base_p8_ep200_bs32 \
    --log_dir ./superviseds/au_beat/log/vit_base_p8_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p16 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_valid.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/au_beat/output/vit_base_p16_ep200_bs32 \
    --log_dir ./superviseds/au_beat/log/vit_base_p16_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_train.pth \
    --test_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/au_beat_valid.pth \
    --mixup 0.5 \
    --output_dir ./superviseds/au_beat/output/vit_base_p32_ep200_bs32 \
    --log_dir ./superviseds/au_beat/log/vit_base_p32_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune_dn.py \
    --model mae_vit_base_patch4 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/dn_beat_train.pth \
    --output_dir ./superviseds/dn_beat/output/mae_vit_base_p4_ep200_bs32 \
    --log_dir ./superviseds/dn_beat/log/mae_vit_base_p4_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune_dn.py \
    --model mae_vit_base_patch8 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/dn_beat_train.pth \
    --output_dir ./superviseds/dn_beat/output/mae_vit_base_p8_ep200_bs32 \
    --log_dir ./superviseds/dn_beat/log/mae_vit_base_p8_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune_dn.py \
    --model mae_vit_base_patch16 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/dn_beat_train.pth \
    --output_dir ./superviseds/dn_beat/output/mae_vit_base_p16_ep200_bs32 \
    --log_dir ./superviseds/dn_beat/log/mae_vit_base_p16_ep200_bs32

OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune_dn.py \
    --model mae_vit_base_patch32 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --train_path /home/guoxin/storage/ssd/public/guoxin/ecgiddb/dn_beat_train.pth \
    --output_dir ./superviseds/dn_beat/output/mae_vit_base_p32_ep200_bs32 \
    --log_dir ./superviseds/dn_beat/log/mae_vit_base_p32_ep200_bs32
