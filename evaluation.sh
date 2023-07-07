OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
--model vit_tiny_p4 \
--batch_size 1024 \
--finetune /home/guoxin/mae/pretrains/output/mae_vit_tiny_patch4_mask60/checkpoint-199.pth \
--output_dir ./finetunes/output/vit_tiny_p4_incart \
--log_dir ./finetunes/log/vit_tiny_p4_incart \
--resume /home/guoxin/mae/finetunes/output/vit_tiny_p4_incart/checkpoint-max_acc.pth \
--num_workers 20 \
--eval \

