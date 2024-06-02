<!--
 * @Author: Guoxin Wang
 * @Date: 2024-01-11 16:50:18
 * @LastEditors: Guoxin Wang
 * @LastEditTime: 2024-06-02 06:28:22
 * @FilePath: /guoxin/maecg/README.md
 * @Description:
 *
 * Copyright (c) 2024 by Guoxin Wang, All Rights Reserved.
-->

## MAECG: A PyTorch Implementation

<p align="center">
  <img src="https://github.com/PriceWang/MAECG/assets/30796250/48680d87-065e-4708-bb3b-3b1dc1835868" width="1080">
</p>

This is a PyTorch/GPU implementation of the paper [A Task-Generic High-Performance Unsupervised Pre-training Framework for ECG](https://ieeexplore.ieee.org/document/10541906):

```
@article{wang2024task,
  title={A Task-Generic High-Performance Unsupervised Pre-training Framework for ECG},
  author={Wang, Guoxin and Wang, Qingyuan and Nag, Avishek and John, Deepu},
  journal={IEEE Sensors Journal},
  year={2024},
  publisher={IEEE}
}
```

This work is inspired by [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) and this repo is inspired by the [MAE repo](https://github.com/facebookresearch/mae).

### Catalog

- [x] Pre-training code
- [x] Fine-tuning code
- [x] Visualization demo

### Requirement

Install the required package:

```
conda env create -n maecg --file environment.yml
```

Activate environment:

```
conda activate maecg
```

### Data Generation

To generate unlabelled ECG datasets, run the following command:

```
python dataprocess.py \
    --task ul_beat \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --width 240 \
    --channel_names ${channel_names} \
    --expansion 1
```

- Here we creat samples of length: `width` \* 2.
- `expansion` is a simple replication to extend dataset.

To generate labelled ECG datasets, run the following command:

```
python dataprocess.py \
    --task ${task} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --width 240 \
    --channel_names ${channel_names} \
    --num_class 5 \
    --expansion 1
```

- Choose `task` from "af_beat", "au_beat" and "dn_beat".
- Set `--prefix ${prefix}` when original data path is nested.
- Set `--channel_names_wn ${channel_names_wn}` for denoising/decoding.
- Set `--num_class 2` or `--num_class 4` for different classifications.
- Set `--inter` to generate datasets from MITDB with special splits.

### Pre-training

To pre-train ViT-Base (recommended default) with multi-node distributed training, run the following on 1 node with 2 GPUs each:

```
OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=2 main_pretrain.py \
    --model mae_vit_base_patch32 \
    --batch_size 1024 \
    --epochs 200 \
    --accum_iter 1 \
    --mask_ratio 0.5 \
    --lr 3e-4 \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --log_dir ${log_dir} \
    --model_ema
```

- Here the effective batch size is 1024 (`batch_size` per gpu) \* 1 (nodes) \* 2 (gpus per node) \* 1 (`accum_iter`) = 2048.
- To train ViT-Atto, ViT-Tiny, ViT-Small, ViT-Large or ViT-Huge with different patch size, set `--model mae_vit_${model_size}_patch${patch_size}`.
- Set `mask_ratio` for mask ratio.
- Set `--data_path ${data_path_1} ${data_path_2} ...` to pre-train with multiple datasets
- See [MAE pre-training](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md) for detailed parameter setting.
- To speed up training, turn on automatic mixed precision (`torch.cuda.amp`). But there is a chance of producing NaN when pre-training ViT-Large/ViT-Huge in GPUs.
- Training time is ~11h in 2 A100 GPUs (200 epochs).

### Fine-tuning with pre-trained checkpoints

#### Fine-tuning

Currently, we implemented fine-tuning by freezing encoder.

To fine-tune with multi-node distributed training, run the following command:

```
OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 1024 \
    --lr 3e-4 \
    --epochs 200 \
    --finetune ${pretrain_ckpt} \
    --train_path ${train_path}\
    --test_path ${test_path} \
    --output_dir ${output_dir} \
    --log_dir ${log_dir} \
    --linear \
    --model_ema \
    --save_best
```

- Here the effective batch size is 1024 (`batch_size` per gpu) \* 1 (node) \* 1 (gpus per node) = 1024.
- Set `--train_path ${train_path_1} ${train_path_2} ...` to fine-tune with multiple datasets
- See [MAE fine-tuning](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) for detailed parameter setting.
- Training time is ~57m in 1 RTX3090 GPU.

Script for human identification:

```
OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune.py \
    --model vit_base_p32 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --finetune ${pretrain_ckpt} \
    --train_path ${train_path}\
    --test_path ${test_path} \
    --mixup 0.5 \
    --output_dir ${output_dir} \
    --log_dir ${log_dir} \
    --linear \
    --model_ema \
    --save_best
```

- Here the effective batch size is 32 (`batch_size` per gpu) \* 1 (node) \* 1 (gpus per node) = 32.
- Training time is ~10m in 1 RTX3090 GPU.

Script for denoising:

```
OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=1 main_finetune_de.py \
    --model mae_vit_base_patch32 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 200 \
    --finetune ${pretrain_ckpt} \
    --train_path ${train_path}\
    --output_dir ${output_dir} \
    --log_dir ${log_dir} \
    --linear \
    --model_ema
```

- Training time is ~23m in 1 RTX3090 GPU.

#### Evaluation

Evaluate arrhythmia classification on MITDB-DS1 (train + valid) and INCARTDB in a single GPU:

```
python main_finetune.py \
    --model vit_base_p32 \
    --resume ${finetune_ckpt} \
    --test_path ${MITDB_train_path} \
    ${MITDB_valid_path} \
    ${INCARTDB_path} \
    --eval
```

Evaluate human identification on ECGIDDB (train + valid):

```
python main_finetune.py \
    --model vit_base_p32 \
    --resume ${finetune_ckpt} \
    --test_path ${ECGIDDB_train_path} \
    ${ECGIDDB_valid_path} \
    --eval
```

### Results

By fine-tuning these pre-trained models, we rank #1 Acc in these tasks:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td colspan="7"><font size="1"><em>following are the results of different models, a patch size of 32 and a mask ratio of 0.5 for the pre-trained MAECG:</em></font></td>
</tr>
<th style="text-align:center"></th>
<th style="text-align:center">ViT-Atto</th>
<th style="text-align:center">ViT-Tiny</th>
<th style="text-align:center">ViT-Small</th>
<th style="text-align:center">ViT-Base</th>
<th style="text-align:center">ViT-Large</th>
<th style="text-align:center">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">MITDB-DS2</td>
<td align="center">93.3</td>
<td align="center">93.4</td>
<td align="center">94.8</td>
<td align="center"><b>95.6</b></td>
<td align="center">95.4</td>
<td align="center">95.4</td>
</tr>
<tr><td align="left">ECGIDDB</td>
<td align="center">94.1</td>
<td align="center">97.1</td>
<td align="center">98.7</td>
<td align="center"><b>98.8</b></td>
<td align="center">98.5</td>
<td align="center">98.3</td>
</tr>
</tbody></table>
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td colspan="7"><font size="1"><em>following are the results of different patch sizes, a ViT-Base model and a mask ratio of 0.5 for the pre-trained MAECG:</em></font></td>
</tr>
<th style="text-align:center"></th>
<th style="text-align:center">2</th>
<th style="text-align:center">4</th>
<th style="text-align:center">8</th>
<th style="text-align:center">16</th>
<th style="text-align:center">32</th>
<th style="text-align:center">96</th>
<!-- TABLE BODY -->
<tr><td align="left">MITDB-DS2</td>
<td align="center">93.8</td>
<td align="center">94.5</td>
<td align="center">94.7</td>
<td align="center">95.1</td>
<td align="center"><b>95.6</b></td>
<td align="center">94.1</td>
</tr>
<tr><td align="left">ECGIDDB</td>
<td align="center">88.7</td>
<td align="center">92.6</td>
<td align="center">95.5</td>
<td align="center">98.4</td>
<td align="center"><b>98.8</b></td>
<td align="center">95.7</td>
</tr>
</tbody></table>
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td colspan="10"><font size="1"><em>following are the results of different mask ratios, a ViT-Base model and a patch size of 32 for the pre-trained MAECG:</em></font></td>
</tr>
<th style="text-align:center"></th>
<th style="text-align:center">0.1</th>
<th style="text-align:center">0.2</th>
<th style="text-align:center">0.3</th>
<th style="text-align:center">0.4</th>
<th style="text-align:center">0.5</th>
<th style="text-align:center">0.6</th>
<th style="text-align:center">0.7</th>
<th style="text-align:center">0.8</th>
<th style="text-align:center">0.9</th>
<!-- TABLE BODY -->
<tr><td align="left">MITDB-DS2</td>
<td align="center">92.6</td>
<td align="center">93.4</td>
<td align="center">95.3</td>
<td align="center">95.4</td>
<td align="center"><b>95.6</b></td>
<td align="center">95.3</td>
<td align="center">95.3</td>
<td align="center">95.2</td>
<td align="center">90.3</td>
</tr>
<tr><td align="left">ECGIDDB</td>
<td align="center">69.4</td>
<td align="center">73.4</td>
<td align="center">97.9</td>
<td align="center">98.5</td>
<td align="center"><b>98.8</b></td>
<td align="center">98.7</td>
<td align="center">98.0</td>
<td align="center">96.5</td>
<td align="center">38.5</td>
</tr>
</tbody></table>

### Visualization demo

Run our interactive visualization demo with [Colab notebook](https://colab.research.google.com/github/PriceWang/MAECG/blob/main/demo.ipynb):

<p align="center">
  <img src="https://github.com/PriceWang/MAECG/assets/30796250/f42fbe96-5fed-47d7-a75e-8dd8fa73b571" width="1080">
</p>

### License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for details.
