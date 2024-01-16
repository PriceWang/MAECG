<!--
 * @Author: Guoxin Wang
 * @Date: 2024-01-11 16:50:18
 * @LastEditors: Guoxin Wang
 * @LastEditTime: 2024-01-16 16:03:23
 * @FilePath: /mae/readme.md
 * @Description: 
 * 
 * Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
-->
## MAECG: A PyTorch Implementation
<p style="text-align:center">
  <img src="https://private-user-images.githubusercontent.com/30796250/297114254-c9670aed-40c2-43cc-9209-1924f5b6e7de.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDU0MjExNjAsIm5iZiI6MTcwNTQyMDg2MCwicGF0aCI6Ii8zMDc5NjI1MC8yOTcxMTQyNTQtYzk2NzBhZWQtNDBjMi00M2NjLTkyMDktMTkyNGY1YjZlN2RlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMTYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTE2VDE2MDEwMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAwMTk0NTM1YTFlMjU5YzRjNGU2NjNkNmRmNjU2MWExNzFjYzI5ZDg5OGJjMjIzMmY2N2ExMTczMTNlMGY2MzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.UloGr1lO4rf2ZUJQBdpYyj2Qj4oYKO-TENHXTuV5cXQ" width="480">
</p>

This is a PyTorch/GPU implementation of the paper [A Task-Generic High-Performance Unsupervised Pre-training Framework for ECG](aaa):

```
aaaa
```

This repo is inspired by the [MAE repo](https://github.com/facebookresearch/mae).

### Catalog
- [x] Visualization demo
- [x] Pre-training code
- [x] Fine-tuning code
- [x] Pre-trained checkpoints

### Requirement
Install the required package:

```
pip install -r requirements.txt
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

- Here we creat samples of length: `width` * 2.
- `expansion` is a simple replication to extend dataset.

To generate labelled ECG datasets, run the following command:

```
python dataprocess.py \
    --task ${task} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --width 240 \
    --channel_names ${channel_names} \
    --numclasses 5 \
    --expansion 1 \
```

- Choose `task` from "af_beat", "au_beat" and "dn_beat".
- Set `--prefix ${prefix}` when original data path is nested.
- Set `--channel_names_wn ${channel_names_wn}` for denoise/decoder task.
- Set `--numclasses 2` or `--numclasses 4` for different classification.
- Set `--mitdb` to generate datasets from MITDB with special splits.

The following table provides the generated datasets used in the paper:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th style="text-align:center"></th>
<th style="text-align:center">CINC2021</th>
<th style="text-align:center">MITDB</th>
<th style="text-align:center">INCARTDB</th>
<th style="text-align:center">ECGIDDB</th>
<!-- TABLE BODY -->
<tr><td style="text-align:left">Unlabelled</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1iRiKVuEFlIrSdhk-rLFTPpNy5DkGsJ1j/view?usp=drive_link">Download</a></td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
</tr>
<tr><td style="text-align:left">Arrhythmia Classification (Train)</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/16D2kmHp_ajZW67OTMr89Dmh7dGvNtkEn/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/13dWVe-S1dPzuVSsrnFvxvle1Akle2gv0/view?usp=drive_link">Download</a></td>
<td style="text-align:center">X</td>
</tr>
<tr><td style="text-align:left">Arrhythmia Classification (Valid)</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1ri9ca7CuDix2xOQnD-5Z7Wz_AcWmNOKF/view?usp=drive_link">Download</a></td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
</tr>
<tr><td style="text-align:left">Arrhythmia Classification (Test)</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1HQLwkKOegrTgLHd94gf1scCo2_fdvuUo/view?usp=drive_link">Download</a></td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
</tr>
<tr><td style="text-align:left">Human Identification (Train)</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1q2Y-htBWQtJOHBll3bzLbHbPYqvgwiE0/view?usp=drive_link">Download</a></td>
</tr>
<tr><td style="text-align:left">Human Identification (Valid)</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1fqe6VyVoowQmATJcaSYHhrSz3bFhecFq/view?usp=drive_link">Download</a></td>
</tr>
<tr><td style="text-align:left">Human Identification (Test)</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1wn35Ou5kIaJAlG_peu4lsn4aV2nbBg8j/view?usp=drive_link">Download</a></td>
</tr>
<tr><td style="text-align:left">Denoising (Train)</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/15zbQ3pVJfugWgVf51bRGnc6wph2paEed/view?usp=drive_link">Download</a></td>
</tr>
<tr><td style="text-align:left">Denoising (Test)</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center">X</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1_EdqbUvmH-bYRxnh-QiQXbdVld3Gspzc/view?usp=drive_link">Download</a></td>
</tr>
</tbody></table>

### Pre-training
To pre-train ViT-Base (recommended default) with multi-node distributed training, run the following on 1 nodes with 2 GPUs each:

```
OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc-per-node=2 main_pretrain.py \
    --model mae_vit_base_patch32 \
    --batch_size 256 \
    --epochs 50 \
    --accum_iter 4 \
    --mask_ratio 0.5 \
    --lr 3e-4 \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --log_dir ${log_dir} \
```

- Here the effective batch size is 256 (`batch_size` per gpu) * 1 (nodes) * 2 (gpus per node) * 4 (`accum_iter`) = 2048.
- To train ViT-Atto, ViT-Tiny, ViT-Small, ViT-Large or ViT-Huge with different patch size, set `--model mae_vit_${model_size}_patch${patch_size}`.
- Set `mask_ratio` for mask ratio.
- See [MAE pre-training](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md) for detailed parameter setting.
- Training time is ~3h in 2 A100 GPUs (500 epochs).

The following table provides the pre-trained checkpoints used in the paper:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th style="text-align:center"></th>
<th style="text-align:center">ViT-Atto</th>
<th style="text-align:center">ViT-Tiny</th>
<th style="text-align:center">ViT-Small</th>
<th style="text-align:center">ViT-Base</th>
<th style="text-align:center">ViT-Large</th>
<th style="text-align:center">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td style="text-align:left">Pre-trained Checkpoint</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1oSd0HqT9KOxtob0Vh260xj_cvu5GELll/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/18kSHlZXpxKMhoq82Ryl8_y56mfvWvnlB/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1EptUU4Yjm2UCxusBt5OaFmRNCi9mkfjd/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/15KQrAaLg-o3xQGvYteZ6b4aVoVzusgKZ/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1Nrok2tbRzRwgjJN0ZgoEKhYptpn2kj3e/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1GJ8cfcNuWFAiCreBC3XpdWdSrVR4QYGM/view?usp=drive_link">Download</a></td>
</tr>
</tbody></table>

### Fine-tuning with pre-trained checkpoints
#### Fine-tuning
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
    --log_dir ${log_dir}
```

- Here the effective batch size is 1024 (`batch_size` per gpu) * 1 (node) * 1 (gpus per node) = 1024.
- Set `--train_path ${train_path_1} ${train_path_2} ...` to fine-tune with multiple datasets
- See [MAE fin-tuning](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) for detailed parameter setting.
- Training time is ~53m in 1 RTX3090 GPU.

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
    --log_dir ${log_dir}
```

- Here the effective batch size is 32 (`batch_size` per gpu) * 1 (node) * 1 (gpus per node) = 32.
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
    --log_dir ${log_dir}
```

- Training time is ~23m in 1 RTX3090 GPU.

#### Evaluation

As a sanity check, run evaluation using our fine-tuned models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th style="text-align:center"></th>
<th style="text-align:center">Arrhythmia Classification</th>
<th style="text-align:center">Human Identification</th>
<!-- TABLE BODY -->
<tr><td style="text-align:left">Fine-tuned Checkpoint</td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/1ecgRMODPc8SCfi45qDWTLZVcuy3jWmMm/view?usp=drive_link">Download</a></td>
<td style="text-align:center"><a href="https://drive.google.com/file/d/16Ncku76qUAWvoHZBaLnotnFWpE2XvX5I/view?usp=drive_link">Download</a></td>
</tr>
<tr><td style="text-align:left">Reference Accuracy</td>
<td style="text-align:center">99.622</td>
<td style="text-align:center">100.000</td>
</tr>
</tbody></table>

Evaluate arrhythmia classification on INCARTDB in a single GPU:

```
python main_finetune.py \
    --model vit_base_p32 \
    --resume ${finetune_ckpt} \
    --test_path ${INCARTDB_path} \
    --eval
```

This should give:

```
* Acc@1 99.622 Acc@3 99.988 loss 0.097
```

Evaluate human identification on ECGIDDB_train:

```
python main_finetune.py \
    --model vit_base_p32 \
    --resume ${finetune_ckpt} \
    --test_path ${ECGIDDB_train_path} \
    --eval
```

This should give:

```
* Acc@1 100.000 Acc@3 100.000 loss 0.188
```














### Results

By fine-tuning these pre-trained models, we rank #1 in these tasks (detailed in the paper):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td colspan="5"><font size="1"><em>following are the results of different models, a patch size of 32 and a mask ratio of 0.5 for the pre-trained MAECG:</em></font></td>
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
<td style="text-align:center">93.3</td>
<td style="text-align:center">93.4</td>
<td style="text-align:center">94.8</td>
<td style="text-align:center"><b>95.6</b></td>
<td style="text-align:center">95.4</td>
<td style="text-align:center">95.4</td>
</tr>
<tr><td align="left">ECGIDDB</td>
<td style="text-align:center">94.1</td>
<td style="text-align:center">97.1</td>
<td style="text-align:center">98.7</td>
<td style="text-align:center"><b>98.8</b></td>
<td style="text-align:center">98.5</td>
<td style="text-align:center">98.3</td>
</tr>
</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td colspan="5"><font size="1"><em>following are the results of different patch sizes, a ViT-Base model and a mask ratio of 0.5 for the pre-trained MAECG:</em></font></td>
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
<td style="text-align:center">93.8</td>
<td style="text-align:center">94.5</td>
<td style="text-align:center">94.7</td>
<td style="text-align:center">95.1</td>
<td style="text-align:center"><b>95.6</b></td>
<td style="text-align:center">94.1</td>
</tr>
<tr><td align="left">ECGIDDB</td>
<td style="text-align:center">88.7</td>
<td style="text-align:center">92.6</td>
<td style="text-align:center">95.5</td>
<td style="text-align:center">98.4</td>
<td style="text-align:center"><b>98.8</b></td>
<td style="text-align:center">95.7</td>
</tr>
</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td colspan="5"><font size="1"><em>following are the results of different mask ratios, a ViT-Base model and a patch size of 32 for the pre-trained MAECG:</em></font></td>
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
<td style="text-align:center">92.6</td>
<td style="text-align:center">93.4</td>
<td style="text-align:center">95.3</td>
<td style="text-align:center">95.4</td>
<td style="text-align:center"><b>95.6</b></td>
<td style="text-align:center">95.3</td>
<td style="text-align:center">95.3</td>
<td style="text-align:center">95.2</td>
<td style="text-align:center">90.3</td>
</tr>
<tr><td align="left">ECGIDDB</td>
<td style="text-align:center">69.4</td>
<td style="text-align:center">73.4</td>
<td style="text-align:center">97.9</td>
<td style="text-align:center">98.5</td>
<td style="text-align:center"><b>98.8</b></td>
<td style="text-align:center">98.7</td>
<td style="text-align:center">98.0</td>
<td style="text-align:center">96.5</td>
<td style="text-align:center">38.5</td>
</tr>
</tbody></table>