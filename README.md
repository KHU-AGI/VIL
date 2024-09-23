# [ECCV 2024] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning

Official PyTorch implementation for ECCV 2024 paper:

**Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning**  
[Min-Yeong Park](https://github.com/pmy0792)\*, [Jaeho Lee](https://github.com/JH-LEE-KR)\*, and Gyeong-Moon Park† 

[![arXiv](https://img.shields.io/badge/arXiv-2409.10956-b31b1b.svg)](https://arxiv.org/abs/2409.10956) 


# Environment
- Python 3.8.x
- PyTorch 1.12.1
- Torchvision 0.13.1
- NVIDIA GeForce RTX 3090
- CUDA 11.3


# Getting Started
## Environment
```bash
git clone git@github.com/KHU-AGI/VIL.git
cd VIL
conda create -n VIL python==3.8
conda activate VIL
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Run ICON on VIL with iDigits dataset
```bash
python main.py --dataset iDigits --num_tasks 5 --seed 42 --versatile_inc --batch-size 24 --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --d_threshold
```
