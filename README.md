# Implicit Visual-Textual (IVT) - Pytorch
 
This repository is the implementation of the [paper](https://arxiv.org/pdf/2208.08608.pdf) "See Finer, See More: Implicit Modality Alignment for Text-based Person Retrieval."

 
## Installation

1. Creating conda environment

```bash 
conda create -n ivt python=3.7
conda activate ivt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

```
 
    
2. Install others

```bash
git clone https://github.com/TencentYoutuResearch/PersonRetrieval-IVT.git
cd PersonRetrieval-IVT
pip install -r requirements.txt
```

## Getting Started
### Pretrain
You can use our [pre-trained model](https://pan.baidu.com/s/1CMuU1Qnummscgw86smmN_A)[zxvu] directly,
otherwise,
you need to download several datasets: [Conceptual Captions](https://aclanthology.org/P18-1238.pdf), [SBU Captions](https://proceedings.neurips.cc/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf), [COCO](https://arxiv.org/pdf/1405.0312.pdf%090.949.pdf), and [Visual Genome](https://link.springer.com/article/10.1007/S11263-016-0981-7)

Change the data root in pretrain_cuhk.py, then run:
```bash
python train_pretrain.py 
```

### Training with text-based re-ID datasets 
```bash 
# We leverage four V-100 GPUs for training on CUHK-PEDES and ICFG-PEDES datasets.
# training with multi-gpus
sh start.sh

# or, you could also train them with a single gpu, but slow speed, maybe better performance.
python train_cuhkpedes_gpu.py
python train_icfg_gpu.py

# As RSTPReid is small, we leverage only one V-100 GPU for training.
python train_rstp.py 
``` 
 

### Trained Models
We provide our trained models at [Baidu Pan](https://pan.baidu.com/s/1lfZoVp9Uxu3Viw3j2nHoAw)[bpvu].
 
## Citations
If you find our work helpful, please cite using this BibTeX:
```bibtex 
@inproceedings{shu2023see,
  title={See finer, see more: Implicit modality alignment for text-based person retrieval},
  author={Shu, Xiujun and Wen, Wei and Wu, Haoqian and Chen, Keyu and Song, Yiran and Qiao, Ruizhi and Ren, Bo and Wang, Xiao},
  booktitle={ Proceedings of the European conference on computer vision Workshops (ECCVW)},
  pages={624--641},
  year={2023},
  organization={Springer}
}
```

## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at shuxj@mail.ioa.ac.cn.
 



