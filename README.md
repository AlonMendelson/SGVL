# Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs

This is an official pytorch implementation of the paper [Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs](https://arxiv.org/abs/2305.06343). In this repository, we provide the PyTorch code we used to train and test our proposed SGVL model.

If you find SGVL useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@misc{herzig2023incorporating,
      title={Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs}, 
      author={Roei Herzig and Alon Mendelson and Leonid Karlinsky and Assaf Arbelle and Rogerio Feris and Trevor Darrell and Amir Globerson},
      year={2023},
      eprint={2305.06343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Data Preparation

Follow the instructions on the official [Visual Genome Website](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) and download the images

Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  



# Installation
create a conda environment with all packages from yaml file:

`conda env create -f environment.yml`

clone the repository

```
git clone https://github.com/AlonMendelson/SGVL
cd SGVL
```
create a data directory and download annotations

```
mkdir Data
cd Data
wget https://drive.google.com/file/d/1uvoS4XK6lu40M-TgX_Zs7UpI4kxm97_G/view?usp=drive_link
wget https://drive.google.com/file/d/1B_1qVpvdpHk-fwDlKorrsuW1dcWHtank/view?usp=drive_link
wget https://drive.google.com/file/d/1bTypvDZBhZYH3Ncb8E93LR2Kd3c2lV75/view?usp=drive_link
```


