# Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs - Work In Progress

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
If you wish to train the model:
<br/>
(1) Follow the instructions on the official [Visual Genome Website](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) and download the images
<br/>
(2) Download the LAION 400M dataset from [LAION](https://laion.ai/)

# Evaluation Datasets
## VL-Checklist
Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md
<br/>
Make sure to change IMG_ROOT accordingly in all yaml files under [this directory](https://github.com/AlonMendelson/SGVL/blob/main/BLIP/VL_CheckList/corpus/v1)

## Winoground
Fill in your HF authentication token in [this file](https://github.com/AlonMendelson/SGVL/blob/main/BLIP/Winoground/evaluate_winoground.py)




# Installation
create a conda environment with all packages from yaml file and activate:

```
conda env create -f environment.yml
conda activate SGVL
```

clone the repository

```
git clone https://github.com/AlonMendelson/SGVL
cd SGVL
```
For training - download the data and annotations directory

```
gdown --fuzzy https://drive.google.com/drive/folders/1exie6ivcRb_RR1Lulcm2-Kdsky4JQgV6?usp=drive_link --folder
```
download the BLIP model checkpoint

```
mkdir BLIP/pretrained_checkpoints
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth -P BLIP/pretrained_checkpoints