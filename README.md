# Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs

This is an official pytorch implementation of the paper [Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs](https://arxiv.org/abs/2305.06343). In this repository, we provide the PyTorch code we used to train and test our proposed SGVL model.

If you find SGVL useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@inproceedings{
herzig2023incorporating,
title={Incorporating Structured Representations into Pretrained Vision {\textbackslash}\& Language Models Using Scene Graphs},
author={Roei Herzig and Alon Mendelson and Leonid Karlinsky and Assaf Arbelle and Rogerio Feris and Trevor Darrell and Amir Globerson},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=7DueCuvmgM}
}
```

# Data Preparation
If you wish to train the model:
<br/>
(1) Follow the instructions on the official [Visual Genome Website](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) and download the images
<br/>
(2) Download the LAION 400M dataset from [LAION](https://laion.ai/)

# Evaluation Datasets
Follow the instructions for the datasets you wish to evaluate the model on
## VL-Checklist
Prepare VL-Checklist datasets as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md
<br/>
Run the VL-Checklist setup code
```
python setup_vlc --VG PATH_TO_VG --Hake PATH_TO_HAKE --Swig PATH_TO_SWIG
```

## Winoground
Fill in your HF authentication token in [this file](https://github.com/AlonMendelson/SGVL/blob/main/BLIP/Winoground/evaluate_winoground.py)

## VSR
Follow the instructions in the [VSR repository](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) to download the images
<br/>
Fill in the path to the images folder [here](https://github.com/AlonMendelson/SGVL/blob/main/BLIP/vsr/vsr_dataset.py)




# Installation
Create a conda environment with all packages from yaml file and activate:

```
conda env create -f environment.yml
conda activate SGVL
```

Clone the repository

```
git clone https://github.com/AlonMendelson/SGVL
cd SGVL
```
Run the code setup file

```
python setup_code.py
```
# Inference
Download the BLIP-SGVL pretrained weights from this [link](https://drive.google.com/file/d/13jzpcLgGalO3hkiqVwziNAlCEZD90ENN/view?usp=drive_link)

Run the evaluation code
```
bash eval.sh
```

# Training
Download the data and annotations directory

```
gdown --fuzzy https://drive.google.com/drive/folders/1exie6ivcRb_RR1Lulcm2-Kdsky4JQgV6?usp=drive_link --folder
```
Download the BLIP model base checkpoint

```
mkdir BLIP/pretrained_checkpoints
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth -P BLIP/pretrained_checkpoints
```


Run the train code
```
bash train.sh
```