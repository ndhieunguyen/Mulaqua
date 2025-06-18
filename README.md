<h1 align="center">Mulaqua</h1>

This repository is the official implementation of [`Mulaqua`]()

## Abstract
> 

## Dependencies and Installation
```
conda env create -f environment.yaml
conda activate molecule
```
or
```
pip install -r requirements.txt
```
Download pretrained weights of [SwinOCSR](https://huggingface.co/datasets/CYF200127/MolNexTR/blob/main/molnextr_best.pth) and [MolNexTR](https://www.kaggle.com/datasets/gogogogo11/moedel), and put them into checkpoints/


## Preparing dataset by extracting molecular image features and converting the data into arrow format
```
python3 train.py --prepare_dataset
```
We defined fold 0 to fold `k-1` as the training process of k-fold cross-validation, while fold `k` fits all training data into the model. Please check `config.py`

## Training
```
python3 train.py
```

## Inferencing
```
python3 inference.py
```

## Citation
```
```