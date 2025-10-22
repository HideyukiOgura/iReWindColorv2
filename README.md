# iReWindColor (ICASSP 2025) & iReWindColor v2 (IEEE ACCESS) Official Implementation

This is the official PyTorch implementation of the paper: 
- [iReWindColor: Vision Transformer with Residual Embedding and Window Encoder for Point-Interactive Image Colorization](https://ieeexplore.ieee.org/document/10890353)
- [iReWindColor v2: Cross-Window Enhanced Transformer for Point Interactive Image Colorization](https://ieeexplore.ieee.org/abstract/document/11153943/).

<p align="center">
  <img width="90%" src="docs/demo.gif">
</p>

## Demo

Try colorizing images yourself with the [demo software](https://github.com/HideyukiOgura/iReWindColorv2/edit/main/demo/).

## Pretrained

Checkpoints for iReWindColor models are available in the links below.
- [iReWindColor](https://drive.google.com/file/d/12qVzjag87ynDLcaoKew8YzEQmx5zZIEz/view?usp=sharing)
- [iReWindColorv2](https://drive.google.com/file/d/1I-zHixTHQci7GaVB8_NxkFTUILy5Xf6s/view?usp=sharing)

## Testing

### Installation

Our code is implemented in Python 3.8, torch>=1.8.2
```
git clone https://github.com/HideyukiOgura/iReWindColorv2.git
pip install -r requirements.txt
```

### Testing

You can generate colorization results when iColoriT is provided with randomly selected groundtruth hints from color images. 
Please fill in the path to the model checkpoints and validation directories in the scripts/infer.sh file.

```
bash scripts/infer.sh
```

Then, you can evaluate the results by running

```
bash scripts/eval.sh
```

The codes used for randomly sampling hint locations can be seen in hint_generator.py

## Training

First prepare an official ImageNet dataset with the following structure. 

```
train
 └ id1
   └ image1.JPEG
   └ image2.JPEG
   └ ...
 └ id2
   └ image1.JPEG
   └ image2.JPEG
   └ ...     

```

Please fill in the train/evaluation directories in the scripts/train.sh file and execute

```
bash scripts/train.sh
```
