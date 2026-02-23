import os
from pathlib import Path
import argparse
from typing import Union

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import numpy as np


import cartoonx 

BASE_DIR = Path(__file__).resolve().parent

def run(devid: int, hparams: dict) -> None:
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = f'cuda:{devid}'
    else:
        raise RuntimeError('No gpu available')
    # Get image classifier
    model = models.mobilenet_v3_small(pretrained=True).eval().to(device)
    # Get image as torch tensor
    img = Image.open(BASE_DIR.parent / 'imgs' / 'kobe.jpg').convert('RGB')
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(256, 256))
        ]
    )
    x = tf(img).unsqueeze(0).to(devid)
    y = torch.tensor([430]).to(devid)  # 430=Basketball class id

    # Init wavelet-based explainer
    cartoon = cartoonx.CartoonXFactory.create(system='wavelets', device=device)

    result = cartoon.explain(model, x, y, **hparams)
    
    return x, result, cartoon

def plot_cartoonx_params_effect():
    """
    Plot CartoonX for increasing wavelet mask penalty and spatial energy penalty
    """
    count = 1
    plt.figure(figsize=(15, 15))
    ws = [10**i for i in range(-7, 0)] # wavelet mask penalty
    ss = [10**i for i in range(-7, 0)] # spatial energy penalty
    for s in ss:
        for w in ws:
            hparams = {'N': 300, 'wavelet_reg': w, 'spatial_reg': s}
            x, result, cartoon = run(devid=0, hparams=hparams)
            plt.subplot(len(ws), len(ss), count)
            plt.imshow(result['cartoonx'].cpu()[0].permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'w {w} - s{s}')
            count += 1
    plt.show()


# Run CartoonX 
hparams = {'N': 300, 'wavelet_reg': 1e-3, 'spatial_reg': 1e-4}
x, result, cartoon = run(devid=0, hparams=hparams)

# Plot CartoonX
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(x.cpu()[0].permute(1, 2, 0))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result['cartoonx'].cpu()[0].permute(1, 2, 0))
plt.axis('off')
plt.savefig('cartoonx.png')
plt.show()

# Uncomment code below to see effect of hyperparameters
# code plots CartoonX with increasing wavelet mask and spatial energy penalties
"""
count = 1
plt.figure(figsize=(15, 15))
ws = [10**i for i in range(-7, 0)] # wavelet mask penalty
ss = [10**i for i in range(-7, 0)] # spatial energy penalty
for s in ss:
    for w in ws:
        hparams = {'N': 300, 'wavelet_reg': w, 'spatial_reg': s}
        x, result, cartoon = run(devid=0, hparams=hparams)
        plt.subplot(len(ws), len(ss), count)
        plt.imshow(result['cartoonx'].cpu()[0].permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'w {w} - s{s}')
        count += 1
plt.show()
"""


