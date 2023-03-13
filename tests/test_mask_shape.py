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

from cartoonx import CartoonXFactory
from cartoonx import WaveletBasedCartoonX

devid = 0

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = f'cuda:{devid}'
else:
    raise RuntimeError('No gpu available')
# Get image classifier
model = models.mobilenet_v3_small(pretrained=True).eval().to(device)
# Get image as torch tensor
img = Image.open(Path('imgs') / 'kobe.jpg').convert('RGB')
tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(256, 256))
    ]
)
# Dummy input image and target label
x = torch.randn(1, 3, 256, 256, device=device)
y = torch.tensor([0], device=device)

# Init wavelet-based explainer
cartoon = CartoonXFactory.create(system='wavelets', device=device)
hparams = {'N': 5} 
explanation = cartoon.explain(model, x, y, **hparams)
assert explanation['cartoonx'].shape == x.shape, explanation['cartoonx'].shape
print('Completed test...')
