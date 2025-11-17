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
from cartoonx import WaveletBasedCartoonX, ShearletBasedCartoonX


def test(devid: int) -> None:
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
    x = tf(img).unsqueeze(0).to(devid)
    y = torch.tensor([430]).to(devid)  # 430=Basketball class id

    # Init wavelet-based explainer
    cartoon = CartoonXFactory.create(system='wavelets', device=device)
    hparams = {'N': 5}
    runonce(cartoon, x, y, model, hparams)

    # Init shearlet-based explainer
    cartoon = CartoonXFactory.create(system='shearlets', device=device)
    hparams = {}
    runonce(cartoon, x, y, model, hparams)

    print('Completed test...')


def runonce(
    cartoon: Union[WaveletBasedCartoonX, ShearletBasedCartoonX],
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    hparams: dict
) -> None:
    """ Computes and visualizes explanation for classification of model
        for x
    """
    cartoon.explain(model, x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devid",
        type=int,
        default=0,
        help='Cuda device id'
    )
    args = parser.parse_args()
    test(args.devid)