import argparse
import os
import sys
import yaml
from shutil import copyfile
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from project.cartoonX import CartoonX 
from project.pixelRDE import PixelRDE

# Get current time for logging
now = datetime.now()
current_time = now.strftime("%d/%m/%Y %H:%M:%S")

# Get list of imagenet labels to convert prediction to string label
LABEL_LIST = tuple(open(os.path.join(sys.path[0], "imagenet_labels.txt")).read().split('\n'))
LABEL_LIST = [x.replace('{',"").replace('\'',"").replace(',',"").replace('-'," ").replace('_'," ") for x in LABEL_LIST]

def main(imgdir, logdir, tensorboard, resize_images):
    # Get device (use GPU if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get classifier to explain
    model = models.mobilenet_v3_small(pretrained=True).eval().to(device)

    # Get hyperparameters for wavelet RDE and pixel RDE
    with open(os.path.join(sys.path[0], "hparams.yaml")) as f:
        HPARAMS_CARTOONX = yaml.load(f, Loader=yaml.FullLoader)["CartoonX"]

    with open(os.path.join(sys.path[0], "hparams.yaml")) as f:
        HPARAMS_PIXEL_RDE = yaml.load(f, Loader=yaml.FullLoader)["PixelRDE"]

    # Initialize wavelet RDE and pixel RDE
    cartoonX = CartoonX(model=model, device=device, **HPARAMS_CARTOONX)
    pixelRDE = PixelRDE(model=model, device=device, **HPARAMS_PIXEL_RDE)

    # Get files of images
    files = os.listdir(imgdir)

    # Explain model decsision for each image in files
    for fname in files:
        print(f"Processing file: {fname}")
        # Get image and transform to tensor    
        x = Image.open(os.path.join(imgdir, fname))
        x = transforms.ToTensor()(x)
        if resize_images: x = transforms.Resize(size=(256,256))(x)
        x = x.to(device)

        # Get prediction for x
        output = model(x.unsqueeze(0).detach())
        max_idx = nn.Softmax(dim=1)(output).max(1)[1].item()
        label = LABEL_LIST[max_idx]

        # Get explanation for x
        exp_cartoonX = cartoonX(x.unsqueeze(0), target=max_idx)
        exp_pixelRDE = pixelRDE(x.unsqueeze(0), target=max_idx)

        # Plot explanations next to original image
        P = [(x, f"Pred:{label}"), (exp_cartoonX, "CartoonX"), (exp_pixelRDE, "Pixel RDE")]
        fig, axs = plt.subplots(1,3,figsize=(10,10))
        for idx, (img, title) in enumerate(P):
            args = {"cmap": "copper"} if idx>0 else {}
            axs[idx].imshow(np.asarray(transforms.ToPILImage()(img)), vmin=0, vmax=255, **args)
            axs[idx].set_title(title, size=8)
            axs[idx].axis("off")
        
        # Log images
        if tensorboard:
            # Log to tensorboard
            writer = SummaryWriter(os.path.join(logdir,f"image{fname}"))
            writer.add_figure(f"Explanations-{current_time}", fig)
            writer.flush()
            writer.close()
        else:
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
            plt.savefig(os.path.join(logdir, f"exp-{fname}"), bbox_inches='tight',transparent = True, pad_inches = 0)
        
    # Log the hparams file to check later what hparams were used 
    copyfile(os.path.join(sys.path[0],"hparams.yaml"), os.path.join(logdir, "hparams.yaml"))




        

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgdir", type=str, help="Directory of images to explain.", default=".")  
    parser.add_argument("--logdir", type=str, help="Directory where explanations are logged", default="exp_logs")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    parser.add_argument("--resize_images", dest="resize_images", action="store_true")
    args = parser.parse_args()

    main(imgdir=args.imgdir, logdir=args.logdir, resize_images=args.resize_images, tensorboard=args.tensorboard)
