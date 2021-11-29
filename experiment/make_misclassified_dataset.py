import sys
import os
import argparse
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from project.cartoonX import CartoonX
from project.pixelRDE import PixelRDE
from experiment.utils import get_model, get_dataset, get_explanation

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np

# Set random seed
random_seed = 6

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get list of imagenet labels to convert prediction to string label
LABEL_LIST = tuple(open(os.path.join(sys.path[0], "imagenet_labels.txt")).read().split('\n'))
LABEL_LIST = [x.replace('{',"").replace('\'',"").replace(',',"").replace('-'," ").replace('_'," ") for x in LABEL_LIST]


def main(logdir, model, dataset, expl):
    """
        logdir is a path
        model is a torch model
        dataset is a list containing images
        expl is an explanation method

    """

    # Initialize lists to log stats to tensorboard
    mask_stats = []

    # Loop over dataset
    for i, (x,_,_) in enumerate(dataset):
        # ce is clean example and y is its predicted label
        x = x.to(device)
        x.unsqueeze_(0)
        x.requires_grad_(False)
        assert len(x.shape)==4

        print(f"Processing image {i}")

        """
        Get explanation mask for clean example
        """
        output = model(x)
        label = nn.Softmax(dim=1)(output).max(1)[1].item() 
        
        x_expl, x_mask, logs = expl(x, label)
        #mask_stats.append({"distortion": logs["distortion"], "l1-norm": logs["l1-norm"]}) 

        # Save mask in logdir
        if type(x_mask) is not torch.Tensor:
            ll_mask_dir = os.path.join(tensordir, "ll_masks")
            if not os.path.isdir(ll_mask_dir): os.makedirs(ll_mask_dir)
            torch.save(x_mask[0], os.path.join(ll_mask_dir,f"img{i}.pt"))

            lh_mask_dir = os.path.join(tensordir, "lh_masks")
            if not os.path.isdir(lh_mask_dir): os.makedirs(lh_mask_dir)
            for j, s in enumerate( x_mask[1]):
                torch.save(s, os.path.join(lh_mask_dir,f"scale={j}-img{i}.pt"))
        else: 
            maskdir = os.path.join(tensordir,"masks")
            if not os.path.isdir(maskdir): os.makedirs(maskdir)
            torch.save(x_mask, os.path.join(maskdir, f"mask-img{i}.pt"))
        
        # Save images in logdir
        imgdir = os.path.join(tensordir,"images")
        if not os.path.isdir(imgdir): os.makedirs(imgdir)
        torch.save(x, os.path.join(imgdir, f"img{i}.pt"))
        

    

if __name__ == "__main__":
    # Pasrse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of dataset", default="imagenet")
    parser.add_argument("--num_imgs", type=int, help="number of images", default=100)
    parser.add_argument("--model", type=str, help="name of model", default="mobilenet")
    parser.add_argument("--expl", type=str, help="name of explanation", default="pixel rde")
    parser.add_argument("--expl_params", type=str, help="name of file with explanation params")
    args = parser.parse_args()

    """
    Get model
    """
    model = get_model(name=args.model).eval().to(device)

    """
    Get dataset 
    """
    dataset = get_dataset(name=args.dataset, model=model, random_seed=random_seed, num_imgs=args.num_imgs, misclassified_only=True) 

    """
    Get explanation method and set its hyperparameters 
    """
    with open(os.path.join(sys.path[0], args.expl_params)) as f:
           hparams_expl = yaml.load(f, Loader=yaml.FullLoader)
    Expl = get_explanation(args.expl)
    expl = Expl(model=model, device=device, **hparams_expl)

    """
    Set up logdir
    """
    logdir = f"/home/groups/ai/kolek/misclassified/logdir/{args.expl}-{args.model}"
    if not os.path.isdir(logdir): os.makedirs(logdir)


    # Get path where tensors are saved
    tensordir = os.path.join(logdir, "tensors")
    if not os.path.isdir(tensordir): os.makedirs(tensordir)

    # Get path where hparams are saved
    hparamsdir = os.path.join(logdir, "hparams")
    if not os.path.isdir(hparamsdir): os.makedirs(hparamsdir)


    with open(os.path.join(hparamsdir,"hparams_expl.yaml"), 'w') as outfile:
        yaml.dump(hparams_expl, outfile, default_flow_style=False)
        
    # Create dataset of clean examples with respective explanation mask
    main(logdir, model, dataset, expl)
    print("Finished script!")
