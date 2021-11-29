import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from datasets.imagenet import ImagenetTrain

from project.cartoonX import CartoonX
from project.pixelRDE import PixelRDE


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import random

SYNSET_IMAGENET = tuple(open("/home/math/kolek/dev/CartoonX/experiment/synset_imagenet_list.txt").read().split('\n'))

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def get_dataset(name: str, model, random_seed: int, num_imgs: int, misclassified_only=True) -> list:
    # Set random seed
    random.seed(random_seed)

    # Get dataset
    if name.lower()=="imagenet":
        dataset = ImagenetTrain(transform=transforms.Resize(size=(256,256)))
    else:
        raise NotImplementedError(f"The dataset {name} was not implemented")
    
    rand_idx = random.sample(range(0,len(dataset)), len(dataset))
    new_dataset = []
    j = 0
    if misclassified_only:
        for i in rand_idx:
            x, y, _ = dataset[i]
            output = model(x.to(device).unsqueeze(0))
            label = nn.Softmax(dim=1)(output).max(1)[1].item()
            y_hat = SYNSET_IMAGENET[label]
            if y_hat!=y[1:]:
                new_dataset.append(dataset[i])
                j+=1
            if j==num_imgs: break
    else:
        for i in rand_idx:
            new_dataset.append(dataset[i])
            j+=1
            if j==num_imgs: break
            
    return new_dataset

def get_model(name: str) -> torch.nn.Module:
    # Get model 
    if name.lower()=="mobilenet":
        model = models.mobilenet_v3_small(pretrained=True)
    elif name.lower()=="vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise NotImplementedError(f"The model {name} was not implemented.")

    return model

def get_explanation(name: str):
    if name.lower()=="cartoonx":
        return CartoonX
    elif name.lower()=="pixelrde":
        return PixelRDE
    else: 
        raise NotImplementedError(f"The explanation {name} was not implemented.")
