import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import numpy as np

import glob

import torch
import torchvision.transforms as transforms

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from skimage import io, transform 

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



from pytorch_wavelets import DWTForward, DWTInverse

from project.cartoonX import CartoonX
from project.pixelRDE import PixelRDE

from notebooks.xaimethods.guidedbackprop import GB
from notebooks.xaimethods.lrp import LRP
from notebooks.xaimethods.vargrad import Vargrad
from notebooks.xaimethods.ig import IG
from notebooks.xaimethods.cartoonRDE import CartoonRDE
from notebooks.xaimethods.pixelRDE import PixelRDE
from notebooks.xaimethods.smoothgrad import Smoothgrad

import random

import seaborn as sns
class DecayExperiment:
    def __init__(self, device, model, J, mode, wave):
        self.model = model
        self.device = device
        self.forward_dwt = DWTForward(J=J, mode=mode, wave=wave).to(device)
        self.inverse_dwt = DWTInverse(mode=mode, wave=wave).to(device)
        self.softmax = torch.nn.Softmax(dim=1)

    def plot_decay(self, img_list, expl_dict):
        expl_dict["random pixel"] = [torch.rand_like(s) for s in expl_dict["pixelrde"]]
        expl_dict["random wavelet"] = [[torch.rand_like(s_ll),[torch.rand_like(s) for s in s_lh]]
                                       for s_ll, s_lh in expl_dict["cartoonx"]]
        pct = [5*i/100 for i in range(0,21)]
        for expl_name in expl_dict.keys():
            print(expl_name)
            expl_list = expl_dict[expl_name]

            distortion_list = self.get_distortion_list(img_list, expl_list, expl_name, pct)
            
            distortion_tensor = torch.Tensor(distortion_list)
            assert distortion_tensor.shape == (len(img_list), len(pct))
            distortion = distortion_tensor.mean(dim=0)
            assert distortion.shape == (len(pct),)

            plt.plot(pct, distortion.flatten().numpy(), label=expl_name)
 

        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("Percentage of non-randomized important coefficients")
        plt.ylabel("Distortion")
        plt.show()

    def get_distortion_list(self, img_list, expl_list, expl_name, pct):

        distortion_list = []
        
        for img, expl in zip(img_list, expl_list):
            d = []
            for p in pct:
                # Compute distortion
                if expl_name.lower() in ["cartoonx", "random wavelet"]: 
                    d_pct = self.get_wavelet_distortion(img, expl, p).detach()
                else: 
                    d_pct = self.get_pixel_distortion(img,expl,p).detach()

                d.append(d_pct)

            distortion_list.append(d)
        
            
        return distortion_list

    def get_pixel_distortion(self, img, expl, p):
        expl = expl.detach().squeeze(0).to(self.device).float()
        img = img.detach().squeeze(0).to(self.device).float()
        
        assert len(img.shape)==3
        assert len(expl.shape)==2

        shape = expl.shape
        
        mask = torch.zeros_like(expl, requires_grad=False)
        
        # Get indices of top p percent entries in explanation mask
        indices = torch.topk(expl.flatten(), int(np.prod(np.asarray(shape))*p))[1]


        
        # Get mean and variance
        mean = torch.mean(img)
        std = torch.std(img)

        # Replace values marked with indices
        mask = mask.flatten()
        mask[indices] = 1
        mask = mask.view(shape)
        noise = mean + std*torch.randn_like(mask, requires_grad=False)
        new_img = img * mask + (1-mask)*noise
        new_img.clamp_(0,1)
        
        
        """
        print(new_img.shape)
        plt.imshow(np.asarray(transforms.ToPILImage()(new_img)))
        plt.axis("off")
        plt.title(f"{p}%")
        plt.show()
        """

        # Get l2 distortion in model output
        distortion = torch.sqrt(((self.softmax(self.model(img.unsqueeze(0))) - self.softmax(self.model(new_img.unsqueeze(0))))**2).sum())
        
        return distortion
    

    
    def get_wavelet_distortion(self, img, expl, p):
        
            if img.shape[0]!=1: img.unsqueeze_(0)
            ll, lh = self.forward_dwt(img)
            ll = ll.detach()
            for s in lh: s.detach()
        
            
            expl_flat = self.get_flat_wavelets(expl[0], expl[1])
            mask_flat = torch.zeros_like(expl_flat)
            
            indices = torch.topk(expl_flat, int(len(mask_flat)*p))[1]
            mask_flat[indices] = 1
            mask = self.unflat_wavelets(mask_flat, expl[0], expl[1])
            
            # Perturb ll band  
            mean = torch.mean(ll)
            std = torch.std(ll)
            noise = mean + std * torch.randn_like(mask[0], requires_grad=False)
            new_ll = ll * mask[0] + (1-mask[0]) * noise
            
            # Perturb lh band
            new_lh = []
            means = [torch.mean(c) for c in lh]
            stds = [torch.std(c) for c in lh]
            
            for i,c in enumerate(lh):
                noise = means[i] + stds[i] * torch.randn_like(mask[1][i], requires_grad=False)
                new_lh.append(c * mask[1][i] + (1-mask[1][i]) * noise)
            

            new_img = self.inverse_dwt((new_ll, new_lh)).clamp(0,1)
            
            """
            print(new_img.shape)
            plt.imshow(np.asarray(transforms.ToPILImage()(new_img.squeeze(0))))
            plt.axis("off")
            plt.title(f"{p}%")
            plt.show()
            """

            # Get l2 distortion in model output
            distortion = torch.sqrt(((self.softmax(self.model(img)) - self.softmax(self.model(new_img)))**2).sum())
            #assert distortion.shape==(1,), print(distortion.shape)

            return distortion
            
    def get_flat_wavelets(self, ll,lh):
        ll_flat = ll.flatten()
        lh_flat = [y.flatten() for y in lh]
        coef_flat = torch.cat([ll_flat]+lh_flat)
        return coef_flat

    def unflat_wavelets(self, coef_flat, ll,lh):
        dim = len(ll.flatten())
        ll_new = coef_flat[:dim].view(ll.shape)
        lh_new = []

        for s in lh:
            next_dim = dim +len(s.flatten())
            lh_new.append(coef_flat[dim:next_dim].view(s.shape))
            dim = next_dim

        return (ll_new, lh_new)
            







