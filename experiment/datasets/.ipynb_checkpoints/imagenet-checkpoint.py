import os
import glob

import nltk
from nltk.corpus import wordnet

from PIL import Image


import torch 

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


import nltk
from nltk.corpus import wordnet


LABEL_LIST = tuple(open("/home/math/kolek/dev/rde_wavelets/synset_words.txt").read().split('\n'))
LABEL_LIST = [x.replace('{',"").replace('\'',"").replace(',',"").replace('-'," ").replace('_'," ") for x in LABEL_LIST]

LABEL_LIST = [x.split(" ")[0] for x in LABEL_LIST]
LABEL_DICT = {}
for i, x in enumerate(LABEL_LIST): LABEL_DICT[x] = i
    
SYNSET_IMAGENET = tuple(open("/home/math/kolek/dev/rde_wavelets/synset_imagenet_list.txt").read().split('\n'))

class ImagenetTestDataset(Dataset):

    def __init__(self, transform=None, path="ILSVRC/Data/DET/test"):

        self.transform = transform
        self.path = path
    def __len__(self): return 5500
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        idx = f"0000{idx+1}"
        
        
        img_path = glob.glob(os.path.join(self.path,f'*{idx}.JPEG'))[0]
        image = Image.open(img_path)
        
        image = transforms.ToTensor()(image)

        if self.transform:
            image = self.transform(image)

        return image
    
    
    
class ImagenetTrain(Dataset):
        def __init__(self, transform=None,
                     root="/home/groups/ai/datasets/imagenet_dataset/ILSVRC2013_DET_train"):
            self.transform = transform
            self.root = root
            self.dirs = os.listdir(self.root)
            # Remove directories from list that are not in imagenet labels list
            """
            for s in ["n02062744","n02084071","n02274259",
                 "n02839592","n02970849","n03581125",
                 "n03841666","n04335209", "n01495701", "n00007846"]:
                self.dirs.remove(s)
            """    
            self.num_labels = len(self.dirs)
            filecount = 0 
            self.files = []
            for d in self.dirs:
                filecount += len(os.listdir(os.path.join(root,d)))
                self.files += [(file,d) for file in os.listdir(os.path.join(root,d))]
            self.length = filecount
            
        def __len__(self): return self.length
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_path, synset_id = self.files[idx]

            image = Image.open(os.path.join(self.root, synset_id, img_path))
            # Normaalize with [0.485, 0.456, 0.406] as  mean and [0.229, 0.224, 0.225] as std from imagenet
            transform_norm = transforms.Compose([transforms.ToTensor()])
            
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            image = transform_norm(image)

            if self.transform:
                image = self.transform(image)
                
            #label = LABEL_DICT[synset_id]
            identifier = img_path.replace('.',"")
            
            return image, synset_id, identifier
 
        
        def synset_to_label(self, synset_id):
            if synset_id[0] == "n": synset_id = synset_id[1:]
            for i in range(len(synset_id)):
                if synset_id[0] == "0":
                    synset_id = synset_id[1:]
                else:
                    return str(wordnet.synset_from_pos_and_offset('n',int(synset_id)))
