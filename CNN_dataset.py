import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms, utils
from torchvision.transforms.functional import resized_crop

#folder:
#images
        #images_b
                #train: 250000 images, SASR.npy
                #val: 50000 images, SASR.npy
        #images_e
                #test: 50000 images, SASR.npy
class SASR_Dataset(Dataset):
    #original image (100,100), 20 entries for one grid. After resizing 20 -> 10.
    def __init__(self, images_path, input_size=(50,50)):
        self.images_path= images_path
        self.tuples=np.load(os.path.join(self.images_path,'SASR.npy'))
        self.trans = transforms.Compose([
            transforms.Resize(input_size),         
            transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.tuples)-1

    def __getitem__(self, idx):
        path = os.path.join(self.images_path, str(idx)+'.png')
        image_s = Image.open(path).convert('RGB')
        path = os.path.join(self.images_path, str(idx+1)+'.png')
        image_sprime = Image.open(path).convert('RGB')
        SASR= self.tuples[idx]
        return [self.trans(image_s), self.trans(image_sprime), SASR]
    
    def get_SASR(self):
        return np.load(os.path.join(self.images_path,'SASR.npy'))
        
    