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
    def __init__(self, images_path):
        self.images_path = images_path
        self.mat = np.zeros((250000,5,5,3))
        k=0
        with open(os.path.join(self.images_path,'mat.npy'),'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            while f.tell() < fsz:
                self.mat[k,:,:,:] =np.load(f)
                k+=1
                print(k)
        self.mat = torch.transpose(torch.Tensor(self.mat), 1, 3)
        self.mat = torch.transpose(self.mat, 2, 3)
        print(self.mat.size())
        self.tuples=np.load(os.path.join(self.images_path,'SASR.npy'))


    def __len__(self):
        return len(self.tuples)-1

    def __getitem__(self, idx):
        SASR= self.tuples[idx]
        return [self.mat[idx], self.mat[idx+1], SASR]
    
    def get_SASR(self):
        return np.load(os.path.join(self.images_path,'SASR.npy'))
        
    