import numpy as np

import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import os.path as osp

import random

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

class TextureDataset(Dataset):
    def __init__(self,root_path, split='train',transform = preprocess,**args):
        self.IMAGE_SIZE = 224
        self.transform = transform
        self.split = split
        self.data_path = osp.join(root_path,split)
        
        self.data_list = []
        self.class_list = ()
        
        for class_name in os.listdir(self.data_path):
            self.class_list += (class_name,)
            
            for image_name in os.listdir(osp.join(self.data_path,class_name)):
                self.data_list.append((class_name,image_name))
            
    def __len__(self):
        return len(self.data_list) * 10
    
    def __getitem__(self, index):
        index %= len(self.data_list)
        
        curItem = self.data_list[index]
        
        image = cv2.imread(osp.join(self.data_path,curItem[0],curItem[1]))
        
        randomRange = (image.shape[1] - self.IMAGE_SIZE,image.shape[0] - self.IMAGE_SIZE)
        
        startPoint = (random.randrange(0, randomRange[0]),random.randrange(0,randomRange[1]))
        
        cropImage = image[startPoint[1]:startPoint[1] + self.IMAGE_SIZE,startPoint[0]:startPoint[0] + self.IMAGE_SIZE,:]

        return (self.class_list.index(curItem[0]), self.transform(cropImage))
    

if __name__ == '__main__':
    train = TextureDataset('./datasets/TextureImage','train')
    
    print(len(train))
    
    cv2.imshow('test',train[1][1])
    cv2.waitKey()
    
    cv2.destroyAllWindows()