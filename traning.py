import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import ImageNet
import torchvision.models as models
from PIL import Image

import argparse

from dataset import TextureDataset

import matplotlib.pyplot as plt
from tqdm import tqdm

# 전처리
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='train MNIST data for study')
    parser.add_argument('--BatchSize','-bs',dest='bs',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--learningRate','-lr',dest='lr',default=1e-3)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = models.mobilenet_v3_small()
    model.classifier.add_module('resize_output',nn.Linear(1000,2))
    model.to(DEVICE)
    
    textur_train = TextureDataset('./datasets/TextureImage','train')
    train_dataloader = DataLoader(textur_train,batch_size=args.bs,shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        for idx, (label, input) in enumerate(train_dataloader):
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(input)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            loss = criterion(probabilities,label)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f'Epoch:{epoch}[{idx}/{len(train_dataloader)}],Loss:{loss.item():.2f}')

        if (epoch + 1) % 2 == 0:
            model.eval()

            correct = 0
            test_loss = 0
            with torch.no_grad():
                for idx, (label, input) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
                    input = input.to(DEVICE)
                    label = label.to(DEVICE)
                    
                    output = model(input)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    loss = criterion(probabilities,label)
                    prediction = probabilities.max(1,keepdim = True)[1]

                    test_loss += loss.item()
                    correct += prediction.eq(label.view_as(prediction)).sum().item()

            test_loss /= len(train_dataloader)
            test_accuracy = 100. * correct / len(train_dataloader.dataset)
            
            print(f'[EPOCH: {epoch}], Test Loss:{test_loss:.4f},Test Accuracy: {test_accuracy:.2f}')
    
        
if __name__ == '__main__':
    main()