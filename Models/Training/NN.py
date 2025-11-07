import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re

INPUT_SIZE = 12 * 8 * 8 + 300
class SlackFishNN(nn.Module):
    def __init__(self):
        super(SlackFishNN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512,1)
        )
        

    def forward(self, x):
        x = self.nn(x)
        return x
        
model_NN = SlackFishNN()
num_params_NN = sum(p.numel() for p in model_NN.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params_NN:,}")