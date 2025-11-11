import torch
import torch.nn as nn

class SlackFishCNN_V1(nn.Module):
    def __init__(self, input_feat):
        super(SlackFishCNN_V1, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
                
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(128 * 2 * 2 + input_feat , 256) # 128 * 2 * 2 for kernels, + 12 for engineered features
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, board, features):
        x = self.cnn(board)

        x = x.view(x.size(0), -1)
    
        x = torch.cat([x,features], dim=1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SlackFishCNN_V2(nn.Module):
    def __init__(self, input_feat):
        super(SlackFishCNN_V2, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
                
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.nn = nn.Sequential(
            nn.Linear(128 * 2 * 2 + input_feat , 256), # 128 * 2 * 2 for kernels, + 12 for engineered features
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, board, features):
        x = self.cnn(board)

        x = x.view(x.size(0), -1)
    
        x = torch.cat([x,features], dim=1)
        
        x = self.nn(x)
        return x

class SlackFishCNN_V3(nn.Module):
    def __init__(self, input_feat):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
                
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
                
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.nn = nn.Sequential(
            nn.Linear(256 * 8 * 8 + input_feat , 256), # 256 * 2 * 2 for kernels, + 12 for engineered features
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, board, features):
        x = self.cnn(board)

        x = x.view(x.size(0), -1)
    
        x = torch.cat([x,features], dim=1)
        
        x = self.nn(x)
        return x