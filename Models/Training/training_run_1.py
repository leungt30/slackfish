import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
import wandb
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class SlackFishCNN(nn.Module):
    def __init__(self, input_feat):
        super(SlackFishCNN, self).__init__()
        
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
        # print("CNN output shape:", x.shape)

        x = x.view(x.size(0), -1)
        # print("Flattened CNN output:", x.shape)
    
        # print("Extra features shape:", features.shape)
        x = torch.cat([x,features], dim=1)
        # print("Concatenated shape:", x.shape)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x





LEARNING_RATE = 0.001
EPOCHS = 1000
SAVE_EVERY = 25


# Test each GPU
for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        x = torch.tensor([1.0]).cuda()
        print(f"✓ GPU {i} ({torch.cuda.get_device_name(i)}) is available!")
        del x
        torch.cuda.empty_cache()
        #break  # Found a working GPU
    except Exception as e:
        print(f"✗ GPU {i} failed: {e}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_one_epoch(model, criterion, optimizer, loader):
    model.train()

    total_loss = 0.0
    for bitboards_batch, x_batch, y_batch in loader:
        bitboards_batch = bitboards_batch.to(device)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(bitboards_batch, x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def validate(model, criterion, optimizer, val_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = len(val_loader.dataset)
    
    with torch.no_grad(): 
        for bitboards_batch, x_batch, y_batch in val_loader:
            bitboards_batch = bitboards_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(bitboards_batch, x_batch)
            loss = criterion(output, y_batch) 
            total_loss += loss.item() * x_batch.size(0)
            

    avg_loss = total_loss / total_samples
    
    model.train()  
    return avg_loss


from sklearn.model_selection import train_test_split

with np.load("preprocessed_chess_data_nov_8.npz") as data:
    
    engineered_features = ['piece_count', 'piece_mobility', 'player_turn', 'en_passant_available', 'in_check', 'castling_rights', 'pst_score' ]
    arrays = [
        data["piece_count"][:,None],
        data["piece_mobility"],
        data["player_turn"][:, None],
        data["en_passant_available"][:, None],
        data["in_check"][:, None],
        data["castling_rights"],
        data["pst_score"][:, None]
    ]
    Xs = np.concatenate(arrays, axis=1)
    print(Xs.shape)
    y = data['eval']
    
    indices = np.arange(len(Xs))
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        Xs, y, indices,
        test_size=0.2,      # 20% test, 80% train
        random_state=42,    # ensures reproducibility
        shuffle=True        # shuffles before splitting (default = True)
    )

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    Scaler_X = StandardScaler()
    Scaler_Y = StandardScaler()
    X_train_scaled = Scaler_X.fit_transform(X_train)
    y_train_scaled = Scaler_Y.fit_transform(y_train)

    X_test_scaled = Scaler_X.transform(X_test)
    y_test_scaled = Scaler_Y.transform(y_test)

    bitboard_train = data['bitboards'][idx_train]
    bitboard_test = data['bitboards'][idx_test]

    # tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)
    bitboard_train_t = torch.tensor(bitboard_train, dtype=torch.float32).reshape(-1, 12, 8, 8)
    bitboard_test_t = torch.tensor(bitboard_test, dtype=torch.float32).reshape(-1, 12, 8, 8)

    train_dataset = TensorDataset(bitboard_train_t, X_train_t, y_train_t)
    test_dataset = TensorDataset(bitboard_test_t, X_test_t, y_test_t)

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity = "timothyrleung-mcmaster-university",
    # Set the wandb project where this run will be logged.
    project="timothyrleung-mcmaster-university-slackfish",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "CNN",
        "dataset": "Chess data",
        "epochs": EPOCHS,
    },
)
num_features = X_train_t.shape[1]
model = SlackFishCNN(input_feat = num_features)
model = model.to(device=device)
# model = torch.nn.DataParallel(model)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    eps=1e-8,
    weight_decay=1e-4,
)

criterion = nn.MSELoss()

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,     
    shuffle=True,      
    num_workers=2      # can be >0 for speed if supported
)

test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
train_losses = []
val_losses = []


file_path = "CNN/SlackFishCNN_499"
checkpoint = torch.load(file_path, map_location="cuda")
past_epoch_stop = checkpoint["epoch"]
train_losses = checkpoint["train_loss"]
val_losses = checkpoint["val_loss"]
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


for epoch in range(past_epoch_stop+1, EPOCHS+past_epoch_stop+1):
    train_loss = train_one_epoch(model, criterion, optimizer, train_loader)
    val_loss = validate(model, criterion, optimizer, test_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % SAVE_EVERY == 0:
        file_path = f"CNN/SlackFishCNN_{epoch}"
        
        torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_losses,
        "val_loss": val_losses,
        }, file_path)
    
    wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    


