import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

class DemoDataset(Dataset):
    def __init__(self, states, actions):
        # Convert lists to tensors
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
class BehaviouralCloningModel(nn.Module):
    def __init__(self, input_dim=259, output_dim=2):
        super(BehaviouralCloningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh() # Continuous action output betwene [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)

def train_model():
    # Load all CSV files from the data directory
    csv_files = glob.glob("../data/*.csv")
    dfs = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2:].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = DemoDataset(X_train, y_train)
    val_dataset = DemoDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Switch to GPU if in Bragg or on Laptop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = BehaviouralCloningModel().to(device)
    
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 50
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)

            optimiser.zero_grad() # clear old gradients
            predictions = model(states) # agent predictions
            loss = criterion(predictions, actions)
            loss.backward()
            optimiser.step() # update

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # Validate on 20%
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                predictions = model(states)
                loss = criterion(predictions, actions)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/bc_model.pth")
            print("Saved")
            print(f"Best validation loss {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()