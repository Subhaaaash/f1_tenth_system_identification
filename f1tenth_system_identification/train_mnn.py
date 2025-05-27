import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from mnn_model import MNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Users/subhash/Documents/projects/f1tenth_system_identification/data/spielberg.csv")
x_vals = df["x"].values
y_vals = df["y"].values
yaw_vals = df["yaw"].values
steer_vals = df["steering_angle"].values
vel_vals = df["velocity"].values

# Create inputs and targets for series-parallel
inputs = []
targets = []

for k in range(1, len(df)):
    input_k = [steer_vals[k], vel_vals[k], x_vals[k-1], y_vals[k-1], yaw_vals[k-1]]
    target_k = [x_vals[k], y_vals[k], yaw_vals[k]]
    inputs.append(input_k)
    targets.append(target_k)

inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = MNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(5000):
    model.train()
    model.mem_layer.memory = None  # reset memory
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "mnn_model.pth")
