import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from mnn_model import MemoryNeuralNetwork

# File paths
CSV_PATH            = "/Users/subhash/Documents/projects/f1tenth_system_identification/data/spielberg.csv"
MODEL_SAVE_PATH     = "mnn_custom_model.pth"
INPUT_SCALER_PATH   = "input_scaler.save"
OUTPUT_SCALER_PATH  = "output_scaler.save"
TRAIN_RMSE_PLOT     = "training_rmse.png"
TRAJECTORY_PLOT     = "trajectory_xy.png"
YAW_PLOT            = "trajectory_yaw.png"

# Hyperparams
EPOCHS = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load & prepare data ---
df = pd.read_csv(CSV_PATH)
X, Y = [], []
for k in range(1, len(df)):
    X.append([
        df.loc[k, "steering_angle"],
        df.loc[k, "velocity"],
        df.loc[k-1, "x"],
        df.loc[k-1, "y"],
        df.loc[k-1, "yaw"]
    ])
    Y.append([
        df.loc[k, "x"],
        df.loc[k, "y"],
        df.loc[k, "yaw"]
    ])

X = np.array(X); Y = np.array(Y)

# --- Normalize to [-1,1] ---
in_scaler  = MinMaxScaler((-1,1))
out_scaler = MinMaxScaler((-1,1))
Xn = in_scaler.fit_transform(X)
Yn = out_scaler.fit_transform(Y)

joblib.dump(in_scaler,  INPUT_SCALER_PATH)
joblib.dump(out_scaler, OUTPUT_SCALER_PATH)

# Tensors
X_t = torch.tensor(Xn, dtype=torch.float32)
Y_t = torch.tensor(Yn, dtype=torch.float32)

# --- Model init ---
model = MemoryNeuralNetwork(
    number_of_input_neurons=5,
    number_of_hidden_neurons=60,
    number_of_output_neurons=3
).to(DEVICE)

# --- Epoch-based training ---
rmse_hist = []
for epoch in range(1, EPOCHS+1):
    perm = np.random.permutation(len(X_t))
    epoch_rmse = 0.0
    for i in perm:
        xi = X_t[i].cpu().numpy()
        yi = Y_t[i].cpu().numpy()
        pred = model.feedforward(xi)
        model.backprop(yi)
        # compute RMSE (normalized)
        epoch_rmse += torch.sqrt(torch.mean((pred - torch.tensor(yi, device=DEVICE))**2)).item()
    avg_rmse = epoch_rmse / len(X_t)
    rmse_hist.append(avg_rmse)
    print(f"Epoch {epoch}/{EPOCHS} — Avg RMSE: {avg_rmse:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --- Plot RMSE ---
plt.figure()
plt.plot(range(1,EPOCHS+1), rmse_hist, marker='o')
plt.title("Training RMSE (normalized)")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid()
plt.savefig(TRAIN_RMSE_PLOT)
plt.close()

# --- Evaluate full trajectory ---
# reset memory
for attr in ['prev_x_nn','prev_x_mn','prev_h_nn','prev_h_mn','prev_o_nn','prev_o_mn']:
    getattr(model, attr).zero_()

preds_n = np.array([ model.feedforward(Xn[i]).cpu().detach().numpy()
                     for i in range(len(Xn)) ])
preds = out_scaler.inverse_transform(preds_n)

# Plot XY trajectory
plt.figure()
plt.plot(Y[:,0], Y[:,1], label="True")
plt.plot(preds[:,0], preds[:,1], '--', label="Pred")
plt.title("XY Trajectory")
plt.legend(); plt.grid()
plt.savefig(TRAJECTORY_PLOT)
plt.close()

# Plot yaw
plt.figure()
plt.plot(Y[:,2], label="True")
plt.plot(preds[:,2], '--', label="Pred")
plt.title("Yaw over Time")
plt.legend(); plt.grid()
plt.savefig(YAW_PLOT)
plt.close()

print("Done: model, scalers, and plots saved.")
