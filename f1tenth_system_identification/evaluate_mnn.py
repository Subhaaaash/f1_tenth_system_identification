import pandas as pd
import torch
import matplotlib.pyplot as plt
from mnn_model import MNN

df = pd.read_csv("levine.csv")

x_vals = df["x"].values
y_vals = df["y"].values
yaw_vals = df["yaw"].values
steer_vals = df["steering_angle"].values
vel_vals = df["velocity"].values

model = MNN()
model.load_state_dict(torch.load("mnn_model.pth"))
model.eval()
model.reset_memory()

predicted = []

for k in range(1, len(df)):
    input_k = torch.tensor([steer_vals[k], vel_vals[k], x_vals[k-1], y_vals[k-1], yaw_vals[k-1]],
                           dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_k)
    predicted.append(output.numpy().flatten())

predicted = np.array(predicted)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x_vals[1:], label="True X")
plt.plot(predicted[:, 0], label="Pred X", linestyle="--")
plt.legend()
plt.title("X Position")

plt.subplot(1, 3, 2)
plt.plot(y_vals[1:], label="True Y")
plt.plot(predicted[:, 1], label="Pred Y", linestyle="--")
plt.legend()
plt.title("Y Position")

plt.subplot(1, 3, 3)
plt.plot(yaw_vals[1:], label="True Yaw")
plt.plot(predicted[:, 2], label="Pred Yaw", linestyle="--")
plt.legend()
plt.title("Yaw")

plt.tight_layout()
plt.show()
