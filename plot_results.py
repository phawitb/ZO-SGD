import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train_log_summary.csv")

def plot_and_save(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8,5))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

plot_and_save(
    df["Epoch"],
    df["Accuracy"],
    "Epoch",
    "Accuracy",
    "Accuracy vs Epoch",
    "accuracy_vs_epoch.png"
)

plot_and_save(
    df["Epoch"],
    df["Loss"],
    "Epoch",
    "Loss",
    "Loss vs Epoch",
    "loss_vs_epoch.png"
)

plot_and_save(
    df["Epoch"],
    df["Grad_scalar"],
    "Epoch",
    "Grad Scalar",
    "Grad Scalar vs Epoch",
    "grad_scalar_vs_epoch.png"
)
