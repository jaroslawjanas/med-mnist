import torch
from torch.utils.data import Dataset
from torch import Tensor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_class_weights(dataset: Dataset) -> Tensor:

    # 1. Count how many times each class occurs
    class_counter = Counter()
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counter[label.item()] += 1

    # Print class counts
    print("Counts per class:", class_counter)

    # 2. Build a list of class counts in ascending class index order
    #    (assuming classes go from 0..N-1; adjust if your labels differ)
    num_classes = len(class_counter)
    classes = list(range(num_classes))
    counts = [class_counter[c] for c in classes]

    # 3. Compute inverse-frequency weights
    inv_freq = [1.0 / c for c in counts]
    mean_inv = sum(inv_freq) / len(inv_freq)  # normalization factor
    weights = [w / mean_inv for w in inv_freq] # optional: makes average weight=1

    # 4. Visualize both counts and weights
    plt.figure(figsize=(10, 4))

    # Plot class counts
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts, color="cornflowerblue")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Frequencies")

    # Plot class weights
    plt.subplot(1, 2, 2)
    plt.bar(classes, weights, color="orange")
    plt.xlabel("Class")
    plt.ylabel("Weight")
    plt.title("Class Weights")

    plt.tight_layout()
    plt.show()

    # 5. Return weights as a GPU tensor
    return torch.tensor(weights, dtype=torch.float)


custom_theme = RichProgressBarTheme(
    description="#a6a000",
    progress_bar="#00ff06",
    progress_bar_finished="#aaffba",
    progress_bar_pulse="#39ff00",
    batch_progress="#00b7bb",
    time="#a6a000",
    processing_speed="#00b7bb",
    metrics="#025aeb",
)


def plot_and_save_metrics(path: str, smooth_window: int = 1):
    csv_path = os.path.join(path, "metrics.csv")
    save_path = os.path.join(path, "metrics_plot.png")

    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Use 'epoch' if available, otherwise 'step'
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].astype(int)
        x_col = "epoch"
        x_label = "Epoch"
    else:
        x_col = "step"
        x_label = "Step"
    
    # Define metric groups
    loss_metrics = ["train_loss", "val_loss"]
    acc_metrics  = ["train_acc", "val_acc"]
    
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot Losses ---
    ax = axs[0]
    loss_data = {}  # to collect all loss values for common y-axis limits
    for metric in loss_metrics:
        if metric in df.columns:
            df_metric = df.dropna(subset=[metric]).copy()
            df_metric.sort_values(x_col, inplace=True)
            if smooth_window > 1:
                df_metric[metric] = df_metric[metric].rolling(window=smooth_window, min_periods=1).mean()
            x_vals = df_metric[x_col].values
            y_vals = df_metric[metric].values
            loss_data[metric] = y_vals
            ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=metric)
    ax.set_title("Loss")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    # Set common y-limits if any loss data exists
    if loss_data:
        combined_losses = [val for arr in loss_data.values() for val in arr]
        y_max = max(combined_losses) * 1.1
        ax.set_ylim(0, y_max)
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax.set_xticks(unique_epochs)
    ax.legend(loc='upper left')
    
    # --- Plot Accuracies ---
    ax = axs[1]
    acc_data = {}  # to collect all accuracy values for common y-axis limits
    for metric in acc_metrics:
        if metric in df.columns:
            df_metric = df.dropna(subset=[metric]).copy()
            df_metric.sort_values(x_col, inplace=True)
            if smooth_window > 1:
                df_metric[metric] = df_metric[metric].rolling(window=smooth_window, min_periods=1).mean()
            x_vals = df_metric[x_col].values
            y_vals = df_metric[metric].values
            acc_data[metric] = y_vals
            ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=metric)
    ax.set_title("Accuracy")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    if acc_data:
        combined_accs = [val for arr in acc_data.values() for val in arr]
        y_max = max(combined_accs) * 1.1
        ax.set_ylim(0, y_max)
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax.set_xticks(unique_epochs)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure to the same location as the logs
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")