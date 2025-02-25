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


def plot_and_save_metrics(path: str):
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
    
    # Define the metrics to plot as (column_name, title)
    metrics_to_plot = [
        ("train_loss", "Train Loss"),
        ("val_loss",   "Validation Loss"),
        ("train_acc",  "Train Accuracy"),
        ("val_acc",    "Validation Accuracy"),
    ]
    
    # Pre-calculate global y-limits for losses and accuracies
    loss_columns = [col for col, title in metrics_to_plot if "loss" in col]
    acc_columns = [col for col, title in metrics_to_plot if "acc" in col]

    loss_vals = []
    for col in loss_columns:
        if col in df.columns:
            loss_vals.extend(df[col].dropna().tolist())
    acc_vals = []
    for col in acc_columns:
        if col in df.columns:
            acc_vals.extend(df[col].dropna().tolist())
    
    # Set common limits if data is available
    if loss_vals:
        global_loss_max = max(loss_vals)
    else:
        global_loss_max = None
        
    if acc_vals:
        global_acc_max = max(acc_vals)
    else:
        global_acc_max = None

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for ax, (col, title) in zip(axs, metrics_to_plot):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not found", ha="center", va="center")
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(title)
            continue
        
        # Filter out rows where this metric is NaN and sort by x_col
        df_metric = df.dropna(subset=[col]).copy()
        df_metric.sort_values(x_col, inplace=True)

        if df_metric.empty:
            ax.text(0.5, 0.5, f"No data for '{col}'", ha="center", va="center")
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(title)
        else:
            x_vals = df_metric[x_col].values
            y_vals = df_metric[col].values
            ax.plot(x_vals, y_vals, marker='o', linestyle='-')
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(title)
            
            # Set integer ticks if x is epoch
            if x_label == "Epoch":
                unique_epochs = sorted(df_metric[x_col].unique())
                ax.set_xticks(unique_epochs)
            
            # Set common y-limits for losses and accuracies
            if "loss" in col and global_loss_max is not None:
                ax.set_ylim(0, global_loss_max*1.1)
            elif "acc" in col and global_acc_max is not None:
                ax.set_ylim(0, global_acc_max*1.1)

    plt.tight_layout()
    plt.show()
    
    # Save the figure to the same location as the logs
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")