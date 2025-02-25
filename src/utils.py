import torch
from torch.utils.data import Dataset
from torch import Tensor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Tuple

def compute_class_and_dataset_weights(dataset: Dataset) -> Tuple[Tensor, Tensor]:
    """
    Computes inverse-frequency weights for both class labels and dataset labels.
    
    Assumes each sample from 'dataset' returns a tuple:
         (image, (class_label_tensor, dataset_label_tensor))
    and that the labels are global integers.
    
    Returns:
         (class_weights_tensor, dataset_weights_tensor)
    where each is a torch.Tensor of type float.
    
    Also visualizes counts and computed weights in a 2x2 subplot grid.
    """
    # Initialize counters for class and dataset labels
    class_counter = Counter()
    dataset_counter = Counter()
    
    for i in range(len(dataset)):
        # Unpack the labels: expect (class_label, dataset_label)
        _, (class_label, dataset_label) = dataset[i]
        class_counter[class_label.item()] += 1
        dataset_counter[dataset_label.item()] += 1

    print("Counts per class:", class_counter)
    print("Counts per dataset:", dataset_counter)
    
    # Compute weights for class labels:
    classes = sorted(class_counter.keys())
    class_counts = [class_counter[c] for c in classes]
    class_inv_freq = [1.0 / c for c in class_counts]
    class_mean_inv = sum(class_inv_freq) / len(class_inv_freq)
    class_weights = [w / class_mean_inv for w in class_inv_freq]
    
    # Compute weights for dataset labels:
    ds_labels = sorted(dataset_counter.keys())
    ds_counts = [dataset_counter[d] for d in ds_labels]
    ds_inv_freq = [1.0 / c for c in ds_counts]
    ds_mean_inv = sum(ds_inv_freq) / len(ds_inv_freq)
    dataset_weights = [w / ds_mean_inv for w in ds_inv_freq]
    
    # Plotting: Create a 2x2 grid of subplots.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Top-left: Class Frequencies
    axs[0, 0].bar(classes, class_counts, color="cornflowerblue")
    axs[0, 0].set_title("Class Frequencies")
    axs[0, 0].set_xlabel("Class")
    axs[0, 0].set_ylabel("Count")
    
    # Top-right: Dataset Frequencies
    axs[0, 1].bar(ds_labels, ds_counts, color="cornflowerblue")
    axs[0, 1].set_title("Dataset Frequencies")
    axs[0, 1].set_xlabel("Dataset Label")
    axs[0, 1].set_ylabel("Count")
    
    # Bottom-left: Class Weights
    axs[1, 0].bar(classes, class_weights, color="orange")
    axs[1, 0].set_title("Class Weights")
    axs[1, 0].set_xlabel("Class")
    axs[1, 0].set_ylabel("Weight")
    
    # Bottom-right: Dataset Weights
    axs[1, 1].bar(ds_labels, dataset_weights, color="orange")
    axs[1, 1].set_title("Dataset Weights")
    axs[1, 1].set_xlabel("Dataset Label")
    axs[1, 1].set_ylabel("Weight")
    
    plt.tight_layout()
    plt.show()
    
    # Convert computed weights to GPU tensors (if needed)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    dataset_weights_tensor = torch.tensor(dataset_weights, dtype=torch.float)
    
    return class_weights_tensor, dataset_weights_tensor


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


def _load_metrics_csv(log_dir: str):
    """
    Helper function to load the metrics.csv file from a Lightning logs directory
    and determine whether to use epoch or step as the x-axis.
    """
    csv_path = os.path.join(log_dir, "metrics.csv")
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].astype(int)
        x_col = "epoch"
        x_label = "Epoch"
    else:
        x_col = "step"
        x_label = "Step"
    return df, x_col, x_label

def _plot_metric(ax, df, metric: str, x_col: str, smooth_window: int = 1):
    """
    Helper function to plot a single metric on a given axes (ax).
    Sorts the DataFrame, optionally applies smoothing, and plots the metric.
    """
    if metric not in df.columns:
        return  # metric doesn't exist, skip
    df_metric = df.dropna(subset=[metric]).copy()
    df_metric.sort_values(x_col, inplace=True)
    if smooth_window > 1:
        df_metric[metric] = df_metric[metric].rolling(window=smooth_window, min_periods=1).mean()
    x_vals = df_metric[x_col].values
    y_vals = df_metric[metric].values
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=metric)

def display_main_loss(log_dir: str, smooth_window: int = 1):
    """
    Plots a single graph with train_loss and val_loss.
    """
    df, x_col, x_label = _load_metrics_csv(log_dir)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_metric(ax, df, "train_loss", x_col, smooth_window)
    _plot_metric(ax, df, "val_loss",   x_col, smooth_window)
    
    ax.set_title("Main Loss")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    ax.legend(loc='upper left')
    
    # Set y-limits based on combined loss data
    loss_data = []
    for metric in ["train_loss", "val_loss"]:
        if metric in df.columns:
            loss_data.extend(df[metric].dropna().tolist())
    if loss_data:
        ax.set_ylim(0, max(loss_data)*1.1)
    
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax.set_xticks(unique_epochs)
    
    plt.tight_layout()
    plt.show()
    
    save_path = os.path.join(log_dir, "main_loss.png")
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")

def display_class_metrics(log_dir: str, smooth_window: int = 1):
    """
    Creates a figure with 2 subplots:
      1) train_loss_class vs val_loss_class
      2) train_acc_class vs val_acc_class
    """
    df, x_col, x_label = _load_metrics_csv(log_dir)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: class losses
    ax_loss = axs[0]
    _plot_metric(ax_loss, df, "train_loss_class", x_col, smooth_window)
    _plot_metric(ax_loss, df, "val_loss_class",   x_col, smooth_window)
    ax_loss.set_title("Class Loss")
    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc='upper left')
    
    # Optionally set common y-limits for class losses
    loss_vals = []
    for metric in ["train_loss_class", "val_loss_class"]:
        if metric in df.columns:
            loss_vals.extend(df[metric].dropna().tolist())
    if loss_vals:
        ax_loss.set_ylim(0, max(loss_vals)*1.1)
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax_loss.set_xticks(unique_epochs)

    # Right subplot: class accuracies
    ax_acc = axs[1]
    _plot_metric(ax_acc, df, "train_acc_class", x_col, smooth_window)
    _plot_metric(ax_acc, df, "val_acc_class",   x_col, smooth_window)
    ax_acc.set_title("Class Accuracy")
    ax_acc.set_xlabel(x_label)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc='upper left')
    # Force y-axis range for accuracies to [0, 1]
    ax_acc.set_ylim(0, 1)
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax_acc.set_xticks(unique_epochs)

    plt.tight_layout()
    plt.show()

    save_path = os.path.join(log_dir, "class_metrics.png")
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")

def display_dataset_metrics(log_dir: str, smooth_window: int = 1):
    """
    Creates a figure with 2 subplots:
      1) train_loss_dataset vs val_loss_dataset
      2) train_acc_dataset vs val_acc_dataset
    """
    df, x_col, x_label = _load_metrics_csv(log_dir)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: dataset losses
    ax_loss = axs[0]
    _plot_metric(ax_loss, df, "train_loss_dataset", x_col, smooth_window)
    _plot_metric(ax_loss, df, "val_loss_dataset",   x_col, smooth_window)
    ax_loss.set_title("Dataset Loss")
    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc='upper left')
    loss_vals = []
    for metric in ["train_loss_dataset", "val_loss_dataset"]:
        if metric in df.columns:
            loss_vals.extend(df[metric].dropna().tolist())
    if loss_vals:
        ax_loss.set_ylim(0, max(loss_vals)*1.1)
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax_loss.set_xticks(unique_epochs)

    # Right subplot: dataset accuracies
    ax_acc = axs[1]
    _plot_metric(ax_acc, df, "train_acc_dataset", x_col, smooth_window)
    _plot_metric(ax_acc, df, "val_acc_dataset",   x_col, smooth_window)
    ax_acc.set_title("Dataset Accuracy")
    ax_acc.set_xlabel(x_label)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc='upper left')
    # Set y-range for accuracies
    ax_acc.set_ylim(0, 1)
    if x_label == "Epoch":
        unique_epochs = sorted(df[x_col].dropna().unique())
        ax_acc.set_xticks(unique_epochs)

    plt.tight_layout()
    plt.show()

    save_path = os.path.join(log_dir, "dataset_metrics.png")
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")