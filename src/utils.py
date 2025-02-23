import torch
from torch.utils.data import Dataset
from torch import Tensor

from collections import Counter
import matplotlib.pyplot as plt

def compute_class_weights(dataset: Dataset, device: str) -> Tensor:

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
    return torch.tensor(weights, dtype=torch.float).to(device)
