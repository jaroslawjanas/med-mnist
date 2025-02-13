import os
from pprint import pprint
import numpy as np
from collections import defaultdict
import json
from typing import List, Literal, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import medmnist
from medmnist import INFO, Evaluator
from medmnist.dataset import MedMNIST


def info(datasets_name_list):
    
    for dataset_name in datasets_name_list:
    
        info = INFO[dataset_name]
        print(f"-"*100)
        print(f"Dataset:\t{dataset_name}")
        print(f"Python:\t{info['python_class']}")
        print(f"Task:\t{info['task']}")
        print(f"Number of classes:\t{len(info['label'])}")
        print(f"Channels:\t{info['n_channels']}")
        print("Classes:")
        pprint(info['label'])

def download(datasets_name_list, datasets_path, image_size):
    # Imagine size can be 28, 64, 128, 256
    
    for dataset_name in datasets_name_list:

        if image_size == 28:
            file_path = os.path.join(datasets_path, f"{dataset_name}_28.npz")
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping download.")
                continue

        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        # Note even though the split argument is set to train all splits are downloaded
        DataClass(root=datasets_path, download=True, size=image_size, split="train")

        # Rename from dataset.npz to dataset_28.npz to match 64+ naming scheme
        if image_size == 28:
            file_path = os.path.join(datasets_path, f"{dataset_name}.npz")
            new_file_path = os.path.join(datasets_path, f"{dataset_name}_28.npz")
            os.rename(file_path, new_file_path)



def make_paths_from_names(datasets_name_list, datasets_path, image_size=64):
    """
    Make paths from names
    """
    paths = []

    for dataset_name in datasets_name_list:
        path = os.path.join(datasets_path, f"{dataset_name}_{image_size}.npz")
        paths.append(path)

    return paths


def ensure_3channel_images(images):
    """
    Convert images to channels-first format (N, 3, H, W).
    
    Handles:
      - (N, H, W) -> grayscale without channel dimension.
      - (N, H, W, 1) or (N, H, W, 3) -> channels-last format.
      - (N, 1, H, W) or (N, 3, H, W) -> channels-first format.
    """
    if len(images.shape) == 3:
        # Grayscale images without channel dimension: (N, H, W)
        images = np.expand_dims(images, axis=1)  # becomes (N, 1, H, W)
        images = np.repeat(images, 3, axis=1)      # becomes (N, 3, H, W)
    elif len(images.shape) == 4:
        # Could be channels-first (N, C, H, W) or channels-last (N, H, W, C)
        if images.shape[1] == 1 or images.shape[1] == 3:
            # Likely channels-first format
            if images.shape[1] == 1:
                # Replicate channel if needed
                images = np.repeat(images, 3, axis=1)
            # Otherwise, already (N, 3, H, W)
        elif images.shape[-1] == 1 or images.shape[-1] == 3:
            # Likely channels-last format: (N, H, W, C)
            if images.shape[-1] == 1:
                images = np.repeat(images, 3, axis=-1)
            # Convert to channels-first
            images = np.transpose(images, (0, 3, 1, 2))
        else:
            raise ValueError("Unsupported channel dimension in image shape: " + str(images.shape))
    else:
        raise ValueError("Unsupported image shape: " + str(images.shape))
    return images


def unify_data(dataset_names: List[str],
               image_size: int,
               datasets_path: str,
               save_path: str,
               filename: str,
               are_unique_classes: bool=True
    ):
    """
    Unify data from multiple datasets and optionally save the unified dataset and class mapping.

    Args:
        names (list of str): List of dataset names.
        are_unique_classes (bool): Whether to ensure unique class labels for each dataset.
        save_path (str): Path to save the unified dataset and class mapping.

    Returns:
        dict: A dictionary of class mappings if `are_unique_classes` is True, else None.
    """

    # Define unified_dataset save path and check if the file already exists
    if save_path is not None and filename is not None:
        unified_dataset_path = os.path.join(save_path, f"{filename}.npz")

        if os.path.exists(unified_dataset_path):
            print(f"File {unified_dataset_path} already exists.")
            return None


    paths = make_paths_from_names(dataset_names, datasets_path, image_size)

    all_train_images = None
    all_train_labels = None
    all_test_images = None
    all_test_labels = None
    all_val_images = None
    all_val_labels = None

    class_mapping = defaultdict(dict) if are_unique_classes else None
    class_offset = 0

    for name, path in zip(dataset_names, paths):
        print(f"Dataset name: {name}")

        # Load the .npz file
        npz_file = np.load(path, mmap_mode="r")

        # Access specific arrays
        train_images = npz_file['train_images']
        train_labels = npz_file['train_labels']
        test_images = npz_file['test_images']
        test_labels = npz_file['test_labels']
        val_images = npz_file['val_images']
        val_labels = npz_file['val_labels']

        unique_train = np.unique(train_labels)
        unique_test = np.unique(test_labels)
        unique_val = np.unique(val_labels)

        all_unique_classes = np.unique(np.concatenate((unique_train, unique_test, unique_val)))
        print(f"Unique classes: {all_unique_classes}")

        if are_unique_classes:
            for _class in all_unique_classes:
                if _class not in class_mapping[name]:
                    class_mapping[name][int(_class)] = class_offset
                    class_offset += 1

            # Add class_offset int to all labels using numpy vectorization
            train_labels = np.vectorize(class_mapping[name].get)(train_labels)
            test_labels = np.vectorize(class_mapping[name].get)(test_labels)
            val_labels = np.vectorize(class_mapping[name].get)(val_labels)

            unique_train = np.unique(train_labels)
            unique_test = np.unique(test_labels)
            unique_val = np.unique(val_labels)
            all_unique_classes = np.unique(np.concatenate((unique_train, unique_test, unique_val)))
            print(f"After mapping: {all_unique_classes}")

        # Ensure grayscale images with one channel are converted to RGB-grayscale with three channels
        train_images = ensure_3channel_images(train_images)
        test_images = ensure_3channel_images(test_images)
        val_images = ensure_3channel_images(val_images)

        # Concatenate the data
        all_train_images = np.concatenate((all_train_images, train_images), axis=0) if all_train_images is not None else train_images
        all_test_images = np.concatenate((all_test_images, test_images), axis=0) if all_test_images is not None else test_images
        all_val_images = np.concatenate((all_val_images, val_images), axis=0) if all_val_images is not None else val_images

        all_train_labels = np.concatenate((all_train_labels, train_labels), axis=0) if all_train_labels is not None else train_labels
        all_test_labels = np.concatenate((all_test_labels, test_labels), axis=0) if all_test_labels is not None else test_labels
        all_val_labels = np.concatenate((all_val_labels, val_labels), axis=0) if all_val_labels is not None else val_labels

    print(f"All train images: {all_train_images.shape}")
    print(f"All train labels: {all_train_labels.shape}")
    print(f"All test images: {all_test_images.shape}")
    print(f"All test labels: {all_test_labels.shape}")
    print(f"All val images: {all_val_images.shape}")
    print(f"All val labels: {all_val_labels.shape}")

    # Save the unified dataset and mapping as files
    if unified_dataset_path:
        os.makedirs(save_path, exist_ok=True)
        np.savez(
            unified_dataset_path,
            train_images=all_train_images,
            train_labels=all_train_labels,
            test_images=all_test_images,
            test_labels=all_test_labels,
            val_images=all_val_images,
            val_labels=all_val_labels
        )
        print(f"Unified dataset saved at {unified_dataset_path}")

        if are_unique_classes:
            mapping_path = os.path.join(save_path, "class_mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(class_mapping, f, indent=4)
            print(f"Class mapping saved at {mapping_path}")

    # Clean up memory
    del all_train_images, all_train_labels, all_test_images, all_test_labels, all_val_images, all_val_labels
    del train_images, train_labels, test_images, test_labels, val_images, val_labels
    del npz_file

    return class_mapping if are_unique_classes else None


class NPZDataset(Dataset):
    def __init__(self, 
                 npz_path: str,
                 split: Literal["train", "val", "test"],
                 transform: Optional[v2.Compose] = None
    ):
        """
        Args:
            npz_path (str): Path to the .npz file.
            split (str): One of 'train', 'val', or 'test'. This determines which arrays to load.
            transform (callable, optional): A function/transform to apply to the images.
        """
        # Load NPZ file without keeping reference in memory
        data = np.load(npz_path, mmap_mode="r")  
        
        # Directly access images & labels (lazy loading using mmap)
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # Load a single sample from disk (memory-efficient)
        img = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()  # Ensure correct shape

        if self.transform:
            img = self.transform(img)

        return img, label