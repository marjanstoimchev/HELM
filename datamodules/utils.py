import os
import torch
import numpy as np
from data.utils import get_image_reader
from datamodules.base_datamodule import ImageDataModule

def print_with_lines(text):
    line = '-' * len(text)
    print(f"{line}\n{text}\n{line}")

def load_images(df, dataset_name, image_size=224, batch_size=32, num_workers=16, path='./Datasets/mlc_datasets_npy'):
    print_with_lines(f"Preparing dataset for {dataset_name}")
    # Directory path for storing images and labels
    dir_ = os.path.join(path, dataset_name)
    images_path = os.path.join(dir_, 'images.npy')
    labels_path = os.path.join(dir_, 'labels.npy')

    # Load from disk if already saved
    if os.path.exists(images_path) and os.path.exists(labels_path):
        try:
            print_with_lines(f"Loading images and labels from {dir_}")
            images = np.load(images_path)
            labels = np.load(labels_path)
            print(f"Images and Labels loaded successfully for the {dataset_name} dataset!")
            print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
            print_with_lines("Load completed")
            return images, labels
        except Exception as e:
            print(f"Error loading images or labels: {e}")
            raise
    else:
        print_with_lines(f"Directory {dir_} does not exist. Creating directory and processing images...")        
        os.makedirs(dir_, exist_ok=True)

    # Load images and labels from the data module
    try:
        print_with_lines(f"Starting image and label processing for {dataset_name}")
        images, labels = process_and_load_images(df, image_size, batch_size, num_workers, dataset_name)
        print(f"Images and Labels processed successfully!")
        save_data(images, labels, images_path, labels_path)
        print_with_lines(f"Images and Labels saved to {dir_}")
    except Exception as e:
        print(f"Error processing images or labels: {e}")
        raise

    print(f"Final Images shape: {images.shape}, Final Labels shape: {labels.shape}")
    print_with_lines("Process completed")

    return images, labels

def process_and_load_images(df, image_size, batch_size, num_workers, dataset_name):
    print_with_lines(f"Setting up ImageDataModule for {dataset_name}")
    data_module = ImageDataModule(
        df, 
        image_size, 
        batch_size, 
        num_workers,
        image_reader=get_image_reader(dataset_name)
    )
    data_module.setup()

    print_with_lines("Processing images and labels...")
    dataset = data_module.dataset
    images = torch.cat([x.unsqueeze(0) for x in dataset.images]).numpy()
    labels = torch.cat([x.unsqueeze(0) for x in dataset.labels]).float().numpy()

    print(f"Processed {len(images)} images and {len(labels)} labels.")
    print_with_lines("Processing completed")
    return images, labels

def save_data(images, labels, images_path, labels_path):
    print_with_lines(f"Saving images to {images_path} and labels to {labels_path}")
    np.save(images_path, images)
    np.save(labels_path, labels)
    print("Images and labels saved successfully.")
    print_with_lines("Save completed")

