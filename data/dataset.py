import torch
import numpy as np
from tqdm import tqdm

class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_images, labeled_labels, labeled_h_labels, unlabeled_images=None, transforms=None):
        """
        Initializes the semi-supervised dataset.

        Args:
            labeled_images (list or np.array): Array or list of labeled images.
            labeled_labels (list or np.array): Labels corresponding to the labeled images.
            unlabeled_images (list or np.array, optional): Array or list of unlabeled images. Default is None.
            transforms (callable, optional): A function/transform to apply to the images.
            image_reader (callable, optional): A function/class to read images from the dataset. Default is None.
        """
        self.transforms = transforms
        self.labeled_images = labeled_images
        self.labeled_labels = labeled_labels
        self.labeled_h_labels = labeled_h_labels

        # If unlabeled images are provided, resample labeled images with repetition to match the size of unlabeled data
        if unlabeled_images is not None:
            indices = np.random.choice(len(labeled_images), size=len(unlabeled_images), replace=True)
            self.labeled_images = np.array(self.labeled_images)[indices]
            self.labeled_labels = np.array(self.labeled_labels)[indices]
            self.labeled_h_labels = np.array(self.labeled_h_labels)[indices]
            self.unlabeled_images = unlabeled_images
        else:
            self.unlabeled_images = None

    def __len__(self):
        """The length should match the longer of the two datasets (labeled or unlabeled)."""
        return len(self.labeled_images)

    def __getitem__(self, idx):
        outputs = {}
        # Read the labeled image
        x = np.moveaxis(self.labeled_images[idx], 0, -1)

        # Extract the corresponding labels
        one_hot = torch.tensor(self.labeled_labels[idx]).float()
        h_one_hot = torch.tensor(self.labeled_h_labels[idx]).float()

        # Handle unlabeled data if it exists
        if self.unlabeled_images is not None:
            u = np.moveaxis(self.unlabeled_images[idx], 0, -1)
            if self.transforms:
                u = self.transforms(u)
            outputs['u'] = u

        # Apply transforms if they are provided
        if self.transforms:
            x = self.transforms(x)

        outputs['x'] = x
        outputs['one_hot'] = one_hot
        outputs['h_one_hot'] = h_one_hot
        return outputs
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, image_reader=None):
        self.images = []
        self.labels = []        

        # Load all images into RAM during initialization
        for idx in tqdm(range(len(df)), desc=f"Loading images and labels ---: ", position=0, leave=True):
            label = df.iloc[idx]['one_hot']
            label = torch.tensor(label)
            img = image_reader.read_image(df.loc[idx])

            if transform:
                img = transform(image = img)['image']
                
            self.images.append(img)
            # Append the label (assuming one-hot encoding)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return image and label from RAM
        return self.images[idx], self.labels[idx]