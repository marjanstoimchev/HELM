import os
import glob
import numpy as np
from PIL import Image
import pandas as pd

class ChestXray8ImageReader:
    def __init__(self):
        pass

    def read_image(self, row):
        """
        Reads a chest X-ray image and converts it to an RGB NumPy array.

        Args:
            row (dict): Dictionary containing the 'image' key with the image file path.

        Returns:
            np.ndarray: The image as an RGB NumPy array.
        """
        image_data = row.get('image')
        if not image_data:
            raise ValueError("Image path is missing.")
        
        # Open and convert to RGB format
        image = Image.open(image_data).convert('RGB')
        return np.array(image)
    
# ChestXray8 Dataset Class
class ChestXray8:
    def __init__(self, dir_='./Datasets/mlc_datasets/ChestX-ray8'):
        self.dir_ = dir_
        self.labels_path = os.path.join(self.dir_, 'labels')
        self.images_train = os.path.join(self.dir_, 'images')

        self.label_names = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Effusion',
            'Emphysema',
            'Fibrosis',
            'Hernia',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pleural Thickening',
            'Pneumonia',
            'Pneumothorax',
            'Pneumoperitoneum',
            'Pneumomediastinum',
            'Subcutaneous Emphysema',
            'Tortuous Aorta',
            'Calcification of the Aorta',
            'No Finding'
            ]

    def prepare_dataset(self):
        # Load label CSV
        label_file = glob.glob(os.path.join(self.labels_path, '**/*.csv'), recursive=True)[0]
        df = pd.read_csv(label_file)
        df.drop('subj_id', axis=1, inplace=True)  # Remove subj_id

        # Add image paths
        df['image'] = df['id'].apply(lambda x: os.path.join(self.images_train, x))

        # Check if the image file exists in the folder
        df['image_exists'] = df['image'].apply(lambda x: os.path.exists(x))

        # Filter rows where the image file exists
        df = df[df['image_exists']]

        # Drop the 'image_exists' column after filtering
        df.drop('image_exists', axis=1, inplace=True)

        # Ensure subj_id is excluded from one_hot
        # label_columns = df.columns.difference(['id', 'image', 'split'])
        df[self.label_names] = df[self.label_names].astype(int)

        # Create one_hot column
        df['one_hot'] = df[self.label_names].values.tolist()
        df['split'] = 'train'  # No predefined test set, mark all as train

        return df.reset_index(drop=True)
