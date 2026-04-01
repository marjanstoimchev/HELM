import os
import re
import glob
import pandas as pd
from natsort import natsorted
from PIL import Image
import numpy as np

class DFC_15ImageReader:
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

# DFC_15 Dataset Class
class DFC_15:
    def __init__(self, dir_="../rs_datasets/mlc/DFC_15"):
        self.dir_ = dir_
        self.labels_path = os.path.join(dir_, 'multilabel.txt')

    def prepare_dataset(self):
        # Load labels
        df = pd.read_csv(self.labels_path, delimiter="\t")
        df.rename(columns={'IMAGE\\LABEL': 'image'}, inplace=True)
        df['one_hot'] = df[df.columns[1:]].values.tolist()
        df.drop(columns=list(df.columns[1:-1]), inplace=True)

        # Load image paths for train and test
        images_train = natsorted(glob.glob(os.path.join(self.dir_, 'images_train', '*.png')))
        images_test = natsorted(glob.glob(os.path.join(self.dir_, 'images_test', '*.png')))

        targets_train = [int(re.split(r'(\d+)', file.split('/')[-1])[1]) - 1 for file in images_train]
        targets_test = [int(re.split(r'(\d+)', file.split('/')[-1])[1]) - 1 for file in images_test]

        df_train = df.iloc[targets_train].reset_index(drop=True)
        df_test  = df.iloc[targets_test].reset_index(drop=True)

        df_train['image'] = images_train
        df_train['split'] = 'train'

        df_test['image'] = images_test
        df_test['split'] = 'test'

        # Combine into a single DataFrame
        df = pd.concat([df_train, df_test], ignore_index=True)
        return df.reset_index(drop=True)