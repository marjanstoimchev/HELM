import os
import numpy as np
import pandas as pd
from PIL import Image
import torchxrayvision as xrv

class PadChestImageReader:
    def __init__(self):
        pass

    def read_image(self, row):
        """
        Reads a chest X-ray image, applies min-max normalization, 
        and repeats the single channel image across 3 channels (RGB-like).

        Args:
            row (dict): Dictionary containing the 'image' key with the image file path.

        Returns:
            np.ndarray: The image as a 3-channel (repeated) NumPy array.
        """
        image_data = row.get('ImageID')
        if not image_data:
            raise ValueError("Image path is missing.")
        
        # Open the image without converting to RGB (keeps original single channel)
        image = np.array(Image.open(image_data))

        # Check for boolean images or uniform images
        if image.dtype == np.bool_:
            # If boolean, assume it's an all-black image and set to zero
            image = np.zeros_like(image, dtype=np.uint8)
        elif image.max() == image.min():
            # If uniform (all pixels have the same value), set to zero
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            # Apply min-max normalization for non-boolean, non-uniform images
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
        
        # Repeat the single channel image across 3 channels
        image = np.stack([image] * 3, axis=-1)

        return image

# class PadChest:
#     def __init__(self, dir_='../Datasets/mlc_datasets/PC/images-224'):
#         self.dir_ = dir_
#         self.df_pc = None
    
#     def prepare_dataset(self):
#         df_pc = xrv.datasets.PC_Dataset(imgpath=self.dir_, unique_patients = True)  # Load dataset using xrv
#         columns = ['image', *df_pc.pathologies]  # Column names: 'image' + pathologies
#         # Create new matrix with image IDs and pathology labels
#         new_matrix = np.column_stack((np.array(df_pc.csv.ImageID.values).reshape(-1, 1), df_pc.labels.astype(int)))
#         df = pd.DataFrame(new_matrix, columns=columns)  # Create DataFrame
#         df['image'] = df['image'].apply(lambda x: os.path.join(self.dir_, x))  # Update image paths
        
#         # Create one_hot column
#         df['one_hot'] = df.values[:, 1:].tolist()  # Skip 'image' column
#         df['split'] = 'train'  # No predefined test set, mark all as train
        
#         self.df_pc = df_pc  # Store the dataset object for future use
#         return df.reset_index(drop=True)
    

# class PadChest:
#     def __init__(self, dir_='/cache/marjan/padchest/'):
#         self.dir_ = dir_
#         self.df_pc = None
    
#     def prepare_dataset(self):
#         dfs = []
#         for split in ['train', 'test', 'val']:
#             df = pd.read_csv(os.path.join(self.dir_, f'labels/splits/{split}.csv'))
#             df['split'] = split
#             df['ImageID'] = self.dir_ + 'images/' + df['ImageID']
#             dfs.append(df)
        
#         dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
#         # Create one_hot column
#         dfs['one_hot'] = dfs.values[:, 3:-1].tolist()  # Skip 'image' column
#         self.df_pc = dfs  # Store the dataset object for future use
#         return dfs
    
class PadChest:
    def __init__(self, dir_='/cache/marjan/padchest/'):
        self.dir_ = dir_
        self.df_pc = None

    def prepare_dataset(self):
        dfs = []
        for split in ['train', 'test', 'val']:
            df = pd.read_csv(os.path.join(self.dir_, f'labels/splits/{split}.csv'))
            df['split'] = split
            df['ImageID'] = self.dir_ + 'images/' + df['ImageID']
            dfs.append(df)

        # Combine all splits into a single DataFrame
        dfs = pd.concat(dfs, axis=0).reset_index(drop=True)

        # Identify label columns (before adding 'normal', from index 3 to -1)
        label_columns = dfs.columns[3:-1]

        # Add a 'normal' column: 1 if all labels are 0, else 0
        dfs['normal'] = dfs[label_columns].sum(axis=1).apply(lambda x: 1 if x == 0 else 0)

        # Reorder columns to place 'normal' before 'split'
        columns_order = list(dfs.columns[:3]) + ['normal'] + list(dfs.columns[3:-2]) + ['split']
        dfs = dfs[columns_order]

        # Update label_columns after adding 'normal' (from index 3 to -2)
    
        # Update the 'one_hot' column to include the 'normal' class
        dfs['one_hot'] = dfs.apply(
            lambda row: [row['normal']] + list(row[label_columns]),
            axis=1
        )

        self.df_pc = dfs
        return dfs