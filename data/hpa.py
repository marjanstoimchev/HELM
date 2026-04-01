import cv2, os
import numpy as np
import pandas as pd

class HPAImageReader:
    def __init__(self):
        pass

    def read_image(self, row):
        """
        Reads an HPA multi-channel image by stacking three grayscale images (red, green, blue) into an RGB NumPy array.

        Args:
            row (dict): Dictionary containing the 'image' key with the base image path (without _red, _green, _blue).

        Returns:
            np.ndarray: A 3-channel RGB NumPy array.
        """
        image_data = row.get('image')
        if not image_data:
            raise ValueError("Base image path is missing.")
        
        colors = ['red', 'green', 'blue']
        # Stack grayscale images for red, green, blue channels
        img = [cv2.imread(f"{image_data}_{color}.png", cv2.IMREAD_GRAYSCALE) for color in colors]
        
        # Stack the images into an RGB format (if all images are found)
        return np.stack(img, axis=-1)
    

# HPA Dataset Class
class HPA:
    def __init__(self, dir_='./Datasets/mlc_datasets/HPA'):
        self.dir_ = dir_
        self.images_path = os.path.join(self.dir_, 'train')

        self.label_names = {
            0:  "Nucleoplasm",  
            1:  "Nuclear membrane",   
            2:  "Nucleoli",   
            3:  "Nucleoli fibrillar center",   
            4:  "Nuclear speckles",
            5:  "Nuclear bodies",   
            6:  "Endoplasmic reticulum",   
            7:  "Golgi apparatus",   
            8:  "Peroxisomes",   
            9:  "Endosomes",   
            10:  "Lysosomes",   
            11:  "Intermediate filaments",   
            12:  "Actin filaments",   
            13:  "Focal adhesion sites",   
            14:  "Microtubules",   
            15:  "Microtubule ends",   
            16:  "Cytokinetic bridge",   
            17:  "Mitotic spindle",   
            18:  "Microtubule organizing center",   
            19:  "Centrosome",   
            20:  "Lipid droplets",   
            21:  "Plasma membrane",   
            22:  "Cell junctions",   
            23:  "Mitochondria",   
            24:  "Aggresome",   
            25:  "Cytosol",   
            26:  "Cytoplasmic bodies",   
            27:  "Rods & rings"
        }

    def fill_targets(self, row):
        # Check if row.Target is a valid string and not empty
        if isinstance(row.Target, str) and len(row.Target) > 0:
            # Split the string to create an array of integers
            target_array = np.array(row.Target.split()).astype(np.int64)
            for num in target_array:
                name = self.label_names[int(num)]
                row[name] = 1
        return row

    def prepare_dataset(self):
        # Load label CSV
        df = pd.read_csv(os.path.join(*[self.dir_, 'train.csv']))

        for key in self.label_names.keys():
            df[self.label_names[key]] = 0

        df = df.apply(self.fill_targets, axis=1)

        # Add image paths
        df['image'] = df['Id'].apply(lambda x: os.path.join(self.images_path, x))

        label_columns = df.columns.difference(['Id', 'image', 'split', 'Target'])  # Exclude non-label columns

        df[label_columns] = df[label_columns].astype(int)

        # Create one_hot column
        df['one_hot'] = df[label_columns].values.tolist()
        df['split'] = 'full'  # No predefined test set, mark all as train
        return df.reset_index(drop=True)
