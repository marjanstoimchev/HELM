import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

class MuRedImageReader:
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
    
# MuRed Dataset Class
class MuRed:
    def __init__(self, dir_='./Datasets/mlc_datasets/MuReD/'):
        self.dir_ = Path(dir_)

        self.label_names = [
            'DR',
            'NORMAL',
            'MH',
            'ODC',
            'TSLN',
            'ARMD',
            'DN',
            'MYA',
            'BRVO',
            'ODP',
            'CRVO',
            'CNV',
            'RS',
            'ODE',
            'LS',
            'CSR',
            'HTR',
            'ASR',
            'CRS',
            'OTHER']
        
    def prepare_dataset(self):
        images_path = self.dir_ / 'images'
        train_path = next(self.dir_.rglob('train_data_modified*.csv'))
        test_path = next(self.dir_.rglob('test_data_modified*.csv'))

        # Load train and test CSV files
        df_train = pd.read_csv(train_path, index_col=0)
        df_test = pd.read_csv(test_path, index_col=0)

        df_train['image'] = df_train['ID_2'].apply(lambda x: str(images_path / x))
        df_test['image'] = df_test['ID_2'].apply(lambda x: str(images_path / x))

        df_train['one_hot'] = df_train[self.label_names].values.tolist()
        df_test['one_hot'] = df_test[self.label_names].values.tolist()

        # Mark data as train or test and combine into one DataFrame
        df_train['split'] = 'train'
        df_test['split'] = 'test'
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        return df.reset_index(drop=True)
