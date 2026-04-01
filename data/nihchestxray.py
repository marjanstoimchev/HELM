import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

class NIHChestXrayImageReader:
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

# class NIHChestXray:
#     def __init__(self, dir_='./Datasets/mlc_datasets/NIH-Chest-X-ray'):
#         self.dir_ = dir_
#         self.images = glob.glob(f'{self.dir_}/images*/images/*.png')

#         path_train  = os.path.join(*[self.dir_, 'train_val_list.txt'])
#         path_test  = os.path.join(*[self.dir_, 'test_list.txt'])

#         self.train_images = open(path_train,'r').read().splitlines()
#         self.test_images = open(path_test,'r').read().splitlines()

#         self.label_names = [
#             'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
#             'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
#             'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding'
#             ]

#     def prepare_dataset(self):
#         # Load label CSV
#         df = pd.read_csv(f'{self.dir_}/Data_Entry_2017.csv')

#         df.drop("Unnamed: 11", axis=1, inplace=True)

#         for label in self.label_names:
#             df[label] = df['Finding Labels'].map(lambda result: 1 if label in result else 0)

#         full_img_paths = {os.path.basename(x): x for x in self.images}
#         df['image'] = df['Image Index'].map(full_img_paths.get)
        
#         df_train = df[df["Image Index"].isin(self.train_images)]
#         df_train = df_train.copy()
#         df_train['split'] = 'train'  # No predefined test set, mark all as train

#         df_test = df[df["Image Index"].isin(self.test_images)]
#         df_test = df_test.copy()
#         df_test['split'] = 'test'  # No predefined test set, mark all as train

#         df = pd.concat([df_train, df_test], ignore_index=True)
        
#         # df = df[df["Finding Labels"] != "No Finding"]
#         df[self.label_names] = df[self.label_names].astype(int)
#         # Create one_hot column
#         df['one_hot'] = df[self.label_names].values.tolist()

#         return df.reset_index(drop=True)


# class NIHChestXray:
#     def __init__(self, path='./Datasets/mlc_datasets/NIH-Chest-X-ray/'):
#         self.path = path

#         self.availabel_label_names = [
#             'Atelectasis',
#             'Cardiomegaly',
#             'Consolidation',
#             'Edema',
#             'Effusion',
#             'Emphysema',
#             'Fibrosis',
#             'Hernia',
#             'Infiltration',
#             'Mass',
#             'Nodule',
#             'Pleural_Thickening',
#             'Pneumonia',
#             'Pneumothorax',
#             'Pneumoperitoneum',
#             'Pneumomediastinum',
#             'Subcutaneous Emphysema',
#             'Tortuous Aorta',
#             'Calcification of the Aorta',
#             'No Finding'
#             ]
        
#         self.label_names = [
#             'Atelectasis',
#             'Consolidation',
#             'Infiltration',
#             'Pneumothorax',
#             'Edema',
#             'Emphysema',
#             'Fibrosis',
#             'Effusion',
#             'Pneumonia',
#             'Pleural_Thickening',
#             'Cardiomegaly', 
#             'Nodule',
#             'Mass',
#             'Hernia',
#             'No Finding'
#         ]
    
#     def to_one_hot(self, df):
#         multi_hot_matrix = df['Finding Labels'].str.get_dummies(sep='|')
#         multi_hot_matrix = multi_hot_matrix.reindex(columns=self.availabel_label_names, fill_value=0)
#         df = pd.concat([df, multi_hot_matrix], axis=1)
#         return df
        
#     def prepare_dataset(self):
#         # Load main dataset and split files
#         df = pd.read_csv(self.path + "Data_Entry_2017.csv")
#         # Remove leading and trailing whitespace from the 'Paths' column
#         path = './Datasets/mlc_datasets/ChestX-ray8/images/'
#         df['image'] = df['Image Index'].apply(lambda x: os.path.join(path, x))


#         df_val = pd.read_csv(self.path + "validation_labels.csv")
        
#         df_test = pd.read_csv(self.path + "test_list.txt", sep="\t", header=None, names=["Image Index"])
#         df_train_val = pd.read_csv(self.path + "train_val_list.txt", sep="\t", header=None, names=["Image Index"])

#         # Filter train and test DataFrames
#         df_train = df_train_val[~df_train_val['Image Index'].isin(df_val['Image Index'])]
#         df_train = df[df['Image Index'].isin(df_train['Image Index'])]
#         df_test = df[df['Image Index'].isin(df_test['Image Index'])]

#         # Reset indices for consistency
#         df_train = df_train.reset_index(drop=True)
#         df_test = df_test.reset_index(drop=True)
#         df_val = df_val.reset_index(drop=True)

#         df_val = df_val[['Image Index', 'Finding Labels']]
#         df_train = df_train[['Image Index', 'Finding Labels']]
#         df_test = df_test[['Image Index', 'Finding Labels']]

#         # Apply one-hot encoding for the labels
#         df_train = self.to_one_hot(df_train)
#         df_test = self.to_one_hot(df_test)
#         df_val = self.to_one_hot(df_val)

#         df_train = df_train[['Image Index', *self.label_names]]
#         df_test = df_test[['Image Index', *self.label_names]]
#         df_val = df_val[['Image Index', *self.label_names]]

#         # Add split identifiers
#         df_train['split'] = 'train'
#         df_test['split'] = 'test'
#         df_val['split'] = 'val'

#         # Combine all splits into a single DataFrame
#         dfs = pd.concat([df_train, df_test, df_val], axis=0).reset_index(drop=True)
#         dfs['one_hot'] = dfs[self.label_names].values.tolist()

#         path = './Datasets/mlc_datasets/ChestX-ray8/images/'
#         dfs['image'] = dfs['Image Index'].apply(lambda x: os.path.join(path, x))

#         return dfs


class NIHChestXray:
    def __init__(self, path='./Datasets/mlc_datasets/NIH-Chest-X-ray/'):
        self.path = path

        self.availabel_label_names = [
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
            'Pleural_Thickening',
            'Pneumonia',
            'Pneumothorax',
            'Pneumoperitoneum',
            'Pneumomediastinum',
            'Subcutaneous Emphysema',
            'Tortuous Aorta',
            'Calcification of the Aorta',
            'No Finding'
            ]
        
        self.label_names = [
            'Atelectasis',
            'Consolidation',
            'Infiltration',
            'Pneumothorax',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Effusion',
            'Pneumonia',
            'Pleural_Thickening',
            'Cardiomegaly', 
            'Nodule',
            'Mass',
            'Hernia',
            'No Finding'
        ]

    def read_dataframe(self):
        # List to store all .png file paths
        png_files = []
        # Traverse directories recursively
        for dirpath, dirnames, filenames in os.walk(self.path):
            for file in filenames:
                if file.endswith(".png"):
                    # Add the full path to the list
                    png_files.append(os.path.join(dirpath, file))

        png_filenames_to_paths = {
            path.split('.')[-2] + '.' + path.split('.')[-1]: path for path in png_files
        }

        return png_filenames_to_paths
            
    def to_one_hot(self, df):
        multi_hot_matrix = df['Finding Labels'].str.get_dummies(sep='|')
        multi_hot_matrix = multi_hot_matrix.reindex(columns=self.availabel_label_names, fill_value=0)
        df = pd.concat([df, multi_hot_matrix], axis=1)
        return df
        
    def prepare_dataset(self):
        png_filenames_to_paths = self.read_dataframe()

        df = pd.read_csv(self.path + "Data_Entry_2017.csv")

        df_val = pd.read_csv(self.path + "validation_labels.csv")
        
        df_test = pd.read_csv(self.path + "test_list.txt", sep="\t", header=None, names=["Image Index"])
        df_train_val = pd.read_csv(self.path + "train_val_list.txt", sep="\t", header=None, names=["Image Index"])

        # Filter train and test DataFrames
        df_train = df_train_val[~df_train_val['Image Index'].isin(df_val['Image Index'])]
        df_train = df[df['Image Index'].isin(df_train['Image Index'])]
        df_test = df[df['Image Index'].isin(df_test['Image Index'])]

        # Reset indices for consistency
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        df_val = df_val[['Image Index', 'Finding Labels']]
        df_train = df_train[['Image Index', 'Finding Labels']]
        df_test = df_test[['Image Index', 'Finding Labels']]

        # Apply one-hot encoding for the labels
        df_train = self.to_one_hot(df_train)
        df_test = self.to_one_hot(df_test)
        df_val = self.to_one_hot(df_val)

        df_train = df_train[['Image Index', *self.label_names]]
        df_test = df_test[['Image Index', *self.label_names]]
        df_val = df_val[['Image Index', *self.label_names]]

        # Add split identifiers
        df_train['split'] = 'train'
        df_test['split'] = 'test'
        df_val['split'] = 'val'

        # Combine all splits into a single DataFrame
        dfs = pd.concat([df_train, df_test, df_val], axis=0).reset_index(drop=True)
        dfs['one_hot'] = dfs[self.label_names].values.tolist()

        # Map the filenames in the DataFrame to their corresponding file paths
        dfs['image'] = dfs['Image Index'].map(png_filenames_to_paths)

        return dfs