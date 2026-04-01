import io
import os
import numpy as np
from PIL import Image
from datasets import load_dataset

class HFImageReader:
    def __init__(self):
        pass

    def read_image(self, row):
        """
        Reads an image from in-memory bytes, converts it to RGB if necessary, 
        and returns it as a NumPy array.

        Args:
            row (dict): Dictionary containing the 'image' key with the image data in bytes.

        Returns:
            np.ndarray: The image as an RGB NumPy array.
        """
        image_data = row.get('image')
        if not isinstance(image_data, dict) or 'bytes' not in image_data:
            raise ValueError("The 'image' field should be a dictionary with 'bytes' key containing the image data.")
        
        # Read image from bytes
        image = Image.open(io.BytesIO(image_data['bytes']))
        
        # Convert to RGB if not already in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert the PIL image to a NumPy array and return
        return np.array(image)

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels].astype(int)

def prepare_dataset(dataset_name, num_classes):
    """
    Placeholder function to load a dataset.
    """
    df = load_dataset(os.path.join("jonathan-roberts1", dataset_name), split='train').to_pandas()
    df = df[['image', 'label']]
    df['one_hot'] = df['label'].apply(lambda x: one_hot_encode(x, num_classes))
    df['one_hot'] = df['one_hot'].apply(lambda x: np.sum(x, axis=0))
    ys = np.array(df['one_hot'].tolist())
    df = df.loc[np.where(ys.sum(1) != 0)[0]].reset_index(drop=True)
    return df

# General Dataset Class (for aid, ucm, mlrsnet)
class HuggingFace:
    def __init__(self, dataset_name, num_classes):
        self.dataset_name = dataset_name
        self.num_classes = num_classes

    def fill_targets(self, row, label_names):
        # Iterate over the one_hot array (binary array)
        for idx, value in enumerate(row['one_hot']):
            if value == 1:  # If the value is 1, set the corresponding label column to 1
                row[label_names[idx]] = 1
        return row

    def create_label_columns(self, df):
        # Assuming the number of labels equals the length of the one_hot vectors
        num_labels = len(df['one_hot'][0])  # Get the length of one_hot vector
        label_columns = [f'Label_{i}' for i in range(num_labels)]

        # Add label columns to the DataFrame and initialize them to 0
        for label in label_columns:
            df[label] = 0

        # Apply the fill_targets function to each row to fill the one-hot encoded columns
        df = df.apply(lambda row: self.fill_targets(row, label_columns), axis=1)
        return df

    def rename(self, df, label_names):
        return df.rename(columns = {f'Label_{i}': name for i, name in enumerate(label_names)})

    def prepare_dataset(self):
        df = prepare_dataset(self.dataset_name, self.num_classes)
        df['split'] = 'full'  # No predefined test set, mark all as train
        return df
