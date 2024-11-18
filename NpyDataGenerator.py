import os
import numpy as np
# from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import numpy as np
# from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split


class NpyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, target_size=(512, 512), num_classes=2,
                 shuffle=True, validation_split=0.0, subset=None):
        """
        Initialize the generator.
        Args:
            directory (Path or str): Path to the root directory containing class subdirectories.
            batch_size (int): Number of samples per batch.
            target_size (tuple): Expected shape of each sample (height, width).
            num_classes (int): Number of classes for one-hot encoding.
            shuffle (bool): Whether to shuffle the data at the end of each epoch.
            validation_split (float): Fraction of data to use for validation (0 to 1).
            subset (str): Either 'training' or 'validation', specifying which subset to load.
        """
        self.directory = Path(directory)  # Convert to Path object if not already
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.subset = subset
        self.filepaths, self.labels = self._load_filepaths_and_labels()
        self.class_names = None
        if validation_split > 0.0:
            self._split_data()
        self.on_epoch_end()

    def _load_filepaths_and_labels(self):
        """
        Load filepaths and assign labels based on subdirectory names.
        """
        filepaths = []
        labels = []
        self.class_names = sorted([d.name for d in self.directory.iterdir() if d.is_dir()])  # Consistent class ordering
        self.class_indices = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.directory / class_name  # Use Path to join paths
            for npy_file in class_dir.glob("*.npy"):  # Use glob to find .npy files
                filepaths.append(npy_file)  # Store as Path objects
                labels.append(self.class_indices[class_name])

        return np.array(filepaths), np.array(labels)

    def _split_data(self):
        """
        Split data into training and validation sets based on validation_split.
        """
        if not hasattr(self, '_split_done'):
            # Perform the split only once
            self._split_done = True
            split_result = train_test_split(
                self.filepaths, self.labels,
                test_size=self.validation_split,
                stratify=self.labels,
                random_state=42  # Ensure reproducibility
            )
            self.train_filepaths, self.val_filepaths, self.train_labels, self.val_labels = split_result

        # Assign subset-specific data
        if self.subset == 'training':
            self.filepaths, self.labels = self.train_filepaths, self.train_labels
        elif self.subset == 'validation':
            self.filepaths, self.labels = self.val_filepaths, self.val_labels

    def __len__(self):
        """
        Number of batches per epoch.
        """
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        Args:
            index (int): Batch index.
        Returns:
            batch_data (numpy array): Batch of input data.
            batch_labels (numpy array): Batch of one-hot encoded labels.
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_filepaths = self.filepaths[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Load and preprocess data
        batch_data = np.array([np.load(str(f)).astype(np.float32) for f in batch_filepaths])
        batch_data = np.expand_dims(batch_data, axis=-1)  # Add channel dimension (H, W) -> (H, W, 1)
        #batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        batch_data = np.repeat(batch_data, 3, axis=-1)

        return batch_data, np.asarray(batch_labels)

    def on_epoch_end(self):
        """
        Shuffle data at the end of each epoch.
        """
        self.indices = np.arange(len(self.filepaths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_class_names(self):
        """Return the class names."""
        return self.class_names

"""
# Path to the dataset
train_dir = 'C:\\Users\\Alessandro\\Desktop\\PRNU_NPY'
# Check overlap between training and validation filepaths
train_generator = NpyDataGenerator(
    directory=train_dir,
    batch_size=32,
    validation_split=0.2,
    subset='training'
)

val_generator = NpyDataGenerator(
    directory=train_dir,
    batch_size=32,
    validation_split=0.2,
    subset='validation'
)
train_filepaths = set(train_generator.filepaths)
val_filepaths = set(val_generator.filepaths)

# Validate there is no overlap
overlap = train_filepaths.intersection(val_filepaths)
assert len(overlap) == 0, f"Overlap found: {overlap}"
print("No overlap between training and validation data!")
"""