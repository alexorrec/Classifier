import random
import tensorflow as tf
import os
import cv2
import numpy as np
import pathlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


SAVE_PATH = 'NprintModels'

class Tester:
    def __init__(self, dataset: str,
                 batch_size: int = 32,
                 ds_split: float = 0.2,
                 epochs: int = 10,
                 seed: int = 123,
                 is_npy=False):

        self.tmp_data = None
        self.seed: int = seed
        self.callbacks: list = []
        self.__name__: str = 'None'
        self.epochs: int = epochs
        self.val_ds = None
        self.train_ds = None
        self.model = None
        self.dataset_path = pathlib.Path(dataset)
        self.batch_size = batch_size
        self.ds_split: float = ds_split


        if '.DS_Store' in os.listdir(dataset):
            os.remove(os.path.join(dataset, '.DS_Store'))
        self.num_classes = len(os.listdir(dataset))

        self.img_height, self.img_widht, self.img_channels = self.get_shape()
        self.build_sets()
        self.history = None

    def specify_model(self, model, label):
        self.model = model
        self.__name__ = label
        print(f'{self.__name__} has {self.train_ds.class_names}')

    def train_model(self, loss_function: str = 'binary_crossentropy', lr=0.0001):
        assert self.model

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           loss=loss_function,
                           metrics=['accuracy', 'AUC', 'Precision', 'Recall']
                           )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=self.callbacks
        )


    def evaluate_model(self, dataset):
        results = self.model.evaluate(dataset, verbose=2)

        test_loss = results[0]  # Loss
        test_acc = results[1]   # Accuracy
        test_auc = results[2]   # AUC
        test_precision = results[3]  # Precision
        test_recall = results[4]  # Recall

        # Print the results
        print(f'{self.__name__} - Loss: {test_loss:.4f} - Accuracy: {test_acc:.4f}')
        print(f'AUC: {test_auc:.4f} - Precision: {test_precision:.6f} - Recall: {test_recall:.6f}')


    def build_sets(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            labels='inferred',
            label_mode='int' if self.num_classes == 2 else 'categorical',  # Categorical for multiclass, Int for binary
            color_mode='rgb',
            validation_split=self.ds_split,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_widht),
            shuffle=True,
            batch_size=self.batch_size
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            labels='inferred',
            label_mode='int' if self.num_classes == 2 else 'categorical',  # Categorical for multiclass, Int for binary
            color_mode='rgb',
            validation_split=self.ds_split,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_widht),
            shuffle=True,
            batch_size=self.batch_size
        )

    def build_set(self, path):
        """Pass to it whatever set"""
        self.tmp_data = tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode= 'int' if self.num_classes == 2 else 'categorical',
            color_mode='rgb',
            seed=self.seed,
            image_size=(self.img_height, self.img_widht),
        )
        print(f'Test set: {self.tmp_data.class_names}')

    def get_shape(self):
        assert self.dataset_path != '', 'Specify Dataset Path.'

        images = list(self.dataset_path.glob('*/*'))
        rnd_img = cv2.imread(str(random.sample(images, 1)[0].resolve()))
        print(f'Retrieved shape: {rnd_img.shape}')
        return rnd_img.shape

    def define_callbacks(self, *args):
        for _ in args:
            self.callbacks.append(_)

    def plot_results(self, path=''):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(path, f'{self.__name__}_plot.png'))


    def export_model(self):
        self.model.save(self.__name__ + 'Last_EPOCH.h5')


    def compute_confusion_matrix(self, dataset):
        true_labels = []
        predicted_labels = []

        for images, labels in dataset:
            true_labels.extend(labels.numpy())
            probabilities = self.model.predict(images)
            predictions = (probabilities > 0.5).astype(int).flatten()
            predicted_labels.extend(predictions)

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        cm = confusion_matrix(true_labels, predicted_labels)
        c_names = self.train_ds.class_names

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=c_names, yticklabels=c_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Validation Set')
        plt.savefig(self.__name__ + '_confusion_matrix.png')
        return cm

    def compute_confusion_matrix_categorical(self, dataset):
        true_labels = []
        predicted_labels = []

        for images, labels in dataset:
            true_labels.extend(np.argmax(labels.numpy(), axis=1))
            probabilities = self.model.predict(images)
            predictions = np.argmax(probabilities, axis=1)
            predicted_labels.extend(predictions)

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        cm = confusion_matrix(true_labels, predicted_labels)
        c_names = self.train_ds.class_names

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=c_names, yticklabels=c_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix NoisePrint - Validation Set')
        plt.savefig('NoisePrintConfusionMatrix.png')
        return cm




