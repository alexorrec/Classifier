import random
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import json
import NpyDataGenerator

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
        # print(self.num_classes)

        self.img_height, self.img_widht, self.img_channels = self.get_shape()
        if is_npy:
            self.build_npyGen()
        else: self.build_sets()

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

    """
    def evaluate_model(self, dataset):
        test_loss, test_acc = self.model.evaluate(dataset, verbose=2)
        print(f'{self.__name__} - Loss {test_loss} - Accuracy {test_acc} on ValSet')
    """
    def evaluate_model(self, dataset):
        results = self.model.evaluate(dataset, verbose=2)

        # Unpack and assign values
        test_loss = results[0]  # Loss
        test_acc = results[1]   # Accuracy
        test_auc = results[2]   # AUC
        test_precision = results[3]  # Precision
        test_recall = results[4]  # Recall

        # Print the results
        print(f'{self.__name__} - Loss: {test_loss:.4f} - Accuracy: {test_acc:.4f}')
        print(f'AUC: {test_auc:.4f} - Precision: {test_precision:.4f} - Recall: {test_recall:.4f}')


    def build_npyGen(self):
        self.train_ds = NpyDataGenerator.NpyDataGenerator(
            directory=self.dataset_path,
            batch_size=self.batch_size,
            validation_split=self.ds_split,
            subset='training'
        )

        self.val_ds = NpyDataGenerator.NpyDataGenerator(
            directory=self.dataset_path,
            batch_size=self.batch_size,
            validation_split=self.ds_split,
            subset='validation'
        )

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

        # print(f'FlowFromDS loaded: {self.train_ds.label_mode}')


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

    def make_prediction(self, image_path):
        """ PASS CV2 IMAGE, GET TILE, GET ELA, PREDICT"""
        img = cv2.imread(image_path)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.train_ds.class_names[np.argmax(score)], 100 * np.max(score))
        )

    def predictor(self, img_array, label: str = ''):
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "Prediction {} with a {:.2f} percent confidence. {}"
            .format(self.train_ds.class_names[np.argmax(score)], 100 * np.max(score), label)
        )

    def predict_(self, img: np.array):
        to_arr = tf.expand_dims(img, 0)
        predictions = self.model.predict(to_arr)
        score = tf.nn.softmax(predictions[0])
        return self.train_ds.class_names[np.argmax(score)], 100 * np.max(score)

    def batch_pred(self, li):
        total = len(os.listdir(li))
        for path in os.listdir(li):
            im = cv2.imread(os.path.join(li, path))
            t = self.predict_(im)
            print(f'{os.path.basename(path)} PREDICTED: {t[0], t[1]}, GROUND: {os.path.basename(li)}')
            if t[0] == os.path.basename(li):
                total -= 1
        print(f'Wrong catch: {total} / {len(os.listdir(li))}')

    def export_model(self):
        try:
            self.model._setattr_metadata("model_labels", json.dumps({"CLASS_NAMES": self.train_ds.class_names}))
        except:
            print('LABELSAVE FAILED?')
        self.model.save(self.__name__ + '.h5')

    def confusion_matrix(self, tensors):
        predictions = self.model.predict(
            x=tensors,
            batch_size=10,
            verbose=0
        )
        rounded_predictions = np.argmax(predictions, axis=1)

        return confusion_matrix(y_true=[0, 1], y_pred=rounded_predictions)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.imsave('confusion_matrix.png')
