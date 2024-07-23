import random
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib


class Tester:
    def __init__(self, dataset: str, batch_size: int = 32, ds_split: float = 0.2, epochs: int = 10, seed: int = 123):
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

        self.img_height, self.img_widht, self.img_channels = self.get_shape()
        self.build_sets()
        self.history = None

    def train_model(self, loss_function: str = 'binary_crossentropy'):
        assert self.model
        # self.build_sets()
        self.model.compile(optimizer='adam',
                           loss=loss_function,
                           metrics=['accuracy']
                           )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

    def evaluate_model(self, dataset):
        test_loss, test_acc = self.model.evaluate(dataset, verbose=2)
        print(f'{self.__name__} - Loss {test_loss} - Accuracy {test_acc} on PassedSet')

    def build_sets(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            validation_split=self.ds_split,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_widht),
            batch_size=self.batch_size
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            validation_split=self.ds_split,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_widht),
            batch_size=self.batch_size
        )

    def build_set(self, path):
        self.tmp_data = tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            seed=self.seed,
            image_size=(self.img_height, self.img_widht),
        )
        print(f'TEMP SET: {self.tmp_data.class_names}')

    def get_shape(self):
        assert self.dataset_path != '', 'Specify Dataset Path.'
        images = list(self.dataset_path.glob('REALS/*'))
        rnd_img = cv2.imread(str(random.sample(images, 1)[0].resolve()))
        return rnd_img.shape

    def plot_results(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
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
        plt.savefig(f'{self.__name__}_plot.png')

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

    def define_callbacks(self, *args):
        for _ in args:
            self.callbacks.append(_)

    def specify_model(self, model, label):
        self.model = model
        self.__name__ = label
        print(f'{self.__name__} has {self.train_ds.class_names}')

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
        print(f'wrong catch: {total} / {len(os.listdir(li))}')

    def export_model(self):
        self.model.save(self.__name__ + '.h5')
