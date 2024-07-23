import tensorflow as tf
from Classifier import Tester

ds_path = 'D:\\BALANCED'
ts = Tester(ds_path, batch_size=256, ds_split=0.3, seed=1, epochs=5)

shape = ts.get_shape()
num_classes: int = len(ts.train_ds.class_names)

""" DEFINE SEQUENTIAL MODEL HERE """
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(4, (5, 5), padding='same', input_shape=shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Conv2D(8, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Conv2D(16, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),


    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
""" END SEQUENTIAL """

ts.specify_model(model=model, label='sigmoid_128')

ts.train_model()
ts.evaluate_model()
ts.plot_results()

res = input('save model - 0 for discard: ')
if res != 0:
    ts.export_model()