import tensorflow as tf
from Classifier import Tester
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

ds_path = 'D:\\PRNUDS'
ts = Tester(ds_path, batch_size=640, ds_split=0.2, seed=123, epochs=100)

shape = ts.get_shape()
num_classes: int = len(ts.train_ds.class_names)

""" DEFINE SEQUENTIAL MODEL HERE 
model = tf.keras.Sequential([

    tf.keras.layers.Normalization(input_shape=shape),

    tf.keras.layers.Conv2D(8, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Conv2D(16, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Conv2D(32, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPool2D((2, 2), strides=(4, 4)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])
END SEQUENTIAL """


effB2_model = tf.keras.applications.EfficientNetB2(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=shape)

for layer in effB2_model.layers:
    layer.trainable = False

x = tf.keras.layers.Flatten()(effB2_model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=effB2_model.input, outputs=output)

ts.specify_model(model=model, label='PRNU_EfficientNetB2_3v')

ts.train_model()
ts.evaluate_model(ts.val_ds)
ts.plot_results()

ts.export_model()
