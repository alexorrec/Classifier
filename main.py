import tensorflow as tf
from Classifier import Tester
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

ds_path = 'D:\\PRNUDS'
ts = Tester(ds_path, batch_size=640, ds_split=0.2, seed=123, epochs=100)

shape = ts.get_shape()
num_classes: int = len(ts.train_ds.class_names)

effB0_model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                                   include_top=False,
                                                   input_shape=shape)

for layer in effB0_model.layers:
    layer.trainable = False

# Lase LAYER Customization
x = tf.keras.layers.Flatten()(effB0_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=effB0_model.input, outputs=output)

ts.specify_model(model=model, label='PRNU_EfficientNetB2_3v')

ts.train_model(loss_function='sparse_categorical_entropy')
ts.evaluate_model(ts.val_ds)
ts.plot_results()

ts.export_model()
