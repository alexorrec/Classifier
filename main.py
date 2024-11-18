import tensorflow as tf
from Classifier import Tester
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

ds_path = '/Users/alessandrocerro/Desktop/ELA_SET'
ts = Tester(ds_path, batch_size=128, ds_split=0.2, seed=123, epochs=100)

shape = ts.get_shape()
num_classes: int = len(ts.train_ds.class_names)
"""
effB_model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                                  include_top=False,
                                                  input_shape=shape)
effB_model.trainable = False

for layer in effB_model.layers[-20:]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    layer.trainable = True

# Last LAYER Customization
x = tf.keras.layers.Flatten()(effB_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.models.Model(inputs=effB_model.input, outputs=output)

"""

model = tf.keras.Sequential([

    tf.keras.layers.Normalization(input_shape=shape),

    tf.keras.layers.Conv2D(8, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(16, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(32, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(64, (5, 5), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')
    # tf.keras.layers.Dense(num_classes, activation='softmax') # 3 CLASS
])


ts.specify_model(model=model, label='Nprint_take1_EffNetB0_3v')

lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # La metrica da monitorare (in questo caso la perdita di validazione)
    factor=0.5,  # Fattore di riduzione del learning rate (es. 0.5 dimezza il learning rate)
    patience=5,  # Numero di epoche senza miglioramento dopo le quali ridurre il learning rate
    min_lr=1e-6,  # Limite inferiore per il learning rate (non scenderà sotto questo valore)
    verbose=1  # Imposta verbose=1 per mostrare i messaggi di riduzione del learning rate
)

restore_best_loss = tf.keras.callbacks.ModelCheckpoint(
    filepath=ts.__name__ + '.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

ts.define_callbacks(lr_on_plateau, restore_best_loss)

ts.train_model(loss_function='binary_crossentropy', lr=0.0001)

ts.evaluate_model(ts.val_ds)
ts.plot_results()

ts.export_model()
