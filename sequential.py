import tensorflow as tf


def get_sequential(num_classes, shape, activation):
    model = tf.keras.Sequential([
        tf.keras.layers.Normalization(input_shape=shape),

        tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes, activation=activation)
    ])

    return model

import os

def get_callbacks(path, name):
    lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # La metrica da monitorare (in questo caso la perdita di validazione)
        factor=0.5,  # Fattore di riduzione del learning rate (es. 0.5 dimezza il learning rate)
        patience=5,  # Numero di epoche senza miglioramento dopo le quali ridurre il learning rate
        min_lr=1e-6,  # Limite inferiore per il learning rate (non scenderà sotto questo valore)
        verbose=1  # Imposta verbose=1 per mostrare i messaggi di riduzione del learning rate
    )

    restore_best_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(path, name + '.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    return lr_on_plateau, restore_best_loss
