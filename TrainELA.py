from Classifier import Tester
import tensorflow as tf
import os
import json

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
save_path = 'ELAModels'

specs = {
    'ds_path': '/Users/alessandrocerro/Desktop/ELA_SET',
    'batch_size': 128,
    'split': 0.2,
    'seed': 10,
    'epochs': 100
}

trainer = Tester(dataset=specs['ds_path'],
                 batch_size=specs['batch_size'],
                 ds_split=specs['split'],
                 seed=specs['seed'],
                 epochs=specs['epochs'])

labels = trainer.train_ds.class_names
num_classes = len(trainer.train_ds.class_names)

model = tf.keras.Sequential([

    tf.keras.layers.Normalization(input_shape=trainer.get_shape()),

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

    tf.keras.layers.Dense(1, activation='sigmoid')
])

lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.25,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

restore_best_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, trainer.__name__ + '.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )


trainer.define_callbacks(lr_on_plateau, restore_best_loss)

trainer.train_model(loss_function='binary_crossentropy', lr=0.0001)

trainer.evaluate_model(trainer.val_ds)
trainer.plot_results(path=save_path)

try:
    json.dump(trainer.history.history, open(os.path.join(save_path, f'{trainer.__name__}_history.txt'), 'w+'))
except Exception as e:
    print(f'HISTORY SAVING FAILED {e}')

with open(os.path.join(save_path, f'report_{trainer.__name__}.txt'), 'w+') as file:
    file.write('LABELS:\n')
    for line in labels:
        file.write(line +'\n')

    file.write('\nSpecs: \n')
    for key, value in specs.items():
        file.write(f'{key} : {value} \n')
