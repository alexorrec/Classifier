import tensorflow as tf
from Classifier import Tester
import os
import json

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
save_path = 'NprintModels'

specs = {
    'ds_path': '/Users/alessandrocerro/Desktop/NOISEPRINT_SET',
    'batch_size': 128,
    'split': 0.2,
    'seed': 100,
    'epochs': 100
}

trainer = Tester(dataset=specs['ds_path'],
                 batch_size=specs['batch_size'],
                 ds_split=specs['split'],
                 seed=specs['seed'],
                 epochs=specs['epochs']
                 )


labels = trainer.train_ds.class_names
num_classes = len(trainer.train_ds.class_names)

effB_model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                                  include_top=False,
                                                  input_shape=trainer.get_shape())
effB_model.trainable = False

for layer in effB_model.layers[-20:]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)
    layer.trainable = True

# Last LAYER Customization
x = tf.keras.layers.Flatten()(effB_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(trainer.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=effB_model.input, outputs=output)

trainer.specify_model(model=model, label=f'NPRINT_{specs["seed"]}')

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

trainer.train_model(loss_function='categorical_crossentropy', lr=0.001)

trainer.evaluate_model(trainer.val_ds)
trainer.plot_results(path=save_path)
trainer.export_model()
trainer.compute_confusion_matrix_categorical(trainer.val_ds)
try:
    json.dump(trainer.history.history, open(os.path.join(save_path, f'{trainer.__name__}_history.txt'), 'w+'))
except Exception as e:
    print(f'HISTORY SAVING FAILED {e}')

with open(os.path.join(save_path, f'report_{trainer.__name__}.txt'), 'w+') as file:
    file.write('LABELS:\n')
    for line in labels:
        file.write(line + '\n')

    file.write('\nSpecs: \n')
    for key, value in specs.items():
        file.write(f'{key} : {value} \n')
