import tensorflow as tf
from Classifier import Tester
import sequential
import os

save_path = 'PRNUModels'

specs = {
    'ds_path': 'C:\\Users\\Alessandro\\Desktop\\PRNU_BALANCED',
    'batch_size': 64,
    'split': 0.2,
    'seed': 3,
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
#num_classes=2
"""
model = sequential.get_sequential(num_classes=num_classes if num_classes > 2 else 1,
                                  shape=trainer.get_shape(),
                                  activation='sigmoid' if num_classes == 2 else 'softmax')
"""


effB_model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                                  include_top=False,
                                                  input_shape=trainer.get_shape())
effB_model.trainable = False
"""
for layer in effB_model.layers[-20:]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    layer.trainable = True
"""
# Last LAYER Customization
x = tf.keras.layers.Flatten()(effB_model.output)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.4)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.models.Model(inputs=effB_model.input, outputs=output)


trainer.specify_model(model=model, label=f'PRNU_{specs["seed"]}')

trainer.define_callbacks(sequential.get_callbacks(save_path, trainer.__name__))

trainer.train_model(loss_function='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                    lr=0.0001)

trainer.evaluate_model(trainer.val_ds)
trainer.plot_results(path=save_path)

# labels = ['naturals', 'stable-diffusion']
with open(f'report_{trainer.__name__}.txt', 'w+') as file:
    file.write('LABELS:\n')
    for line in labels:
        file.write(line + '\n')

    file.write('Specs: \n')
    for key, value in specs.items():
        file.write(f'{key} : {value} \n')
