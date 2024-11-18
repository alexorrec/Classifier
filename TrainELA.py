import tensorflow as tf
from Classifier import Tester
import sequential
import os

save_path = 'ELAModels'

specs = {
    'ds_path': 'C:\\Users\\Alessandro\\Desktop\\ELAJPG',
    'batch_size': 512,
    'split': 0.2,
    'seed': 1,
    'epochs': 100
}

trainer = Tester(dataset=specs['ds_path'],
                 batch_size=specs['batch_size'],
                 ds_split=specs['split'],
                 seed=specs['seed'],
                 epochs=specs['epochs'])

labels = trainer.train_ds.class_names
num_classes = len(trainer.train_ds.class_names)

model = sequential.get_sequential(num_classes=num_classes if num_classes > 2 else 1,
                                  shape=trainer.get_shape(),
                                  activation='sigmoid' if num_classes == 2 else 'softmax')

trainer.specify_model(model=model, label=f'ELA_{specs["seed"]}')


trainer.define_callbacks(sequential.get_callbacks(save_path, trainer.__name__))

trainer.train_model(loss_function='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                    lr=0.0001)

trainer.evaluate_model(trainer.val_ds)
trainer.plot_results(path=save_path)

with open(os.path.join(save_path, f'report_{trainer.__name__}.txt'), 'w+') as file:
    file.write('LABELS:\n')
    for line in labels:
        file.write(line +'\n')

    file.write('\n Specs: \n')
    for key, value in specs.items():
        file.write(f'{key} : {value} \n')
