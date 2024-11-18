from Classifier import Tester
import sequential
import os
import json

save_path = 'ELAModels'

specs = {
    'ds_path': '/Users/alessandrocerro/Desktop/ELA_SET',
    'batch_size': 32,
    'split': 0.2,
    'seed': 1000,
    'epochs': 3
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


lr, restore = sequential.get_callbacks(save_path, trainer.__name__)
trainer.define_callbacks(lr, restore)

trainer.train_model(loss_function='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                    lr=0.0001)

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
