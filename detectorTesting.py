import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pathlib

ds_path = 'C:\\Users\\Alessandro\\Desktop\\ELA_PATCH'

epochs: int = 5
seed: int = 123

batch_size = 160
img_height = 512
img_width = 512

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ds_path,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='training',
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ds_path,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=seed,
    validation_split=0.2,
    subset='validation',
)

class_names = train_ds.class_names
print(f'ClassNames: {class_names}')

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

# FIRST TESTING MODEL - ON small-DS REACHED 97% ON 9TH EPOCH
model = tf.keras.Sequential([

    #tf.keras.layers.Rescaling(1. / 255, input_shape=(512, 512, 3)),
    data_augmentation,

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=2,
                                                  verbose=0,
                                                  mode='auto')

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    # callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(f'\nTest accuracy: {test_acc} - Loss: {test_loss}')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

##### prediction

tampered_path = 'C:\\Users\\Alessandro\\Desktop\\XL.png'
real_path = 'C:\\Users\\Alessandro\\Desktop\\notXL.png'

tampered = tf.keras.utils.load_img(
    tampered_path, target_size=(img_height, img_width)
)

tampered_array = tf.keras.utils.img_to_array(tampered)
tampered_array = tf.expand_dims(tampered_array, 0)  # Create a batch

predictions = model.predict(tampered_array)
score = tf.nn.softmax(predictions[0])

print(
    "XL_image - Prediction: {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

real = tf.keras.utils.load_img(
    real_path, target_size=(img_height, img_width)
)

real_array = tf.keras.utils.img_to_array(real)
real_array = tf.expand_dims(real_array, 0)  # Create a batch

predictions = model.predict(real_array)
score = tf.nn.softmax(predictions[0])

print(
    "REAL_image - Prediction: {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
