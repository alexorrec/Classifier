import numpy as np
import tensorflow as tf
import os
from Classifier import Tester
import cv2
import PreProcessing as pp
import PRNU
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#ts = Tester('C:\\Users\\Alessandro\\Desktop\\PRNU_BALANCED')

#ts.specify_model(tf.keras.models.load_model('PRNUModels/PRNU_3.keras'), 'TEST PRNU')

model = tf.keras.models.load_model('PRNUModels/PRNU_3.keras')





import matplotlib.pyplot as plt

def plot_training_history(history):
    # Extract data from the history object
    acc = history.history['accuracy']  # Training accuracy
    val_acc = history.history['val_accuracy']  # Validation accuracy
    loss = history.history['loss']  # Training loss
    val_loss = history.history['val_loss']  # Validation loss

    # Set up the figure
    plt.figure(figsize=(12, 5))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Example usage
# Assume `history` is the result of `model.fit()`
plot_training_history(model.history)
