import numpy as np
import tensorflow as tf
import os
from Classifier import Tester
import cv2
import PreProcessing as pp
import PRNU
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ts = Tester('/Users/alessandrocerro/Desktop/TO_PRED_PRNU')

ts.specify_model(tf.keras.models.load_model('/Volumes/NO NAME/LAST_PRNU_EffNet0_2v.h5'), 'TEST PRNU')
ts.build_set('/Users/alessandrocerro/Desktop/TO_PRED_PRNU')

predictions = ts.model.predict(ts.tmp_data)
predicted_classes = np.argmax(predictions, axis=1)



conf_matrix = confusion_matrix(ts.tmp_data.class_names, predicted_classes)

plt.figure(figsize=(10, 8))
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=ts.tmp_data.class_names)
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

"""
path = input('testset path: ')
ts.batch_pred(path)


ON MAX REALS: 10% FALSE POSITIVE 28/246
ON MAX AI: 0% FALSE NEGATIVE

ON MIN REALS: 8/246 FALSE POSITIVE
ON MIN AI_XL: 3/246 FALSE NEGATIVE 

path: str = input('Define imagePath: ')
"""
