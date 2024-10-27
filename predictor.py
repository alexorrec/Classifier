import numpy as np
import tensorflow as tf
import os
from Classifier import Tester
import cv2
import PreProcessing as pp
import PRNU

model = tf.keras.models.load_model('NPRINT_EffNet0_3vTMP.h5')
pred = model.predict(cv2.imread('C:\\Users\\Alessandro\\Desktop\\D01_I0270.png_NPRINT.png'))


print(np.argmax(pred, axis=1))

"""
path = input('testset path: ')
ts.batch_pred(path)


ON MAX REALS: 10% FALSE POSITIVE 28/246
ON MAX AI: 0% FALSE NEGATIVE

ON MIN REALS: 8/246 FALSE POSITIVE
ON MIN AI_XL: 3/246 FALSE NEGATIVE 

path: str = input('Define imagePath: ')
"""
