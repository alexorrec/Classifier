import tensorflow as tf
import os
from Classifier import Tester
import cv2

ts = Tester('D:\\BALANCED')

model = tf.keras.models.load_model('sigmoid_128.h5')
ts.specify_model(model, 'LOADED')
print(f'{ts.__name__} has been loaded!')

path = input('testset path: ')
ts.batch_pred(path)

"""
ON MAX REALS: 10% FALSE POSITIVE 28/246
ON MAX AI: 0% FALSE NEGATIVE

ON MIN REALS: 8/246 FALSE POSITIVE
ON MIN AI_XL: 3/246 FALSE NEGATIVE 

path: str = input('Define imagePath: ')


"""
