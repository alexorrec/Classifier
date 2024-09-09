import tensorflow as tf
import os
from Classifier import Tester
import cv2
import PreProcessing as pp
import PRNU

ts = Tester('D:\\BALANCED')

model = tf.keras.models.load_model('PRNU_softmax_512N.h5')
ts.specify_model(model, '512-MODEL')
print(f'{ts.__name__} has been loaded!')

model.summary()
ts.build_set('D:\\TO_PRED_MAX')
ts.evaluate_model(ts.tmp_data)


"""
path = input('testset path: ')
ts.batch_pred(path)


ON MAX REALS: 10% FALSE POSITIVE 28/246
ON MAX AI: 0% FALSE NEGATIVE

ON MIN REALS: 8/246 FALSE POSITIVE
ON MIN AI_XL: 3/246 FALSE NEGATIVE 

path: str = input('Define imagePath: ')
"""
