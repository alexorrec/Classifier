import tensorflow as tf
import os
from Classifier import Tester
import cv2
import PreProcessing as pp

ts = Tester('D:\\BALANCED')

model = tf.keras.models.load_model('softmax_512.h5')
ts.specify_model(model, 'SIGMOID512')
print(f'{ts.__name__} has been loaded!')

model.summary()
ts.build_set('D:\\TO_PRED_MIN')

ts.evaluate_model(ts.tmp_data)
raise SystemExit

img = cv2.imread(input('imagepath: '))

max_tile, min_tile = pp.get_tiles(img)
ela_min = pp.get_ELA_(min_tile)
ela_max = pp.get_ELA_(max_tile)
ts.predictor(ela_min, 'min_tile')
ts.predictor(ela_max, 'max_tile')

"""
path = input('testset path: ')
ts.batch_pred(path)


ON MAX REALS: 10% FALSE POSITIVE 28/246
ON MAX AI: 0% FALSE NEGATIVE

ON MIN REALS: 8/246 FALSE POSITIVE
ON MIN AI_XL: 3/246 FALSE NEGATIVE 

path: str = input('Define imagePath: ')
"""
