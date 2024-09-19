import os
import cv2
import PreProcessing as pp
import matplotlib.pyplot as plt
import random

"""import pathlib

import cv2
import tensorflow as tf
import random
import shutil as sh"""

# CREAZIONE DATASET FINALE: VOGLIO AL PIù 1K V CLASSE
"""
# REALI:
dest = 'C:\\Users\\Alessandro\\Desktop\\ELA_Dataset\\REALS'
path = 'C:\\Users\\Alessandro\\Desktop\\SD_Dataset\\REALS_ds'

files = []
pp.ciclic_findings(path, files)
for file in files:
    im = cv2.imread(file)
    maxT, minT = pp.get_tiles(im)
    to_save = pp.get_ELA_(minT)
    cv2.imwrite(os.path.join(dest, os.path.basename(file)), to_save)
    print(f'{os.path.basename(file)} - SUCCESFULLY SAVED')
"""
# XL:
dest = 'C:\\Users\\Alessandro\\Desktop\\ELA_Dataset\\REALS'
path = 'C:\\Users\\Alessandro\\Desktop\\SD_Dataset\\REALS_ds'

files = []
pp.ciclic_findings(path, files)
random.shuffle(files)
for file in random.choices(files, k=700):
    im = cv2.imread(file)
    try:
        tiles = pp.get_tiles(im, more_tiles=1)
        i = 0
        for tile in tiles:
            to_save = pp.get_ELA_(tile)
            cv2.imwrite(os.path.join(dest, str(i) + '_' + os.path.basename(file)), to_save)
            print(f'{str(i) + os.path.basename(file)} - SUCCESFULLY SAVED')
            i += 1
    except Exception as e:
        pass
