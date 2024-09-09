import os
import pathlib
import PreProcessing as pp
import cv2
import tensorflow as tf
import random
import shutil as sh

"""
ORIGIN = 'D:\\TO_PRED_MIN\\REALS'
destination = 'C:\\Users\\Alessandro\\Desktop\\TO_PRED_JPG\\REALS'

for path in os.listdir(ORIGIN):
    im = cv2.imread(os.path.join(ORIGIN, path))
    pp.export_jpg(im, os.path.basename(path), path=destination)


DESTINATION = 'C:\\Users\\Alessandro\\Desktop\\TO_PRED_JPG\\AI_XL\\'
REALS = 'C:\\Users\\Alessandro\\Desktop\\JPG_T'

for img in os.listdir(REALS):
    im = cv2.imread(os.path.join(REALS, img))
    max_tile, min_tile = pp.get_tiles(im)
    cv2.imwrite(DESTINATION + 'min_' + img, pp.get_ELA_(min_tile))
    cv2.imwrite(DESTINATION + 'max_' + img, pp.get_ELA_(max_tile))
"""

destination = 'C:\\Users\\Alessandro\\Desktop\\PRNU_TEST'
img_list = []
pp.ciclic_findings('D:\XL_INPAINTED', img_list)

for img_p in img_list:
    sh.copy2(img_p, destination)