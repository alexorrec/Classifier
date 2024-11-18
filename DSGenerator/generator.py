import cv2
import os
import numpy as np
import random
import PRNU
import PreProcessing as pp

folders = [
           'C:\\Users\\Alessandro\\Desktop\\SD_Dataset\\V2_INPAINTED']
destination = [
               'C:\\Users\\Alessandro\\Desktop\\PRNU_NPY\\stable-diffusion']

for folder in folders:
    img_paths = []
    pp.ciclic_findings(folder, img_paths)
    random.shuffle(img_paths)
    for img_p in img_paths[:150]:
        print(f'processing prnu {os.path.basename(img_p)}')
        im = cv2.imread(img_p)
        crop = pp.get_tiles(im, just_MinMax=True)[0]
        _prnu = PRNU.extract_single(crop)
        np.save(os.path.join(destination[folders.index(folder)], 'PRNU_' + os.path.basename(img_p)),
                _prnu)
