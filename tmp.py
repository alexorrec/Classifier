import pathlib
import PreProcessing as pp
import cv2

img = cv2.imread('GENERALTESTERS/export.png')
real = cv2.imread('GENERALTESTERS/D10_L4S2C3_0.png')

max_tile, min_tile = pp.get_tiles(img)
ela_min = pp.get_ELA_(min_tile, brigh_it_up=10)
ela_max = pp.get_ELA_(max_tile, brigh_it_up=10)

cv2.imshow('shop_min', ela_min)
cv2.imshow('shop_max', ela_max)

max_tile, min_tile = pp.get_tiles(real)
ela_min = pp.get_ELA_(min_tile, brigh_it_up=10)
ela_max = pp.get_ELA_(max_tile, brigh_it_up=10)

cv2.imshow('real_min', ela_min)
cv2.imshow('real_max', ela_max)

cv2.waitKey(0)
cv2.destroyAllWindows()
