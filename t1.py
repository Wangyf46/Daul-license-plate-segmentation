import cv2
import numpy as np
import os
import ipdb
from matplotlib import pyplot as plt
import copy

filepath = '/data/wangyf/datasets/svt-plate/daul1'
for filename in os.listdir(filepath):
    img_path = os.path.join(filepath, '%s' % filename)
    image = cv2.imread(img_path)
    cv2.imshow('img', image)
    GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(GrayImage, 100, 255, cv2.THRESH_BINARY)
    (h, w) = thresh.shape
    thresh1 = copy.deepcopy(thresh)
    thresh2 = copy.deepcopy(thresh)

    ## shuiping
    shuiping = [0 for z in range(0, h)]
    for j1 in range(0, h):
        for i1 in range(0, w):
            if thresh1[j1, i1] == 0:
                shuiping[j1] += 1
                thresh1[j1, i1] = 255
    for j2 in range(0, h):
        for i2 in range(0, shuiping[j2]):
            thresh1[j2, i2] = 0
    print(shuiping)
    ## chuizhi
    chuizhi = [0 for z in range(0, w)]
    for j3 in range(0, w):
        for i3 in range(0, h):
            if thresh2[i3, j3] == 0:
                chuizhi[j3] += 1
                thresh2[i3, j3] = 255
    for j4 in range(0, w):
        for i4  in range((h - chuizhi[j4]), h):
            thresh2[i4, j4] = 0
    print(chuizhi)
    cv2.imshow('shuiping', thresh1)
    cv2.imshow('chuizhi', thresh2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

