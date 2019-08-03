#-*-coding:utf-8-*-
import os
import cv2
import math
import numpy as np
import time
import copy
import ipdb
import matplotlib.pyplot as plt
from scipy import ndimage

filepath = '/data/wangyf/datasets/svt-plate/daul2'

def main():
    count = 0
    start = time.time()
    LowThreshold = 70
    for filename in os.listdir(filepath):
        if count >= 500:
            break
        count += 1
        img_path = os.path.join(filepath, '%s' % filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        # kernel = np.ones((3, 3), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        erosion = cv2.erode(img, kernel, iterations=1)  # fushi
        dilation = cv2.dilate(img, kernel, iterations=1) # pengzhang
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

        cv2.imshow('src', img)
        cv2.imshow('gray', gray)
        cv2.imshow('show', erosion)
        cv2.imshow('dilation', dilation)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)
        cv2.imshow('gradient', gradient)
        cv2.waitKey()









if __name__ == '__main__':
    main()