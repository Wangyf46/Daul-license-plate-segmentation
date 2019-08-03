import cv2
import os
import numpy as np
import ipdb
import time

filepath = '/data/wangyf/datasets/svt-plate/daul1-haugh'
count = 0
start = time.time()
for filename in os.listdir(filepath):
    count += 1
    img_path = os.path.join(filepath, '%s' % filename)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area = gray.shape[0] * gray.shape[1]
    thresh = np.sum(gray) * 1.0/ area ## TODO
    ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)   ## TODO
    (h, w) = thresh1.shape
    a = [0 for z in range(0, h)]
    for j in range(0, h):
        for i in range(0, w):
            if thresh1[j, i] == 0:
                a[j] += 1
                thresh1[j, i] = 255
    for j in range(0, h):
        for i in range(0, a[j]):
            thresh1[j, i] = 0

    min_line = a[0]
    line = None
    top = int(len(a) * 0.3)# 0.15/0.2/0.25
    bottom = int(len(a) - top)
    for m in range(top, bottom):
        if a[m] < min_line:
            min_line = a[m]
            line = m
    print(top, bottom, ' ', line, count, thresh)
    crop_img1 = img[0:line+1, 0:w]
    crop_img2 = img[line:h, 0:w]
    crop_img11 = cv2.resize(crop_img1, (crop_img2.shape[1], crop_img2.shape[0]))
    output = np.hstack((crop_img11, crop_img2))

    cv2.imshow('img', img)
    cv2.imshow('thresh1', thresh1)
    cv2.imshow('crop_img1', crop_img1)
    cv2.imshow('crop_img11', crop_img11)
    cv2.imshow('crop_img11', crop_img2)
    cv2.imshow('output', output)
    cv2.waitKey(0)
    if count >= 200:
        break
    cv2.imwrite('/data/wangyf/datasets/svt-plate/daul1-seg/%s' % filename, output)

end = time.time()
print((end - start)/count)

