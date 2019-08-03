import os
import cv2
import math
import numpy as np
import time
import copy
import ipdb
import matplotlib.pyplot as plt
from scipy import ndimage
from FuzzyReduction import *

filepath = '/nfs-data/wangyf/131/data/datasets/svt-plate/daul2'
#filepath = '/data2/wangyf/131/data/datasets/svt-plate/daul2'
#filepath='/home/wangyf/w-data/daul2'
dir_haugh = '/data/wangyf/datasets/svt-plate/daul2-haugh'
dir_LSD = '/data/wangyf/datasets/svt-plate/daul2-LSD'
dir_seg = '/data/wangyf/datasets/svt-plate/daul2-seg'


def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)


def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def defocus_kernel(d, sz=10): # 65
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


def ThresholdProcess(gray):
    ##TODO:adaptive threshold
    # Hist = gray.ravel()
    # plt.hist(gray.ravel(), 256)
    # plt.show()
    #blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # ret, binary1 = cv2.threshold(gray, 0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary


def ContoursDetect(gray, binary):
    Height, Width = binary.shape
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #for rho, theta in var:
    maxArea = 0
    bbox = None
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        for i in range(len(box)):
            if box[i][0] < 0:
                box[i][0] = 0
            if box[i][1] < 0:
                box[i][1] = 0
        # Xs = [i[0] for i in box]
        # Ys = [i[1] for i in box]
        minArea = rect[1][0] * rect[1][1]
        if minArea > maxArea:
            maxArea = minArea
            bbox = box
    cv2.drawContours(gray, [bbox], 0, (0, 0, 255), 1)  # -1
    '''
        diff_x = np.max(Xs) - np.min(Xs)
        diff_y = np.max(Ys) - np.min(Ys)
        if diff_x > (Width / 2) and diff_y > (Height / 2):
            cv2.drawContours(img, [box], 0, (0, 0, 255), 1)  # -1
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1)
    '''
    return gray


def EdgeDetect(img_contour, LowThreshold):
    ####TODO: adaptive threshold
    gray = cv2.cvtColor(img_contour.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50 , 180, apertureSize=3) #70 - 180
    return  edges


def LineDetect_Haugh(img_contour, edges):
    ####TODO: adaptive threshold 0/1/2
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 1) ##
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_contour, (x1, y1), (x2, y2), (255, 0, 0), 1)
    print(x1, x2, y1, y2)  ## TODO
    return x1, x2, y1, y2, img_contour



def LineDetect_LSD(img_contour, edges):
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]
    # drawn_img = lsd.drawSegments(image, lines)
    max_lsd = 0
    x1_lsd, x2_lsd, y1_lsd, y2_lsd = 0, 0, 0, 0
    for l in lines:
        x1, y1, x2, y2 = l.flatten()
        diff_x = abs(x2 - x1)
        diff_y = abs(y2 - y1)
        if diff_x > max_lsd and diff_y < 3.0:
            max_lsd = diff_x
            x1_lsd = x1
            x2_lsd = x2
            y1_lsd = y1
            y2_lsd = y2
    cv2.line(img_contour, (x1_lsd, y1_lsd), (x2_lsd, y2_lsd), (0, 255, 0), 1)
    return x1_lsd, x2_lsd, y1_lsd, y2_lsd, img_contour


def Rotate(img_src,  x1, x2, y1, y2, filename, count):
    if x1 == x2 or y1 == y2:
        print('==',  count)
        #cv2.imwrite('/data/wangyf/datasets/svt-plate/daul1-haugh/%s' % filename, img_src)
        return  img_src
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    print(rotate_angle, count)
    rotate_img = ndimage.rotate(img_src, rotate_angle) ########################TODO
    return  rotate_img


def Horizontal(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    binary = ThresholdProcess(blur)
    (h, w) = binary.shape
    shuiping = [0 for z in range(0, h)]
    for j in range(0, h):
        for i in range(0, w):
            if binary[j, i] == 0:
                shuiping[j] += 1
                binary[j, i] = 255
    for j in range(0, h):
        for i in range(0, shuiping[j]):
            binary[j, i] = 0
    return binary, shuiping


def Vertical(gray):
    binary = ThresholdProcess(gray)
    (h, w) = binary.shape
    chuizhi = [0 for z in range(0, w)]
    for j in range(0, w):
        for i in range(0, h):
            if binary[i, j] == 0:
                chuizhi[j] += 1
                binary[i, j] = 255
    for j in range(0, w):
        for i in range((h - chuizhi[j]), h):
            binary[i, j] = 0
    return binary


def SegLine(a):
    min_line = 1000
    line = None
    top = int(len(a) * 0.34)     # 0.15/0.2/0.25/0.3
    bottom = int(len(a) - top)
    for m in range(top, bottom):
        if a[m] < min_line:
            min_line = a[m]
            line = m
    return line


def SegLicensePlate(line, img):
    crop_img_1 = img[0:line+2, 0:img.shape[1]] ##2
    crop_img_2 = img[line+1:img.shape[0], 0:img.shape[1]] #1-2
    crop_img11 = cv2.resize(crop_img_1, (crop_img_2.shape[1], crop_img_2.shape[0]))
    output = np.hstack((crop_img11, crop_img_2))
    return crop_img_1, crop_img_2, crop_img11, output


def cropImg(shuiping, img):
    maxVal1 = 0
    maxVal2 = 0
    top = int(len(shuiping) * 0.2)
    bottom = len(shuiping) - top
    for z in range(0, top):
        if z+1 < len(shuiping):
            diff = abs(shuiping[z+1] - shuiping[z])
            if diff >= maxVal1:
                maxVal1 = diff
                x1 = z
    for z in range(bottom-1, len(shuiping)):
        if z + 1 < len(shuiping):
            diff = abs(shuiping[z + 1] - shuiping[z])
            if diff >= maxVal2:
                maxVal2 = diff
                x2 = z + 1
    crop = img[x1:x2, ]
    return crop


def main():
    count = 0
    start = time.time()
    LowThreshold = 70
    for filename in os.listdir(filepath):
        if count >= 2000:
            break
        count += 1
        img_path = os.path.join(filepath, '%s' % filename)
        img = cv2.imread(img_path)
        # img_pad1 = cv2.copyMakeBorder(img, 0, 0, 5, 1, cv2.BORDER_CONSTANT, 
        #                                value=(255, 255, 255))
        img_pad2 = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_REPLICATE)
        # img_pad3 = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
        # img_pad4 = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
        # img_pad5 = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_WRAP)

        # h, w = img.shape[0], img.shape[1]
        # M = cv2.getRotationMatrix2D((h/2, w/2), 2, 1)
        # img_src = cv2.warpAffine(img, M, (w, h))

        gray_src = cv2.cvtColor(img_pad2.copy(), cv2.COLOR_BGR2GRAY)

        binary = ThresholdProcess(gray_src)
        img_contour = ContoursDetect(img_pad2.copy(), binary)
        edges = EdgeDetect(img_contour, LowThreshold)
        x1, x2, y1, y2, img_Line_h = LineDetect_Haugh(img_pad2.copy(), edges)
        rotate_img_h = Rotate(img_pad2.copy(), x1, x2, y1, y2, filename, count)
       # x1, x2, y1, y2, img_Line_lsd = LineDetect_LSD(img_pad2.copy(), edges)
       # rotate_img_lsd = Rotate(img_pad2.copy(), x1, x2, y1, y2, filename, count)
        #cv2.imwrite('/data/wangyf/datasets/svt-plate/daul1-haugh/%s' % filename, rotate_img)

        # h_lsd, w_lsd = rotate_img_lsd.shape[0], rotate_img_lsd.shape[1]
        # rotate_img_lsd = rotate_img_lsd[3:h_lsd-3, 3:w_lsd-3]
        h_h, w_h= rotate_img_h.shape[0], rotate_img_h.shape[1]
        rotate_img_h = rotate_img_h[3:h_h-3, 3:w_h-3]
        rotate_img_h = cv2.resize(rotate_img_h, (150, 80), interpolation=cv2.INTER_CUBIC) # 60

        # kernel = np.ones((3, 3), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        #
        # erosion = cv2.erode(rotate_img_h, kernel, iterations=1)  # fushi
        # dilation = cv2.dilate(rotate_img_h, kernel, iterations=1) # pengzhang
        # opening = cv2.morphologyEx(rotate_img_h, cv2.MORPH_OPEN, kernel) # open
        # closing = cv2.morphologyEx(rotate_img_h, cv2.MORPH_CLOSE, kernel) # close
        # gradient = cv2.morphologyEx(rotate_img_h, cv2.MORPH_GRADIENT, kernel)
        #
        gray_rotate = cv2.cvtColor(rotate_img_h, cv2.COLOR_BGR2GRAY)
        #lsgray_rotate = FuzzyReducation(gray_rotate)

        binary_H, shuiping = Horizontal(gray_rotate)
        line_H = SegLine(shuiping)
        crop1, crop2, crop11, dst = SegLicensePlate(line_H, rotate_img_h)
        #cv2.imwrite('/data/wangyf/datasets/svt-plate/daul1-seg/%s' % filename, dst)

        cv2.imshow('img', img)
        #cv2.imshow('gray_src', gray_src)
        #cv2.imshow('ThresholdProcess', binary)
       # cv2.imshow('ww', img_pad2)
        cv2.imshow('img_contour', img_contour)
        cv2.imshow('EdgeDetect', edges)
        cv2.imshow('img_Line_h', img_Line_h)
        # cv2.imshow('img_Line_lsd', img_Line_lsd)
        cv2.imshow('rotate_img_h', rotate_img_h)
        # cv2.imshow('rotate_img_lsd', rotate_img_lsd)
        # cv2.imshow('erosion', erosion)
        # cv2.imshow('dilation', dilation)
        # cv2.imshow('opening', opening)
        # cv2.imshow('closing', closing)
        # cv2.imshow('gradient', gradient)

        #cv2.imshow('binary_H', binary_H)
       # cv2.imshow('crop1', crop1)
       # cv2.imshow('crop2', crop2)
        cv2.imshow('dst', dst)

        cv2.waitKey(0)

    end = time.time()
    print(count / (end - start))



if __name__ == '__main__':
    main()
