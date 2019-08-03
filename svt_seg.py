#-*-coding:utf-8-*-
import os
import cv2
import math
import time
import argparse
import ipdb
import numpy as np
from scipy import ndimage


parse = argparse.ArgumentParser()
parse.add_argument('--filepath', type=str, default='/home/wangyf/daul2')
parse.add_argument('--FrameNum', type=int, default=2000,
                   help='test sample count')
parse.add_argument('--LowThreshold', type=int, default=50,
                   help='Canny edge detection lowthreshold')
parse.add_argument('--HighThreshold', type=int, default=180,
                   help='Canny edge detection HighThreshold')
parse.add_argument('--HoughThreshold', type=int, default=15,
                   help='Hough threshold')
args = parse.parse_args()


def ThresholdProcess(gray):
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary


def ContoursDetect(img, binary):
    Height, Width = binary.shape
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea)[-1]
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    for i in box:
        i[0] = np.where(i[0] <= 0, 1, i[0])
        i[0] = np.where(i[0] >= Width, Width - 1, i[0])
        i[1] = np.where(i[1] <= 0, 1, i[1])
        i[1] = np.where(i[1] >= Height, Height - 1, i[1])
    cv2.drawContours(img, [box], -1, (255, 0, 0), 2)
    return img


def EdgeDetect(img, LowThreshold, HighThreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, LowThreshold, HighThreshold, apertureSize=3)
    return edges


def LineDetect_Haugh1(img, edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, args.HoughThreshold, min_theta=0.3, max_theta=2.0)
    try:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    except:
        return 0, 0, 0, 0, img

    return x1, x2, y1, y2, img



def Rotate(img, x1, x2, y1, y2, count):
    if x1 == x2 or y1 == y2:
        # print('==',  count)
        return img
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    # print(rotate_angle, count)
    rotate_img = ndimage.rotate(img, rotate_angle)
    return rotate_img


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



def SegLine(projection):
    length = len(projection)
    min_line = 1000
    line = None
    top = int(length * 0.34)
    bottom = length - top  ## TODO
    for m in range(top, bottom):
        if projection[m] < min_line:
            min_line = projection[m]
            line = m
    return line


def SegLicensePlate(line, img):
    crop_img_1 = img[0:line + 2, 0:img.shape[1]]  ## TODO: 2
    crop_img_2 = img[line + 1:img.shape[0], 0:img.shape[1]]  ## TODO: 1-2
    # crop_img11 = cv2.resize(crop_img_1, (crop_img_2.shape[1], crop_img_2.shape[0]))
    # output = np.hstack((crop_img11, crop_img_2))
    return crop_img_1, crop_img_2


def show(img, img_pad, gray, binary, img_contour, img_new, edges, img_Line,
         rotate_img, rotate_gray, binary_H, crop1, crop2):
    cv2.imshow('img', img)
    # cv2.imshow('img_pad', img_pad)
    # cv2.imshow('gray', gray)
    # cv2.imshow('binary', binary)
    # cv2.imshow('img_contour', img_contour)
    # cv2.imshow('img_new', img_new)
    # cv2.imshow('edges', edges)
    # cv2.imshow('img_Line', img_Line)
    cv2.imshow('rotate_img', rotate_img)
    # cv2.imshow('binary_H', binary_H)
    cv2.imshow('crop1', crop1)
    cv2.imshow('crop2', crop2)
    cv2.waitKey(0)


def main(img_path):
    img = cv2.imread(img_path)
    img_pad = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_REPLICATE)  ##TODO
    gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
    binary = ThresholdProcess(gray)
    img_contour = ContoursDetect(img_pad.copy(), binary)
    Height, Width, _ = img_contour.shape

    t1 = int(Height * 0.35)
    img_contour_crop1 = img_contour[0:t1, ]
    img_contour_crop2 = img_contour[Height - t1:Height, ]
    img_new = np.vstack((img_contour_crop1, img_contour_crop2))

    edges = EdgeDetect(img_new, args.LowThreshold, args.HighThreshold)
    x1, x2, y1, y2, img_Line = LineDetect_Haugh1(img_pad.copy(), edges)
    rotate_img = Rotate(img_pad.copy(), x1, x2, y1, y2, count)
    H, W = rotate_img.shape[0], rotate_img.shape[1]
    rotate_img = rotate_img[3:H - 3, 3:W - 3]
    # rotate_img = cv2.resize(rotate_img, (150, 80)) # 60
    rotate_gray = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
    binary_H, projection_H = Horizontal(rotate_gray)
    line_H = SegLine(projection_H)
    crop1, crop2 = SegLicensePlate(line_H, rotate_img)
    # show(img, img_pad, gray, binary, img_contour, img_new, edges, img_Line, rotate_img, rotate_gray, binary_H, crop1, crop2)


if __name__ == '__main__':
    start = time.time()
    count = 0
    for filename in os.listdir(args.filepath):
        img_path = os.path.join(args.filepath, '%s' % filename)
        count += 1
        if count > args.FrameNum:
            break
        main(img_path)
    end = time.time()
    fps = count / (end - start)
    print(fps)
