import os
import cv2
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import copy
import ipdb


def main():
    i = 0
    start = time.time()
    filepath = '/data/wangyf/datasets/svt-plate/daul1'
    for filename in os.listdir(filepath):
        i += 1
        img_path = os.path.join(filepath, '%s'%filename)
        img = cv2.imread(img_path)
        image = copy.deepcopy(img)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Height, Width = gray.shape
        # gray_pad = cv2.copyMakeBorder(gray, 0, 0, 5, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # plt.hist(gray.ravel(), 256)
        # plt.show()
        ##
        thresh = int(np.sum(gray) * 1.0 / (Height * Width))
        ret, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            diff_x = np.max(Xs) - np.min(Xs)
            diff_y = np.max(Ys) - np.min(Ys)
            if diff_x > (Width / 2) and diff_y > (Height / 2):
                cv2.drawContours(image, [box], 0, (0, 255, 0), 1)   # -1
            # x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        lowThreshold = 70
        edges = cv2.Canny(blur, lowThreshold, lowThreshold * 2.5, apertureSize=3)  ## TODO: 150/200/100
        '''
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(edges)[0]
        #drawn_img = lsd.drawSegments(image, lines)
        max_lsd = 0
        x0_lsd = 0
        x1_lsd = 0
        y0_lsd = 0
        y1_lsd = 0
        for l in lines:
            x0, y0, x1, y1 = l.flatten()
            diff = abs(x1-x0)
            if diff > max_lsd:
                max_lsd = diff
                x0_lsd = x0
                x1_lsd = x1
                y0_lsd = y0
                y1_lsd = y1
        cv2.line(img, (x0_lsd, y0_lsd), (x1_lsd, y1_lsd), (0, 0, 255), 1)
        x1 = x0_lsd
        x2 = x1_lsd
        y1 = y0_lsd
        y2 = y1_lsd

        '''
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 15) ## TODO: 0/1/2
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        if x1 == x2 or y1 == y2:
            print(filename)
            cv2.imwrite('/data/wangyf/datasets/svt-plate/daul1-haugh/%s' % filename, img)
            continue
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        print(rotate_angle, i)
        rotate_img = ndimage.rotate(img, rotate_angle)  ## TODO

        #rotate_img = cv2.resize(rotate_img, (Width, Height))

        # gray = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # (h, w) = thresh.shape
        # thresh1 = copy.deepcopy(thresh)
        # ## shuiping
        # shuiping = [0 for z in range(0, h)]
        # for j1 in range(0, h):
        #     for i1 in range(0, w):
        #         if thresh1[j1, i1] == 0:
        #             shuiping[j1] += 1
        #             thresh1[j1, i1] = 255
        # for j2 in range(0, h):
        #     for i2 in range(0, shuiping[j2]):
        #         thresh1[j2, i2] = 0
        # #print(shuiping)
        # maxVal1 = 0
        # maxVal2 = 0
        # top = int(len(shuiping) * 0.2)
        # bottom = len(shuiping) - top
        # #print(top, bottom)
        # for z in range(0, top):
        #     if z+1 < len(shuiping):
        #         diff = abs(shuiping[z+1] - shuiping[z])
        #         if diff >= maxVal1:
        #             maxVal1 = diff
        #             x1 = z
        # #print(x1, maxVal1)
        # for z in range(bottom-1, len(shuiping)):
        #     if z + 1 < len(shuiping):
        #         diff = abs(shuiping[z + 1] - shuiping[z])
        #         if diff >= maxVal2:
        #             maxVal2 = diff
        #             x2 = z + 1
        # #print(x2, maxVal2)
        # crop = rotate_img[x1:x2, ]



        '''
        rotate_gray = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
        #rotate_blur = cv2.GaussianBlur(rotate_gray, (9, 9), 0)  ## TODO  (9,9)
        rotate_edges = cv2.Canny(rotate_gray, 50, 150, apertureSize=3)
        contours, hierarchy = cv2.findContours(rotate_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_LIST/cv2.RETR_EXTERNAL
        area = cv2.arcLength(contours[0], True)
        for mid_c in contours:
            if area <= cv2.arcLength(mid_c, True):
                area = cv2.arcLength(mid_c, True)  ## TODO
                dst_c = mid_c
                #print(area)
        # x, y, w, h = cv2.boundingRect(contours[-1])
        # cv2.rectangle(rotate_img, (x, y), (x + w, y + h), (0, 255, 0), 0)
        rect = cv2.minAreaRect(dst_c)   ##TODO
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        Xs = [i[0] for i in box]
        Xs= sorted(Xs)
        x1 = Xs[1]
        x2 = Xs[2]
        Ys = [i[1] for i in box]
        Ys = sorted(Ys)
        y1 = Ys[1]
        y2 = int((Ys[2] + Ys[3]) / 2.0)
        crop_img = rotate_img[y1:y2, x1:x2]
        '''

        cv2.imshow('img', img)
        cv2.imshow('image', image)
        cv2.imshow('gray', gray)
        cv2.imshow('edges', edges)
        cv2.imshow('rotate_img', rotate_img)
        cv2.waitKey(0)

        if i >= 200:
            break
        cv2.imwrite('/data/wangyf/datasets/svt-plate/daul1-haugh/%s' % filename, rotate_img)


    end = time.time()
    print((end - start) / i)

if __name__ == '__main__':
    main()