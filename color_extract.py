#-*-coding:utf-8-*-


# -*- coding:utf-8 -*-

import cv2
import os
import numpy as np

import time

if __name__ == '__main__':

    filepath ='/home/wangyf/daul2'
    for filename in os.listdir(filepath):
        img_path = os.path.join(filepath, '%s' % filename)
        Img = cv2.imread(img_path)

        kernel_2 = np.ones((2,2),np.uint8)

        kernel_3 = np.ones((3,3),np.uint8)

        kernel_4 = np.ones((4,4),np.uint8)

        if Img is not None:
            HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
        Lower = np.array([255, 0, 0])#要识别颜色的下限
        Upper = np.array([255, 255, 0])#要识别的颜色的上限
        mask = cv2.inRange(HSV, Lower, Upper)
        erosion = cv2.erode(mask,kernel_4,iterations = 1)
        erosion = cv2.erode(erosion,kernel_4,iterations = 1)
        dilation = cv2.dilate(erosion,kernel_4,iterations = 1)
        dilation = cv2.dilate(dilation,kernel_4,iterations = 1)
        target = cv2.bitwise_and(Img, Img, mask=dilation)
        ret, binary = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        p=0
        for i in contours:#遍历所有的轮廓

            x,y,w,h = cv2.boundingRect(i)#将轮廓分解为识别对象的左上角坐标和宽、高

            #在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）

            cv2.rectangle(Img,(x,y),(x+w,y+h),(0,255,),3)

            #给识别对象写上标号

            font=cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(Img,str(p),(x-10,y+10), font, 1,(0,0,255),1)#加减10是调整字符位置

            p +=1

        print(p)

        cv2.imshow('target', target)

        cv2.imshow('Mask', mask)

        cv2.imshow("prod", dilation)

        cv2.imshow('Img', Img)

        #cv2.imwrite('Img.png', Img)
        cv2.waitKey(0)
    # while True:
    #      Key = chr(cv2.waitKey(15) & 255)
    #      if Key == 'q':
    #          cv2.destroyAllWindows()
    #          break
