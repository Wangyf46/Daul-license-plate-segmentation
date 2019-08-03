#-*-coding:utf-8-*-
from __future__ import print_function

import numpy as np
import cv2
import sys
import os
import ipdb


def blur_edge(img, d=1): # 3
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    dst = img*w + img_blur*(1-w)
    # cv2.imshow('img_pad', img_pad)
    # cv2.imshow('img_blur', img_blur)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    return dst

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=10):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    # cv2.imshow('kern', kern)
    # cv2.waitKey(0)
    return kern


def FuzzyReducation(gray_src):
    gray_src = np.float32(gray_src)/255.0
    edges = blur_edge(gray_src)
    IMG = cv2.dft(edges, flags=cv2.DFT_COMPLEX_OUTPUT)

    defocus = '--circle'

    def update(_):
        # ang = np.deg2rad(cv2.getTrackbarPos('angle', 'wiener')) # x * pi / 180
        # d = cv2.getTrackbarPos('d', 'wiener')
        # noise = 10**(-0.1*cv2.getTrackbarPos('SNR (db)', 'wiener'))

        ang = np.deg2rad(140)
        d = 0
        noise = 10**(-0.1*9)

        if defocus:
            psf = defocus_kernel(d)
        else:
            psf = motion_kernel(ang, d) ##TODO
        psf /= psf.sum()
        psf_pad = np.zeros_like(gray_src)
        kh, kw= psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)

        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
        RES = cv2.mulSpectrums(IMG, iPSF, 0)
        res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        res = np.roll(res, -kh//2, 0)
        res = np.roll(res, -kw//2, 1)
        return res

    # cv2.namedWindow('wiener')
    # cv2.createTrackbar('angle', 'wiener', 135, 180, update)
    # cv2.createTrackbar('d', 'wiener', 1, 20, update)
    # cv2.createTrackbar('SNR (db)', 'wiener', 25, 50, update)
    res = update(None)
    return np.uint8(res * 255.0)


if __name__ == '__main__':
    filepath = '/data/wangyf/datasets/svt-plate/daul2'
    count = 0
    LowThreshold = 70
    for filename in os.listdir(filepath):
        if count >= 500:
            break
        count += 1
        img_path = os.path.join(filepath, '%s' % filename)
        img = cv2.imread(img_path)
        gray_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = FuzzyReducation(gray_src)
        print(res)
        cv2.imshow('res', res)
        cv2.waitKey(0)