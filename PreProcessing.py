import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def Gamma(img):#伽马矫正
    s = img.shape
    for i in range(s[0]):
        for j in range(s[1]):
            img[i][j] = img[i][j]**1.1
    return img

def HistEqual(img):#分区域直方图均衡化，用于图像增强
    #这个参数可以调
    clash = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clash.apply(img)
    return cl

def prep(img):
    cv2.GaussianBlur(img,(5,5),0)
    #cv2.Laplacian(img,cv2.CV_64F,ksize = 5)
    cl = HistEqual(img)
    sobelY = cv2.Scharr(img,cv2.CV_64F,0,1)
    return img