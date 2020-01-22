# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:35:40 2019

@author: Administrator
"""

import numpy as np
import cv2

#打开图像，显示图像，存储图像
path=r"d:\Users\Pictures\Saved Pictures\3.jpg"
img = cv2.imread(path)  
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite(r'd:\Users\Pictures\Saved Pictures\first.jpg',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

#对一张图像进行缩放，观察其分辨率
#获取图像的基本
path=r"d:\Users\Pictures\Saved Pictures\3.jpg"
img = cv2.imread(path, 1)
imgInfo = img.shape
height = imgInfo[0]
width =imgInfo[1]
dstHeight = int(height/2)
dstWidth = int(width/2)
#创建空白模板，其中np.uint8代表图片的数据类型0-255
dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
#对新的图像坐标进行重新计算，对矩阵进行行列遍历
for i in range(0, dstHeight):
    for j in range(0, dstWidth):
        iNew = int(i*(height*1.0 / dstHeight))
        jNew = int(j*(width*1.0/dstWidth))
        dstImage[i, j] = img[iNew, jNew]
cv2.imshow('dst', dstImage)
cv2.waitKey(0)

#降低灰度分辨率
path=r"d:\Users\Pictures\Saved Pictures\3.jpg"
img = cv2.imread(path)
imgInfo = img.shape
height = imgInfo[0]
width =imgInfo[1]
#创建空白模板，其中np.uint8代表图片的数据类型0-255
dstImage = np.zeros((height, width)) 
for i in range(0,height):
    for j in range(0,width):
        dstImage[i,j]=1/3*img[i,j,0]+1/3*img[i,j,1]+1/3*img[i,j,2]
#why uint8
cv2.imshow('dst1', dstImage.astype("uint8"))
cv2.waitKey(0)
