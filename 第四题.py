# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:10:27 2019

@author: Administrator
"""
'''
对一副图像进行傅立叶变换，显示频谱，取其5，50，150为截至频率，
进行频率域平滑，锐化，显示图像
'''

import numpy as np
from matplotlib import pyplot as plt
import cv2

path=r"d:\Users\Pictures\Saved Pictures\4.jpg" 

#巴特沃斯低通滤波器
img = cv2.imread(path,0) #直接读为灰度图像
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.show()
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到0-255
s1 = np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(s1,'gray')
plt.title('Frequency Domain')
plt.show()

#使用高斯平滑滤波器
def GaussianLowFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

img_d1 = GaussianLowFilter(img,10)
img_d2 = GaussianLowFilter(img,30)
img_d3 = GaussianLowFilter(img,50)
plt.subplot(131)
plt.axis("off")
plt.imshow(img_d1,cmap="gray")
plt.title('D_1 10')
plt.subplot(132)
plt.axis("off")
plt.title('D_2 30')
plt.imshow(img_d2,cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("D_3 50")
plt.imshow(img_d3,cmap="gray")
plt.show()

#高通锐化

def LaplaceFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = -4*(np.pi**2)*(dis**2)
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

img_d1 = LaplaceFilter(img,50)
plt.title('D_1 10')
plt.imshow(img_d1,cmap="gray")
plt.show()