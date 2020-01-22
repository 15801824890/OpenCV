# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:11:50 2019

@author: Administrator
"""

'''
对一副图像加噪，进行几何均值，算术均值，谐波，逆谐波处理
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

path=r"d:\Users\Pictures\Saved Pictures\4.jpg"
apple = cv2.imread(path)
apple = cv2.resize(cv2.cvtColor(apple,cv2.COLOR_BGR2RGB),(200,200))
plt.imshow(apple)
plt.axis("off")
plt.show()

def spNoisy(image,s_vs_p = 0.5,amount = 0.004):
    row,col,ch = image.shape

    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))  for i in image.shape]
    out[coords] = 1
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out

def GaussieNoisy(image,sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy.astype(np.uint8)

plt.imshow(GaussieNoisy(apple,25))
plt.show()

#算术均值滤波
def ArithmeticMeanOperator(roi):
    return np.mean(roi)

def ArithmeticMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = ArithmeticMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)

def rgbArithmeticMean(image):
    r,g,b = cv2.split(image)
    r = ArithmeticMeanAlogrithm(r)
    g = ArithmeticMeanAlogrithm(g)
    b = ArithmeticMeanAlogrithm(b)
    return cv2.merge([r,g,b])
plt.imshow(rgbArithmeticMean(apple))
plt.show()

#几何均值滤波ef 
def GeometricMeanOperator(roi):
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return p**(1/(roi.shape[0]*roi.shape[1]))
    
def GeometricMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = GeometricMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)

def rgbGemotriccMean(image):
    r,g,b = cv2.split(image)
    r = GeometricMeanAlogrithm(r)
    g = GeometricMeanAlogrithm(g)
    b = GeometricMeanAlogrithm(b)
    return cv2.merge([r,g,b])
plt.imshow(rgbGemotriccMean(apple))
plt.show()


#谐波均值
def HMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = scipy.stats.hmean(roi.reshape(-1))
    return roi

def HMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] =HMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)

def rgbHMean(image):
    r,g,b = cv2.split(image)
    r = HMeanAlogrithm(r)
    g = HMeanAlogrithm(g)
    b = HMeanAlogrithm(b)
    return cv2.merge([r,g,b])
plt.imshow(rgbHMean(apple))
plt.show()

#逆谐波均值
def IHMeanOperator(roi,q):
    roi = roi.astype(np.float64)
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))
def IHMeanAlogrithm(image,q):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] = IHMeanOperator(image[i-1:i+2,j-1:j+2],q)
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbIHMean(image,q):
    r,g,b = cv2.split(image)
    r = IHMeanAlogrithm(r,q)
    g = IHMeanAlogrithm(g,q)
    b = IHMeanAlogrithm(b,q)
    return cv2.merge([r,g,b])
plt.imshow(rgbIHMean(apple,2))
plt.show()

spApple = spNoisy(apple,0.5,0.1)
gaussApple = GaussieNoisy(apple,25)
plt.subplot(121)
plt.title("Salt And peper Image")
plt.imshow(spApple)
plt.axis("off")
plt.subplot(122)
plt.imshow(gaussApple)
plt.axis("off")
plt.title("Gauss noise Image")
plt.show()

arith_gs_apple = rgbArithmeticMean(gaussApple)
gemo_gs_apple = rgbGemotriccMean(gaussApple)
plt.subplot(121)
plt.title("Arithmatic to gsImage")
plt.imshow(arith_gs_apple)
plt.axis("off")
plt.subplot(122)
plt.imshow(gemo_gs_apple)
plt.axis("off")
plt.title("Geomotric to gsImage")
plt.show()

arith_sp_apple = rgbHMean(spApple)
gemo_sp_apple = rgbIHMean(spApple,3)
plt.subplot(121)
plt.title("H Mean to spImage")
plt.imshow(arith_sp_apple)
plt.axis("off")
plt.subplot(122)
plt.imshow(gemo_sp_apple)
plt.axis("off")
plt.title("IH mean to spImage")
plt.show()

arith_gs_apple = rgbHMean(gaussApple)
gemo_gs_apple = rgbIHMean(gaussApple,3)
plt.subplot(121)
plt.title("HMean to gsImage")
plt.imshow(arith_gs_apple)
plt.axis("off")
plt.subplot(122)
plt.imshow(gemo_gs_apple)
plt.axis("off")
plt.title("IHMean to gsImage")
plt.show()
