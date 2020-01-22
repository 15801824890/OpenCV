# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:09:40 2019

@author: Administrator
"""

'''
对一副图像加噪声，进行平滑，锐化作用
'''
import cv2
import numpy as np
import random

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

path=r"d:\Users\Pictures\Saved Pictures\4.jpg"
img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#new_img=np.zeros(img.shape,np.uint8)
new_img=sp_noise(img,0.1)
#cv2.imshow(new_img)
cv2.imshow('noise_new_img',new_img.astype(np.uint8)) 
cv2.waitKey(0)

#定义kernel进行卷积

def imgMedianFilter(img,kernel):
    rows,cols=img.shape
    kernel_rows,kernel_cols=kernel.shape
    padding_rows,padding_cols=int((kernel_rows-1)/2),int((kernel_cols-1)/2)
    
    convolve_rows=rows+2*padding_rows
    convolve_cols=cols+2*padding_cols
    
    img_padding=np.zeros((convolve_rows,convolve_cols))
    img_padding[padding_rows:padding_rows+rows,padding_cols:padding_cols+cols]=img[:,:]
    
    img_median=np.zeros((rows,cols))
    
    for i in range(padding_rows,padding_rows+rows):
        for j in range(padding_cols,padding_cols+cols):
            img_median[i-padding_rows][j-padding_cols]=int(np.median(img_padding
                                                                    [i-padding_rows:i+padding_rows+1,
                                                                     j-padding_cols:j+padding_cols+1]))
    return img_median

#进行平滑
    
#3*3
kernel=np.random.randint(3,size=(3,3))
new_img1=imgMedianFilter(new_img,kernel)
cv2.imshow('new_img1',new_img1.astype(np.uint8))
cv2.waitKey(0)

#7*7
kernel=np.random.randint(3,size=(7,7))
new_img1=imgMedianFilter(new_img,kernel)
cv2.imshow('new_img1',new_img1.astype(np.uint8))
cv2.waitKey(0)

#锐化作用
def Sharp(image,flag1=0,flag2=0):
    h,w = image.shape
    iSharp = np.zeros((h, w, 3), np.uint8)
    for i in range(h-1):
        for j in range(w-1):
            if flag2 == 0:
                x = abs(image[i,j+1]-image[i,j])
                y = abs(image[i+1,j]-image[i,j])
            else:
                x = abs(image[i+1,j+1]-image[i,j])
                y = abs(image[i+1,j]-image[i,j+1])
            if flag1 == 0:
                iSharp[i,j] = max(x,y)
            else:
                iSharp[i,j] = x+y
    return iSharp 

image = img
iMaxSharp = Sharp(image)
iAddSharp = Sharp(image,1)
iRMaxSharp = Sharp(image,0,1)
iRAddSharp = Sharp(image,1,1)
cv2.imshow('iMaxSharp',iMaxSharp.astype('uint8'))
cv2.imshow('image',image.astype('uint8'))
cv2.imshow('iAddSharp',iAddSharp.astype('uint8'))
cv2.imshow('iRAddSharp',iRAddSharp.astype('uint8'))
cv2.imshow('iRMaxSharp',iRMaxSharp.astype('uint8'))
cv2.waitKey(0)



