# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:46:22 2016
@author: Saber
"""
import skimage
import os
import numpy as np
import cv2
def read_image_3channel(line,size=[64,64],root=''):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param： filelist 图像列表文件
    @size:网络需要的规格
    @return ：4D 的矩阵，样本数目
    '''
    word=line.strip().split(' ')
    filename=root+'/'+word[0]
    perfile,lasefile=os.path.splitext(filename)
    try:
        im1=norm_img(cv2.imread(filename))
    except:
        print filename
        r='/world/data-gpu-90/rex/lfw/data/lfw_lightcnn_96_rgb/'
        im1 = norm_img(cv2.imread(r+word[0]))
    crop_pix = int(round(im1.shape[0] - size[0]) / 2.0)

    cropped_image = cv2.resize(
        im1[crop_pix:crop_pix + size[0], crop_pix:crop_pix + size[0]]
        , (size[1], size[1]))
    return cropped_image
#caffe.io.load_image()  float64 需要乘以255
#skimage.io.imread()  unit8    RGB
#cv2.imread() unit8  BGR
    #skimage.transform.resize   float64    *255
    #cv2.resize   unit8
def read_image_1channel(line,size=[64,64],root=''):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param： filelist 图像列表文件
    @size:网络需要的规格
    @return ：4D 的矩阵，样本数目
    '''
    word=line.split('\n')
    perfile,lasefile=os.path.splitext(word[0])
    if root =='':
        filename=word[0]
    else:
        filename=root+'/'+word[0]
#        print filename
    try:
        print filename
        im1=norm_img(cv2.imread(filename,0))
    except:
        filename=perfile+'.bmp'
        filename=r'E:\database\LFW_test\lightcnn128\lfw\image\\'+filename
        print filename
        im1=cv2.imread(filename,0)
    return im1

def norm_img( img):
    return np.array(img)/127.5 - 1