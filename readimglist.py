# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:46:22 2016
@author: Saber
"""
import skimage
import os
import numpy as np
import cv2
def read_imagelist_3channel(filelist,size=[64,64],root=''):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param： filelist 图像列表文件
    @size:网络需要的规格
    @return ：4D 的矩阵，样本数目
    '''
    fid=open(filelist)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    X=np.zeros((test_num,size[0],size[1],3))
    i =0
    for line in lines:
        word=line.strip().split(' ')
        filename=root+'/'+word[0]
        perfile,lasefile=os.path.splitext(filename)
        try:
            print filename
            im1=norm_img(cv2.imread(filename))
        except:
            print filename
        #     perfile=perfile.split('/')[-1]
        #     filename=perfile+'.jpg'
        #     print filename
            fdir=word[0].split('/')
            w=fdir[1].split('_')
            filename=root+'/'+fdir[0]+'/'+w[0]+'_'+w[1]+'_0001.jpg'
            print filename
        #     filename=r'/world/data-gpu-90/rex/lfw/data/lfw_lightcnn_96_rgb/'+word[0]
            im1 = norm_img(cv2.imread(filename))

        #        image =cv2.resize(im1,(size[0],size[1]))
        image=im1
        if image.ndim<3:
            print 'gray:'+filename
            X[i,:,:,0]=image[:,:]
            X[i,:,:,1]=image[:,:]
            X[i,:,:,2]=image[:,:]
        else:
            X[i,:,:,0]=image[:,:,0]
            X[i,:,:,1]=image[:,:,1]
            X[i,:,:,2]=image[:,:,2]
        i=i+1
    return X,test_num
#caffe.io.load_image()  float64 需要乘以255
#skimage.io.imread()  unit8    RGB
#cv2.imread() unit8  BGR
    #skimage.transform.resize   float64    *255
    #cv2.resize   unit8
def read_imagelist_1channel(filelist,size=[64,64],root=''):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param： filelist 图像列表文件
    @size:网络需要的规格
    @return ：4D 的矩阵，样本数目
    '''
    fid=open(filelist)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    X=np.zeros((test_num,size[0],size[1],1))
    i =0
    for line in lines:
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
        im1 =im1
        if im1.ndim<3:
#            print 'gray:'+filename
            X[i,:,:,0]=im1[:,:]
        else:
            print 'read_imagelist_1channel error!!!'
        i=i+1
    return X,test_num

def norm_img( img):
    return np.array(img)/127.5 - 1