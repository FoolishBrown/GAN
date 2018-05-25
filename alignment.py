# -*- coding: utf-8 -*-
"""
@author: Saber
"""
import math
import cv2
import numpy as np
# import matplotlib
import skimage


def transform(x, y, ang, s0, s1):
    '''
    @x:x point
    @y:y point
    @ang:angle
    @s0:size of original image
    @s1:size of target image
    '''
    x0 = x - s0[1] / 2
    y0 = y - s0[0] / 2
    xx = x0 * math.cos(ang) - y0 * math.sin(ang) + s1[1] / 2
    yy = x0 * math.sin(ang) + y0 * math.cos(ang) + s1[0] / 2
    return xx, yy


def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


def guard(x, N):
    b = []
    for i in x:
        if i < 1: i = 1
        if i > N: i = N
        b.append(i)
    return b


'''
传入参数的时候应该把大小调整过
'''


def align(img, f5pt, crop_size, ec_mc_y, ec_y, scale=0, norote=False):
    '''
    @img:某一张图片的
    @f5pt:
    @crop_size:
    @ec_mc_y:
    @ec_y:
    '''
    # 取眼睛的中点
    e_x = (f5pt[0][0] + f5pt[1][0]) / 2.0
    e_y = (f5pt[0][1] + f5pt[1][1]) / 2.0

    # 嘴巴的中点
    m_x = (f5pt[3][0] + f5pt[4][0]) / 2.0
    m_y = (f5pt[3][1] + f5pt[4][1]) / 2.0

    ang_tan = float(e_x - m_x) / (e_y - m_y)
    ang = -math.atan(ang_tan) / math.pi * 180
    if norote or scale != 0:
        img_rot = img
        eyec = np.round([e_x, e_y])
        mouthc = np.round([m_x, m_y])
    else:
        img_rot = rotate_about_center(img, ang)
        ang = -ang / 180.0 * math.pi
        e_xx, e_yy = transform(e_x, e_y, ang, img.shape, img_rot.shape)
        eyec = np.round([e_xx, e_yy])
        m_xx, m_yy = transform(m_x, m_y, ang, img.shape, img_rot.shape)
        mouthc = np.round([m_xx, m_yy])
    imgh = img.shape[0]
    imgw = img.shape[1]
    if scale == 0:
        resize_scale = ec_mc_y / float(mouthc[1] - eyec[1])
    else:
        resize_scale = scale
    img_resize = cv2.resize(img_rot, (int(img_rot.shape[0] * resize_scale), int(img_rot.shape[1] * resize_scale)))
    eyec2 = (eyec - [img_rot.shape[1] / 2.0, img_rot.shape[0] / 2.0]) * resize_scale + [img_resize.shape[1] / 2.0, img_resize.shape[0] / 2.0]
    eyec2 = np.round(eyec2)
    img_crop = np.zeros((crop_size, crop_size, img.shape[2]))
    crop_y = eyec2[1] - ec_y
    crop_y_end = crop_y + crop_size
    crop_x = eyec2[0] - np.floor(crop_size / 2.0)
    crop_x_end = crop_x + crop_size
    box = [crop_x, crop_x_end, crop_y, crop_y_end]
    box = np.asarray(box, dtype=np.int32)
    box[box < 0] = 0

    temp_row = img_resize.shape[0]
    temp_col = img_resize.shape[1]
    x_add = 0
    y_add = 0
    if img_resize.shape[0] < box[3]:
        temp_row = box[3]
        x_add = box[3] - tem
    if img_resize.shape[1] < box[1]:
        temp_col = box[1]
    #####只写了右侧补齐，没写做左侧补齐
    img_temp = np.zeros((int(temp_row), int(temp_col), int(img.shape[2])))
    img_temp[0:img_resize.shape[0], 0:img_resize.shape[1], :] = img_resize[:, :, :]
    box = [int(i) for i in box]
    try:
        img_crop[:, :, :] = img_temp[box[2]:box[3], box[0]:box[1], :]
    except:
        img_temp = img_temp[box[2]:box[3], box[0]:box[1], :]
        img_crop = cv2.resize(img_temp, (crop_size, crop_size))
    cropped = img_crop
    cropped = cropped[:, :, :]
    return img_resize, eyec2, cropped, resize_scale