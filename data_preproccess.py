import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import os
import sys

import sys
sys.path.insert(0,'/home/sunkejia/sourcecode/pycharm/tupu_backend_server/aitupu/')
from interface.http_interface_geometry import FaceGeometry
from common import ip_manager
from Alignrotate import align

url = ip_manager.get_server_url(ip_manager.SERVERID_FACEGEOMETRY, version='model_3D_landmarks')
fg = FaceGeometry()
filename='/home/sunkejia/sourcecode/pycharm/GAN/lfw_all.txt'
saveroot = '/home/sunkejia/emp/lfw_crop_2/'
dataroot = '/world/data-gpu-90/origin_face/lfw/data/'

def detect_crops(saveroot,dataroot,filename):
    f = open(filename, 'r')
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
    f_landmarks = saveroot + 'landmarks.txt'
    f_land = open(f_landmarks, 'w')
    lines = f.readlines()
    lines = lines[::-1]
    for line_sub in lines:
        line = line_sub.strip()
        line_sub = dataroot + line
        line_sub = line_sub.split(' ')
        line_sp = line.split('/')
        for l_sub in line_sp[:-1]:
            path_temp = os.path.join(saveroot,l_sub)
            if not os.path.exists(path_temp):
                os.mkdir(path_temp)
        savedict = {}
        infodict = {}
        image_dict = {}
        img = cv2.imread(line_sub[0])
        h, w, c = img.shape
        margin = int((w - h) / 2.0)
        if margin>0:
            img = img[:, margin:margin + h]
        else:
            img = img[ -margin:-margin + w,:]
        image_dict[line_sub[0]] = {}
        image_dict[line_sub[0]]['binary_image'] = cv2.imencode('.jpg', img)[1]

        res = fg.inference(image_dict, url)
        try:
            for k in res:
                l_p = len(res[k]['landmarks'])
                point = np.zeros((l_p, 5, 2), dtype=np.int32)  # image only have one person
                for i, marks in enumerate(res[k]['landmarks']):
                    marks = np.array(marks, dtype=np.float32)
                    point[i, 0] = np.average(marks[36:42, :], axis=0)
                    point[i, 1] = np.average(marks[42:48, :], axis=0)
                    point[i, 2] = marks[30]
                    point[i, 3] = marks[60]
                    point[i, 4] = marks[64]
                    imresize, eye2, cropped, scale = align(img, point[i], crop_size=96, ec_mc_y=36, ec_y=30)#110 36 37
                    cv2.imwrite(saveroot + '/' + line, cropped)

                res[k]['point5_landmarks'] = point.tolist()
            f_land.write(json.dumps(res))
        except:
            print line
            cv2.imwrite(saveroot + '/' + line, cv2.resize(img,(96,96)))
    f.close()
if __name__=='__main__':
    detect_crops(saveroot, dataroot, filename)