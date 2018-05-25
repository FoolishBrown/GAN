#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import skimage
from tqdm import trange
import datetime
import tensorflow.contrib.slim as slim
import os
import cv2
import re
import time
import scipy.misc
from tensorflow.contrib.tensorboard.plugins import projector
import scipy
import nets_frontalization as Nets
import utils
from tqdm import trange
import sys
sys.path.insert(0,'./test_for_lfw')
import read_image_path as readlist
from resnet_yd import resnet_yd

flags=tf.app.flags
flags.DEFINE_integer('lfw_batch_size',25,'')
flags.DEFINE_integer('lfw_output_size',96,'')
flags.DEFINE_integer('lfw_output_channel',3,'')
flags.DEFINE_string('lfw_data_root','/home/sunkejia/emp/lfw_crop/','')
flags.DEFINE_string('lfw_save_root','/world/data-gpu-58/wangyuequan/data_sunkejia/lfw_synthesis_final/Setting2_104081/','')
flags.DEFINE_string("lfw_right",'/home/sunkejia/sourcecode/pycharm/GAN/test_for_lfw/lfw_right.txt', '')
flags.DEFINE_string("lfw_left",'/home/sunkejia/sourcecode/pycharm/GAN/test_for_lfw/lfw_left.txt', '')
FLAGS=flags.FLAGS
class Test(object):

    def __init__(self,input_size,input_channel,gpus_list,check_point=None,save_path=None,test_count=10,model_id=[],fullmodelroot=None,pose=7,noise=50):
        self.pose_c=pose
        self.input_size=input_size
        self.input_channel=input_channel
        self.gpus_list=gpus_list
        self.check_point=check_point
        self.noise_z=noise
        self.path=save_path
        # self.mkdir_result(self.check_point)#savepath
        # self.model_count=len(model_id)
        self.model_id=model_id
        self.test_count=test_count
        self.output=512
        self.fullmodelroot=fullmodelroot

        self.mkdir_result(save_path)
    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))


    def predict_drgan(self,batch_data,noise=None,pose=None):
        # _,output_en= Nets.netG_encoder_gamma(batch_data)
        _,_,output_en=resnet_yd(batch_data)
        # pose=tf.expand_dims(pose,1)
        # pose=tf.expand_dims(pose,1)
        # pose=tf.cast(pose,dtype=tf.float32)
        # print pose,output_en
        # sample_add_z = tf.concat([output_en,pose],1)

        sample_add_zp = tf.concat([output_en,noise],1)
        output_de_middle = Nets.netG_deconder_gamma(sample_add_zp,self.input_channel)
        output_de = Nets.merge_net_16_unet(output_de_middle, batch_data)

        return output_de

    def inference(self):
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpus_list
        # tf.set_random_seed(20)
        with tf.Graph().as_default():
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
            input_pose = tf.placeholder(dtype= tf.int32,shape=[None],name='input_poses')#pose
            index = tf.placeholder(tf.int32, None,name='input_nums')
            poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
            noise = tf.random_uniform(shape=(index,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
            sample=self.predict_drgan(input_data,noise=noise,pose=poses)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True
            train_vars = tf.trainable_variables()
            self.vars = [var for var in train_vars if 'slerpcode' not in var.name]
            with tf.Session(config=config) as sess:
                try:
                    tf.global_variables_initializer().run()
                except:
                    tf.initialize_all_variables().run()
                saver = tf.train.Saver(self.vars)
                if self.fullmodelroot!=None:
                    saver.restore(sess,self.fullmodelroot)
                else:
                    saver.restore(sess,self.check_point+'-'+self.model_id[0])
                im_path = open(FLAGS.lfw_right,'r').readlines()
                X_num=len(im_path)
                img_path = []
                img_batch = []

                for i in range(X_num):
                    img_batch.append(readlist.read_image_3channel(im_path[i], size=[self.input_size, self.input_size], root=FLAGS.lfw_data_root))
                    img_path.append(im_path[i].strip())
                    if len(img_batch) == FLAGS.lfw_batch_size or i == X_num - 1:
                        ind=len(img_batch)
                        pose = np.zeros([len(img_batch)])+3
                        ims = sess.run(sample, feed_dict={
                            input_data: img_batch,
                            input_pose:pose,
                            index:ind
                        })
                        for idx,im in enumerate(ims):

                            savepath=os.path.join(self.path,img_path[idx])
                            # print  savepath,os.path.basename(savepath)
                            self.mkdir_result(os.path.dirname(savepath))
                            # print os.path.dirname(savepath)
                            cv2.imwrite(savepath,inverse_transform(im))
                        img_path=[]
                        img_batch=[]
                im_path = open(FLAGS.lfw_right,'r').readlines()
                img_path = []
                img_batch = []
                X_num=len(im_path)

                for i in range(X_num):
                    img_batch.append(readlist.read_image_3channel(im_path[i], size=[self.input_size, self.input_size], root=FLAGS.lfw_data_root))
                    img_path.append(im_path[i].strip())
                    if len(img_batch) == FLAGS.lfw_batch_size or i == X_num - 1:

                        pose = np.zeros([len(img_batch)]) + 3
                        ims = sess.run(sample, feed_dict={
                            input_data: img_batch,
                            input_pose: pose,
                            index: len(img_batch)
                        })
                        for idx, im in enumerate(ims):
                            savepath = os.path.join(self.path, img_path[idx])
                            self.mkdir_result(os.path.dirname(savepath))
                            cv2.imwrite(savepath, inverse_transform(im))
                        img_path = []
                        img_batch = []


def inverse_transform(images):
    return (images + 1.) * 127.5

def saveimage(path,saveimg,savedata,modelcount,testcount,pose,inputsize,channel):
    for id_i in xrange(testcount):
        for p in xrange(pose):
            for p_i in xrange(pose):
                newimg=np.zeros([inputsize,(1+modelcount)*inputsize,channel])
                newimg[:,:inputsize,:]=savedata[id_i,p,:,:,:]
                for m_i in xrange(modelcount):
                   newimg[:,inputsize*(m_i+1):(m_i+2)*inputsize,:]= saveimg[id_i,m_i,p,p_i,:,:,:]
                im=inverse_transform(newimg)
                cv2.imwrite('{}/{:02}_{:02}_{:02}_{:02}.png'.format(path,id_i,p,p_i,modelcount),im)
def main(modelname,savepath,gpu):
    t = Test(FLAGS.lfw_output_size, FLAGS.lfw_output_channel, gpus_list=gpu,fullmodelroot=modelname,save_path=savepath,
             test_count=10)
    t.inference()


if __name__=='__main__':
    modelname="./logdir_caspeal/logdir_frontalization/20171101_gpu_0_vMultiPIE_Unet_finetun/checkpoint/frontalization-104081"
    #"./logdir_caspeal/logdir_share_gan_disentangle_light_pose/20170831_gpu_1_vdisentangle0.3/checkpoint/checkpoint_save/share_gan_disentangle_light_pose",

    main(modelname,FLAGS.lfw_save_root,'3')