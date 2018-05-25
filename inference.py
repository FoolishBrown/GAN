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

# flags=tf.app.flags
lfw_batch_size=256
lfw_output_size=96
lfw_output_channel=3
data_root='/home/sunkejia/emp/multi_one/'
save_root='/home/sunkejia/emp/multi_one/'
filelist='/home/sunkejia/sourcecode/pycharm/GAN/multipie_one.txt'
# FLAGS=flags.FLAGS
class Test(object):

    def __init__(self,input_size,input_channel,gpus_list,check_point=None,save_path=None,test_count=10,model_id=[],fullmodelroot=None,noise=50):
        self.input_size=input_size
        self.input_channel=input_channel
        self.gpus_list=gpus_list
        self.check_point=check_point
        self.noise_z=noise
        self.path=save_path
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
                print str_dir
                # os.mkdir('./{}'.format(str_dir))


    def predict_gan(self,batch_data,noise=None,pose=None):
        _,_,output_en=resnet_yd(batch_data)

        sample_add_zp = tf.concat([output_en,noise],1)
        output_de_middle = Nets.netG_deconder_gamma(sample_add_zp,self.input_channel)
        output_de = Nets.merge_net_16_unet(output_de_middle, batch_data)

        return output_de

    def slerp_gan(self,bath_data):
        '''
        init model for identity and slerpcode
        :return:
        '''
        with tf.name_scope('test'):
            _,_,self.encode_slerp= resnet_yd(bath_data, reuse=True)
            self.encode_slerp_z=tf.get_variable('code',[3,562])
            self.image_syn_slerp_middle = Nets.netG_deconder_gamma(self.encode_slerp_z, 3, reuse=True)
            self.image_syn_slerp = Nets.merge_net_16_unet(self.image_syn_slerp_middle, bath_data, reuse=True)

    def inference(self,slerp_flag=False):
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpus_list
        # tf.set_random_seed(20)
        with tf.Graph().as_default():
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
            input_pose = tf.placeholder(dtype= tf.int32,shape=[None],name='input_poses')#pose
            index = tf.placeholder(tf.int32, None,name='input_nums')
            noise = tf.random_uniform(shape=(index,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
            sample=self.predict_gan(input_data,noise=noise)
            self.slerp_gan(input_data)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True
            train_vars = tf.trainable_variables()
            self.vars = [var for var in train_vars if 'slerpcode' not in var.name]
            for i_model in self.model_id:
                with tf.Session(config=config) as sess:
                    try:
                        tf.global_variables_initializer().run()
                    except:
                        tf.initialize_all_variables().run()
                    saver = tf.train.Saver(self.vars)
                    if self.fullmodelroot!=None:
                        saver.restore(sess,self.fullmodelroot)
                    else:
                        saver.restore(sess,self.check_point+'-'+i_model)
                    im_path = open(filelist,'r').readlines()
                    X_num=len(im_path)
                    img_path = []
                    img_batch = []
                    count=0
                    for i in range(X_num):
                        img_batch.append(readlist.read_image_3channel(im_path[i], size=[self.input_size, self.input_size], root=data_root))
                        img_path.append(im_path[i].strip())
                        if len(img_batch) == lfw_batch_size or i == X_num - 1:
                            ind=len(img_batch)
                            pose = np.zeros([len(img_batch)])+3
                            ims = sess.run(sample, feed_dict={
                                input_data: img_batch,
                                input_pose:pose,
                                index:ind
                            })
                            if slerp_flag:
                                self.slerp_interpolation(input_data,img_batch,sess,savepath)
                            for idx,im in enumerate(ims):
                                img_path_sub=img_path[idx].split(' ')[0].split('/')
                                savepath=self.path
                                for p in img_path_sub:
                                    savepath=os.path.join(savepath,p)
                                    self.mkdir_result(os.path.dirname(savepath))
                                # print '{}_{}_syn.jpg'.format(savepath,self.check_point.split("/")[-1]+self.model_id),
                                # cv2.imwrite('{}_{}_{}_syn.jpg'.format(savepath,img_path_sub[-1],self.check_point.split("/")[-1]+self.model_id),inverse_transform(im))
                                # print savepath,img_path_sub[-1]
                                cv2.imwrite(savepath+'{08}.png'.i_model,inverse_transform(im))
                            count+=lfw_batch_size
                            print count
                            img_path=[]
                            img_batch=[]

    def slerp_interpolation(self,input_data,batch_image,sess,savepath,test_slerp_count=1):
        #用了patch的方法是不能做slerp—interpolation操作的
        # 球面差值法测试
        index_select = np.random.randint(0, batch_image.shape[0], test_slerp_count*2)
        encode_512 = sess.run(self.encode_slerp,
                             feed_dict={input_data: batch_image[index_select]})
        encode_50=np.random.uniform(high=1, low=-1, size=[test_slerp_count*2,self.noise_z])
        encode_562=np.concatenate([encode_512,encode_50],axis=1)
        encode_sub = np.split(encode_562,2,axis=0)
        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([utils.slerp(ratio, r1, r2) for r1, r2 in
                          zip(encode_sub[0], encode_sub[1])])
            z = np.reshape(z, [-1,562])
            z_decode = sess.run(self.image_syn_slerp,
                                feed_dict={input_data: batch_image[index_select],
                                    self.encode_slerp_z: z})
            decodes.append(z_decode)
        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        index_sub=np.split(index_select,2,axis=0)
        index_sub=np.split(index_sub[0],2,axis=0)
        for idx, img in enumerate(decodes):
            img = np.concatenate([[batch_image[index_sub[0][idx]]], img,
                                  [batch_image[index_sub[1][idx]]]], 0)
            img = utils.inverse_transform(img)[:, :, :, ::-1]
            utils.save_image(img, os.path.join('./{}'.format(savepath),
                                               'test{:08}_interp_G.png'.format(self.model_id )), nrow=10 + 2)

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
def main(model_id,check_path,gpus):
    t = Test(lfw_output_size, lfw_output_channel, gpus_list=gpus,
             check_point=check_path,
             save_path=save_root,
             model_id=model_id, test_count=10)
    t.inference()
if __name__=='__main__':
    model_id =['0','10372','12965','15558','18151','20744','23337','25930','2593','28523','31116','33709','36302','38895','5186','7779']
    t = Test(lfw_output_size, lfw_output_channel, gpus_list='3',
             check_point="./logdir_caspeal/logdir_FinalTrain_Setting2/20171116_gpu_0_vSetting2_fr_6631_resnet_original_setting2/checkpoint/FinalTrain_Setting2",
             save_path=save_root,
             model_id=model_id, test_count=10)
    t.inference()