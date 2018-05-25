#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
# import skimage
# import datetime
import tensorflow.contrib.slim as slim
import os
import cv2
import re
import time
# import scipy.misc
# from tensorflow.contrib.tensorboard.plugins import projector
import utils
import logging
import Net_8_13 as Nets
from tqdm import trange


class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,lr_alpha=0.5,lr_beta=0.5,lr_gamma=1,version='0',
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,test_interval=1,
                 data_loader_train=None,data_loader_valid=None,epoch=10000,log_dir='logdir',d_learning_rate=0.0002,g_learning_rate=0.0002,beta1=0.5,delta=0.2,
                 test_batch_size=10,gpus_list=None,write_type=0,model_name='dr_gan',noise_z=50,pose_c=7,random_seed=25,sample_pose_interval=10,
                 loss_type=0,Gloop=3,savepath='savename',imagepath='image',logfile='file.log',summary_dir='summary',warmup=True,discribe=''):
        self.lr_alpha=lr_alpha
        self.lr_beta = lr_beta
        self.lr_gamma = lr_gamma
        self.lr_delta=delta
        self.losstype=loss_type
        self.restored=restored
        self.d_lr=d_learning_rate
        self.g_lr=g_learning_rate
        self.beta1=beta1
        self.batch_size = batch_size
        self.test_batch_size=test_batch_size
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_size = output_size
        self.output_channel = output_channel
        self.sample_interval = sample_interval
        # self.interval = test_interval
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.sample_pose_interval = sample_pose_interval
        self.test_interval=test_interval
        self.epoch = epoch
        self.warmup=warmup
        #dir
        self.log_dir = log_dir
        self.savename=savepath
        self.result_path = imagepath
        if restored:
            self.check_point=check_point

            self.check_point_path=os.path.join(os.path.dirname(logfile),'checkpoint')
        else:
            self.check_point=check_point
            self.check_point_path=check_point

        self.logfile=logfile
        self.summarypath=summary_dir

        self.gpus_list=gpus_list
        self.model_name=model_name
        self.write_type=write_type

        self.noise_z=noise_z
        self.pose_c=pose_c
        self.ifsave=ifsave
        self.Gloop=Gloop
        self.class_nums=self.data_loader_train.class_nums

        self.random_seed=random_seed
        #save loss and vars
        self.lossG=None
        self.lossD=None
        self.varsG=None
        self.varsD=None
        self.version=version
        self.mkdir_result(self.log_dir)#save mode dir
        self.mkdir_result(self.summarypath)#save summary
        self.mkdir_result(self.savename)#save result dir
        self.mkdir_result(self.result_path)#save image dir
        self.mkdir_result(self.check_point_path)#checkpoint dir
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',datefmt='%a,%d %b %Y %H:%M:%S',filename=self.logfile,filemode='w')
        logging.info('gamma:{} alpha:{} beta:{} delta:{} poselabel:(-1,1), discribe{}'.format(lr_gamma,lr_alpha,lr_beta,delta,discribe))


        self.gpus_arr=np.asarray(self.gpus_list.split(','),np.int32)
        print 'use gpu nums:',len(self.gpus_arr)
        self.gpus_count=len(self.gpus_arr)
        if self.gpus_count>1:
            self.multigpus=True
        else:
            self.multigpus=False
        self.batch_idxs = self.data_loader_train.batch_idxs / self.gpus_count
    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))


    def predict_drgan_caspeal(self,batch_data,noise=None,pose=None,random_pose=None):
#        with tf.variable_scope('drgan'):
        with tf.name_scope('generator_encoder_decoder'):
            output_en = Nets.netG_encoder_gamma(batch_data)
        #----------noise
        # print 'noise:max{},min{}'.format(np.max(sampel_z),np.min(sampel_z))
            sample_add_z = tf.concat([output_en,noise],3)
            pose=tf.expand_dims(pose,1)
            pose=tf.expand_dims(pose,1)
            random_pose = tf.expand_dims(random_pose, 1)
            random_pose = tf.expand_dims(random_pose, 1)
            pose=tf.cast(pose,dtype=tf.float32)
            random_pose=tf.cast(random_pose,dtype=tf.float32)
            sample_add_zp = tf.concat([sample_add_z,pose],3)

            sample_add_random_pose=tf.concat([sample_add_z,random_pose],3)
        # print  sample_add_zp.shape
    #------------
            size_noise=noise.shape.as_list()[0]
        # print 'size_noise',pose.shape.as_list(),noise.shape.as_list()

            output_de = Nets.netG_deconder_gamma(sample_add_zp,self.output_channel)
            output_de_rp = Nets.netG_deconder_gamma(sample_add_random_pose,self.output_channel,reuse=True)
        tf.summary.histogram('generator/outputencoder',output_en)
        tf.summary.histogram('generator/inputdecoder',sample_add_zp)
        tf.summary.histogram('generator/outputdecoder',output_de)
        tf.summary.histogram('input_discrimintor/inputrealdata',batch_data)
        tf.summary.histogram('input_discrimintor/inputsyndata',output_de)
        tf.summary.histogram('input_discrimintor/inputsyndata',output_de_rp)

        with tf.name_scope('discriminator_total'):
            pidr_softmax,pidrlogits\
            = \
            Nets.Custom_netD_discriminator_idloss(batch_data,class_nums=self.data_loader_train.class_nums+1)
            ppsr_logits,prcontent\
            =\
            Nets.Custom_netD_discriminator_psloss(batch_data,posenum=self.pose_c)

            pidf1_softmax,pidf1logits\
            = \
            Nets.Custom_netD_discriminator_idloss(output_de,class_nums=self.data_loader_train.class_nums+1,reuse=True)
            ppsf1_logits,pf1content\
            =\
            Nets.Custom_netD_discriminator_psloss(output_de,posenum=self.pose_c,reuse=True)

            pidf2_softmax,pidf2logits\
            = \
            Nets.Custom_netD_discriminator_idloss(output_de_rp,class_nums=self.data_loader_train.class_nums+1,reuse=True)
            ppsf2_logits,pf2content\
            =\
            Nets.Custom_netD_discriminator_psloss(output_de_rp,posenum=self.pose_c,reuse=True)

        tf.summary.histogram('discriminator/real_id',pidrlogits)
        tf.summary.histogram('discriminator/fake1_id',pidf1logits)
        tf.summary.histogram('discriminator/fake2_id',pidf2logits)
        tf.summary.histogram('discriminator/real_pose',ppsr_logits)
        tf.summary.histogram('discriminator/fake1_pose',ppsf1_logits)
        tf.summary.histogram('discriminator/fake2_pose',ppsf2_logits)

        return pidr_softmax,pidrlogits,ppsr_logits,\
                pidf1_softmax,pidf1logits,ppsf1_logits,\
                pidf2_softmax,pidf2logits,ppsf2_logits,\
               output_de,output_en,output_de_rp

    def predict_drgan_3d6w(self,batch_data,index,pose=None):
        with tf.name_scope('generator_encoder_decoder'):
            output_en = Nets.Custom_netG_encoder(batch_data)
            shape = output_en.get_shape().as_list()
            # print shape
        #----------noise
            noise = tf.random_uniform(shape=(index, 6, 6, 50), minval=-1, maxval=1, dtype=tf.float32,name='input_noise')
            ps_noise_map=Nets.Custom_netG_pose_and_noise(output_en,shape,pose,noise)

            pose_sp=tf.split(pose,2,axis=0)
            pose_ex=tf.concat(pose_sp[::-1],axis=0)
            ps_noise_map_exchange = Nets.Custom_netG_pose_and_noise(output_en, shape, pose_ex, noise,reuse=True)
            # pose_1=tf.zeros_like(pose)
            # ps_noise_map_1=Nets.Custom_netG_pose_and_noise(output_en,shape,pose_1,noise,reuse=True)
            # pose_2=tf.ones_like(pose)
            # ps_noise_map_2=Nets.Custom_netG_pose_and_noise(output_en,shape,pose_2,noise,reuse=True)
            # pose_3=tf.ones_like(pose)*0.5
            # ps_noise_map_3=Nets.Custom_netG_pose_and_noise(output_en,shape,pose_3,noise,reuse=True)
            # size_noise=noise.shape.as_list()[0]
            output_de = Nets.Custom_netG_decoder(output_en,ps_noise_map)
            output_de_exchange = Nets.Custom_netG_decoder(output_en,ps_noise_map_exchange,reuse=True)
            # output_de_zeros=Nets.Custom_netG_decoder(output_en,ps_noise_map_1,reuse=True)
            # output_de_halves=Nets.Custom_netG_decoder(output_en,ps_noise_map_3,reuse=True)
            # output_de_ones=Nets.Custom_netG_decoder(output_en,ps_noise_map_2,reuse=True)
        #
        # tf.summary.histogram('input_discrimintor/inputrealdata',batch_data)
        # tf.summary.histogram('input_discrimintor/inputsyndata',output_de)
        with tf.name_scope('discriminator_total'):
            # softmax_ad_real,adlogits_real=\
            # Nets.Custom_netD_discriminator_adloss(batch_data)

            softmax_id_real,idlogits_real= \
            Nets.Custom_netD_discriminator_idloss(batch_data,class_nums=self.data_loader_train.class_nums+1)

            pslogits_real,content_feature_real=\
            Nets.Custom_netD_discriminator_psloss(batch_data)

            # softmax_ad_fake, adlogits_fake = \
            # Nets.Custom_netD_discriminator_adloss(output_de,reuse=True)

            softmax_id_fake, idlogits_fake = \
            Nets.Custom_netD_discriminator_idloss(output_de, class_nums=self.data_loader_train.class_nums+1,reuse=True)

            pslogits_fake, content_feature_fake = \
            Nets.Custom_netD_discriminator_psloss(output_de,reuse=True)

            # softmax_ad_fake_ex, adlogits_fake_ex = \
            # Nets.Custom_netD_discriminator_adloss(output_de, reuse=True)

            softmax_id_fake_ex, idlogits_fake_ex = \
            Nets.Custom_netD_discriminator_idloss(output_de_exchange, class_nums=self.data_loader_train.class_nums+1, reuse=True)
            #
            pslogits_fake_ex, content_feature_fake_ex = \
            Nets.Custom_netD_discriminator_psloss(output_de_exchange, reuse=True)

        # tf.summary.histogram('discriminator/real_d',predict_r)
        # tf.summary.histogram('discriminator/fake_d',predict_f)
        # tf.summary.histogram('discriminator/real_id',predict_r_label)
        # tf.summary.histogram('discriminator/fake_id',predict_f_label)
        # tf.summary.histogram('discriminator/real_pose',predict_r_pose)
        # tf.summary.histogram('discriminator/fake_pose',predict_f_pose)

        return softmax_id_real, idlogits_real, \
               pslogits_real, content_feature_real, \
               softmax_id_fake, idlogits_fake, \
               pslogits_fake, content_feature_fake, \
                softmax_id_fake_ex,idlogits_fake_ex,\
                pslogits_fake_ex,content_feature_fake_ex,\
                output_de

    def predict_drgan_multipie(self, batch_data, noise=None, pose=None):
        #        with tf.variable_scope('drgan'):
        with tf.name_scope('generator_encoder_decoder'):
            output_en = Nets.netG_encoder_gamma(batch_data)
            pose_reshape = tf.reshape(pose, [-1, 1, 1, self.pose_c])
            pose_float = tf.cast(pose_reshape, dtype=tf.float32)
            pose_add_noise=tf.concat([pose_float,noise],3)
            sample_add_pn = tf.concat([output_en,pose_add_noise ], 3)

            #pose reverse
            pose_add_noise_sp=tf.split(pose_add_noise,2,axis=0)
            pose_add_noise_ex=tf.concat(pose_add_noise_sp[::-1],axis=0)
            sample_add_pn_ex = tf.concat([output_en,pose_add_noise_ex],3)

            output_de = Nets.netG_deconder_gamma(sample_add_pn, self.output_channel)
            output_de_rp = Nets.netG_deconder_gamma(sample_add_pn_ex, self.output_channel, reuse=True)
            tf.summary.histogram('outputencoder', output_en)
            tf.summary.histogram('inputdecoder', sample_add_pn)
            tf.summary.histogram('inputdecoder_exchange',sample_add_pn_ex)
            tf.summary.histogram('outputdecoder', output_de)
            tf.summary.histogram('inputrealdata', batch_data)
            tf.summary.histogram('inputsyndata', output_de)
            tf.summary.histogram('inputsyndata', output_de_rp)

        with tf.name_scope('discriminator_total'):
            pidr_softmax, pidrlogits,pidcontent\
                = \
                Nets.Custom_netD_discriminator_idloss(batch_data, class_nums=self.class_nums,usecontant=True)
            ppsr_logits, prcontent \
                = \
                Nets.Custom_netD_discriminator_psloss(batch_data, posenum=self.pose_c)
            _,reallogits=Nets.Custom_netD_discriminator_adloss(batch_data)

            pidf1_softmax, pidf1logits,pidf1content \
                = \
                Nets.Custom_netD_discriminator_idloss(output_de, class_nums=self.class_nums,usecontant=True,
                                                      reuse=True)
            ppsf1_logits, pf1content \
                = \
                Nets.Custom_netD_discriminator_psloss(output_de, posenum=self.pose_c, reuse=True)
            _, fake1logits = Nets.Custom_netD_discriminator_adloss(output_de,reuse=True)


            pidf2_softmax, pidf2logits,pidf2content \
                = \
                Nets.Custom_netD_discriminator_idloss(output_de_rp,class_nums=self.class_nums, usecontant=True,
                                                      reuse=True)
            ppsf2_logits, pf2content \
                = \
                Nets.Custom_netD_discriminator_psloss(output_de_rp, posenum=self.pose_c, reuse=True)
            _, fake2logits = Nets.Custom_netD_discriminator_adloss(output_de_rp,reuse=True)

            tf.summary.histogram('real_id', pidrlogits)
            tf.summary.histogram('fake1_id', pidf1logits)
            tf.summary.histogram('fake2_id', pidf2logits)
            tf.summary.histogram('real_pose', ppsr_logits)
            tf.summary.histogram('fake1_pose', ppsf1_logits)
            tf.summary.histogram('fake2_pose', ppsf2_logits)

        return pidr_softmax, pidrlogits, ppsr_logits, \
               pidf1_softmax, pidf1logits, ppsf1_logits, \
               pidf2_softmax, pidf2logits, ppsf2_logits, \
               output_de, output_en, output_de_rp,\
               pidcontent,pidf1content,pidf2content,\
                reallogits,fake1logits,fake2logits

    def validation_drgan(self,batch_data,data_size,noise=None,pose=None,reuse=True):
        sample_outen=Nets.netG_encoder(batch_data,reuse=reuse)
        valid_add_z = tf.concat([sample_outen, noise], 3)
        pose=tf.expand_dims(pose,1)
        pose=tf.expand_dims(pose,1)
        valid_add_zp = tf.concat([valid_add_z,pose],3)
        # ------------
        sample_valid, _ = Nets.netG_deconder(valid_add_zp, data_size, self.output_channel,reuse=reuse)

        return sample_valid

    def predict_multi_gpus(self,predict_function,batch_data,params=None):
        print 'INFO:using %d GPUs for traning'%len(self.gpus_list)
        image_splits = tf.split(batch_data)
        params_splits=None
        if params:
            params_splits = tf.split(batch_data)
        tower_predict_r = []
        tower_predict_f = []
        tower_predict_f = []
        tower_sample = []
        for i in self.gpus_list:
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('tower_%d'%i):
                    predict_r,predict_f,sample,_=predict_function(image_splits[i])
                    tf.get_variable_scope().reuse_variable()
                    tower_predict_r.append(predict_r)
                    tower_predict_f.append(predict_f)
                    tower_sample.append(sample)
        predict_data = tf.concat(tower_predict_r,0)
        predict_fake = tf.concat(tower_predict_f,0)
        sample = tf.concat(tower_sample,0)

        return predict_data,predict_fake,sample,

    def multi_predict_drgan_multipie(self,input_data,noise,pose):
        input_data_split = tf.split(input_data,self.gpus_count,0)
        noise_split = tf.split(noise,self.gpus_count,0)
        pose_split = tf.split(pose,self.gpus_count,0)
        tower_pidr_softmax=[]
        tower_pidrlogits=[]
        tower_ppsr_logits=[]
        tower_pidf1_softmax=[]
        tower_pidf1logits=[]
        tower_ppsf1_logits=[]
        tower_pidf2_softmax=[]
        tower_pidf2logits=[]
        tower_ppsf2_logits=[]
        tower_output_de=[]
        tower_output_en=[]
        tower_output_de_rp=[]
        tower_pidrcontent=[]
        tower_pidf1content=[]
        tower_pidf2content=[]
        tower_reallogits=[]
        tower_fake1logits=[]
        tower_fake2logits=[]
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self.gpus_count):
                with tf.device('/gpu:%d' %i):
                    with tf.name_scope('tower_%d' %i):
                        pidr_softmax_sub, pidrlogits_sub, ppsr_logits_sub, \
                        pidf1_softmax_sub, pidf1logits_sub, ppsf1_logits_sub, \
                        pidf2_softmax_sub, pidf2logits_sub, ppsf2_logits_sub, \
                        output_de_sub, output_en_sub, output_de_rp_sub, \
                        pidrcontent_sub, pidf1content_sub, pidf2content_sub ,\
                        reallogits, fake1logits, fake2logits \
                                    = self.predict_drgan_multipie(input_data_split[i], noise=noise_split[i], pose=pose_split[i])
                        tf.get_variable_scope().reuse_variables()
                        # print 'pidr_softmax',pidr_softmax_sub
                        # print 'tower',tower_pidr_softmax
                        #real data
                        tower_pidr_softmax.append(pidr_softmax_sub)
                        tower_pidrlogits.append(pidrlogits_sub)
                        tower_ppsr_logits.append(ppsr_logits_sub)
                        #fake data1
                        tower_pidf1_softmax.append(pidf1_softmax_sub)
                        tower_pidf1logits.append(pidf1logits_sub)
                        tower_ppsf1_logits.append(ppsf1_logits_sub)
                        #fake data2
                        tower_pidf2_softmax.append(pidf2_softmax_sub)
                        tower_pidf2logits.append(pidf2logits_sub)
                        tower_ppsf2_logits.append(ppsf2_logits_sub)
                        #output sample
                        tower_output_de.append(output_de_sub)
                        tower_output_en.append(output_en_sub)
                        tower_output_de_rp.append(output_de_rp_sub)
                        #contant
                        tower_pidrcontent.append(pidrcontent_sub)
                        tower_pidf1content.append(pidf1content_sub)
                        tower_pidf2content.append(pidf2content_sub)
                        #adlogits
                        tower_reallogits.append(reallogits)
                        tower_fake1logits.append(fake1logits)
                        tower_fake2logits.append(fake2logits)


        print tower_pidr_softmax
        pidr_softmax = tf.concat(tower_pidr_softmax,0,name='pidr_softmax')
        pidrlogits = tf.concat(tower_pidrlogits,0,name='pidrlogits')
        ppsr_logits = tf.concat(tower_ppsr_logits,0,name='ppsr_logits')

        pidf1_softmax = tf.concat(tower_pidf1_softmax,0,name='pidf1_softmax')
        pidf1logits = tf.concat(tower_pidf1logits,0,name='pidf1logits')
        ppsf1_logits = tf.concat(tower_ppsf1_logits,0,name='ppsf1_logits')

        pidf2_softmax = tf.concat(tower_pidf2_softmax,0,'pidf2_softmax')
        pidf2logits = tf.concat(tower_pidf2logits,0,name='pidf2logits')
        ppsf2_logits = tf.concat(tower_ppsf2_logits,0,name='ppsf2_logits')

        output_de = tf.concat(tower_output_de,0,name='output_de')
        output_en = tf.concat(tower_output_en,0,name='output_en')
        output_de_rp = tf.concat(tower_output_de_rp,0,name='output_de_rp')

        pidrcontent = tf.concat(tower_pidrcontent,0,'pidrcontent')
        pidf1content = tf.concat(tower_pidf1content,0,'pidf1content')
        pidf2content = tf.concat(tower_pidf2content,0,'pidf2content')

        reallogits = tf.concat(tower_reallogits,0,'reallogits')
        fake1logits = tf.concat(tower_fake1logits,0,'fake1logits')
        fake2logits = tf.concat(tower_fake2logits,0,'fake2logits')

        return      pidr_softmax, pidrlogits, ppsr_logits, \
                    pidf1_softmax, pidf1logits, ppsf1_logits, \
                    pidf2_softmax, pidf2logits, ppsf2_logits, \
                    output_de, output_en, output_de_rp, \
                    pidrcontent, pidf1content, pidf2content, \
                    reallogits, fake1logits, fake2logits

    def loss_gan_multipie(self, input_data, noise=None, pose=None,label=None,fakelabel=None):
        if self.multigpus:
            pidr_softmax, pidrlogits, ppsr_logits, \
            pidf1_softmax, pidf1logits, ppsf1_logits, \
            pidf2_softmax, pidf2logits, ppsf2_logits, \
            output_de, output_en, output_de_rp, \
            pidrcontent, pidf1content, pidf2content, \
            reallogits, fake1logits, fake2logits\
                = self.multi_predict_drgan_multipie(input_data, noise=noise, pose=pose)
        else:
            pidr_softmax, pidrlogits, ppsr_logits, \
            pidf1_softmax, pidf1logits, ppsf1_logits, \
            pidf2_softmax, pidf2logits, ppsf2_logits, \
            output_de, output_en, output_de_rp, \
            pidrcontent, pidf1content, pidf2content, \
            reallogits, fake1logits, fake2logits\
                = self.predict_drgan_multipie(input_data, noise=noise, pose=pose)

        with tf.name_scope('Discriminator_loss'):
            id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=pidrlogits))
            id_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel, logits=pidf1logits))
            id_loss_fake_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel, logits=pidf2logits))
            #pose loss
            ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=pose, logits=ppsr_logits))
            #adversarial
            ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(reallogits),logits=reallogits
            ))
            ad_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake1logits),logits=fake1logits
            ))
            ad_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake2logits),logits=fake2logits
            ))

        with tf.name_scope('Generator_loss'):
            id_loss_syn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=pidf1logits))
            id_loss_syn_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=pidf2logits))

            #ps loss
            input_pose_ex = tf.concat(tf.split(pose, 2, axis=0)[::-1], axis=0)
            ps_loss_syn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=pose, logits=ppsf1_logits))
            ps_loss_syn_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=input_pose_ex, logits=ppsf2_logits))

            #identity preserved loss

            id_preserve_f1=tf.reduce_mean(tf.squared_difference(pidrcontent,pidf1content))
            pidrcontent_ex = tf.concat(tf.split(pidrcontent, 2, axis=0)[::-1], axis=0)
            id_preserve_f2=tf.reduce_mean(tf.squared_difference(pidrcontent_ex,pidf2content))

            #adversarial
            ad_loss_syn1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake1logits),logits=fake1logits
            ))
            ad_loss_syn2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake2logits),logits=fake2logits
            ))
        with tf.name_scope('loss_discriminator'):
            summary_train_lossD_id = tf.summary.scalar('id_loss_d', id_loss_real)
            summary_train_lossD_ps = tf.summary.scalar('ps_loss_d', ps_loss_real)
            summary_train_lossD_idf1 = tf.summary.scalar('id_loss_fake', id_loss_fake)
            summary_train_lossD_idf2 = tf.summary.scalar('id_loss_fake_ex', id_loss_fake_ex)
            summary_train_lossD_ps = tf.summary.scalar('ad_loss_real', ad_loss_real)
            summary_train_lossD_idf1 = tf.summary.scalar('ad_loss_fake1', ad_loss_fake1)
            summary_train_lossD_idf2 = tf.summary.scalar('ad_loss_fake2', ad_loss_fake2)
        with tf.name_scope('loss_generator'):
            summary_train_lossG_id = tf.summary.scalar('id_loss_syn1', id_loss_syn)
            summary_train_lossG_id2 = tf.summary.scalar('id_loss_syn2', id_loss_syn_ex)
            summary_train_lossG_ps = tf.summary.scalar('ps_loss_syn1', ps_loss_syn)
            summary_train_lossG_ps2 = tf.summary.scalar('ps_loss_syn2', ps_loss_syn_ex)
            summary_train_lossG_ps = tf.summary.scalar('ad_loss_syn1', ad_loss_syn1)
            summary_train_lossG_ps2 = tf.summary.scalar('ad_loss_syn2', ad_loss_syn2)

        # with tf.name_scope('image_synthesis'):
        summary_train_image_batch = tf.summary.image('image0/input', tf.expand_dims(input_data[0], 0))
        summary_train_image_sample = tf.summary.image('image0/decoder', tf.expand_dims(output_de[0], 0))
        summary_train_image_sample = tf.summary.image('image0/decoder_re', tf.expand_dims(output_de_rp[0], 0))

        summary_train_image_batch = tf.summary.image('image1/input', tf.expand_dims(input_data[1], 0))
        summary_train_image_sample = tf.summary.image('image1/decoder', tf.expand_dims(output_de[1], 0))
        summary_train_image_sample = tf.summary.image('image1/decoder_re', tf.expand_dims(output_de_rp[1], 0))

        summary_train_image_batch = tf.summary.image('image2/input', tf.expand_dims(input_data[2], 0))
        summary_train_image_sample = tf.summary.image('image2/decoder', tf.expand_dims(output_de[2], 0))
        summary_train_image_sample = tf.summary.image('image2/decoder_re', tf.expand_dims(output_de_rp[2], 0))
        return id_loss_real, id_loss_fake, id_loss_fake_ex, ps_loss_real, \
                id_loss_syn, ps_loss_syn, \
                id_loss_syn_ex , ps_loss_syn_ex,\
                pidr_softmax,pidf1_softmax,pidf2_softmax, \
                id_preserve_f1,id_preserve_f2,\
                ad_loss_real,ad_loss_fake1,ad_loss_fake2,ad_loss_syn1,ad_loss_syn2,\
               output_de, output_de_rp
    def loss_drgan_3d6w(self,input_data,input_pose,label,fakelabel,index):
        # get predict data
        softmax_id_real, idlogits_real, \
        pslogits_real, content_feature_real, \
        softmax_id_fake, idlogits_fake, \
        pslogits_fake, content_feature_fake, \
        softmax_id_fake_ex, idlogits_fake_ex, \
        pslogits_fake_ex, content_feature_fake_ex, \
        output_de\
            = self.predict_drgan_3d6w(input_data,index, pose=input_pose)
        count_n=tf.cast(self.batch_size,tf.float32)
        with tf.name_scope('Discriminator_loss'):
            id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=idlogits_real))
            id_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel, logits=idlogits_fake))
            id_loss_fake_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel,logits=idlogits_fake_ex))

            ps_loss_real = tf.reduce_mean(tf.squared_difference(input_pose, pslogits_real))*10
        # with tf.name_scope('perceptual_loss'):
        #     feature_shape = tf.shape(content_feature_real)
        #     feature_size = tf.cast(feature_shape[1] * feature_shape[2] * feature_shape[3], dtype=tf.float32)
        #     feature_reconstruction_loss = tf.reduce_sum(
        #         tf.squared_difference(content_feature_real, content_feature_fake)) / feature_size

        with tf.name_scope('Generator_loss'):
            id_loss_syn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=idlogits_fake))
            id_loss_syn_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label,logits=idlogits_fake_ex))
            ps_loss_syn = tf.reduce_mean(tf.squared_difference(input_pose, pslogits_fake))*10

            input_pose_sp=tf.split(input_pose,2,axis=0)
            input_pose_ex=tf.concat(input_pose_sp[::-1],axis=0)
            ps_loss_syn_ex = tf.reduce_mean(tf.squared_difference(input_pose_ex,pslogits_fake_ex))

            # ex_loss_syn = tf.reduce_mean(tf.squared_difference(idlogits_fake,idlogits_fake_ex))

            #拉近他们的相似性



        with tf.name_scope('compute_loss'):
            # Dloss=self.lr_gamma*id_loss_real+self.lr_alpha*(id_loss_fake+id_loss_fake_ex)+ps_loss_real
            # Gloss_warmup=self.lr_gamma*id_loss_syn+self.lr_alpha*(id_loss_syn+id_loss_syn_ex)*self.warmup+self.lr_beta*(ps_loss_syn+ps_loss_syn_ex)*self.warmup
            # Gloss=self.lr_gamma*id_loss_syn+self.lr_alpha*(id_loss_syn+id_loss_syn_ex)+self.lr_beta*(ps_loss_syn+ps_loss_syn_ex)
        # return Dloss,Gloss,Gloss_warmup,output_de,output_de_exchange

            Dloss_warmup =id_loss_real + id_loss_fake + ps_loss_real
            Gloss_warmup=id_loss_syn + ps_loss_syn
            ps_loss_syn=self.lr_beta * ps_loss_syn
            ps_loss_syn_ex=self.lr_beta*ps_loss_syn_ex

            id_loss_fake=self.lr_alpha * id_loss_fake
            id_loss_fake_ex=self.lr_alpha*id_loss_fake_ex

            id_loss_real=self.lr_gamma * id_loss_real

            id_loss_syn = self.lr_delta * id_loss_syn
            id_loss_syn_ex=self.lr_delta*id_loss_syn_ex

            Dloss = id_loss_real + id_loss_fake + ps_loss_real +id_loss_fake_ex
            Gloss = id_loss_syn + ps_loss_syn +id_loss_syn_ex +ps_loss_syn_ex
        summary_train_lossD_id = tf.summary.scalar('lossD/id_loss_d_real', id_loss_real)
        summary_train_lossD_ps = tf.summary.scalar('lossD/ps_loss_d', ps_loss_real)
        summary_train_lossD_id = tf.summary.scalar('lossD/id_loss_d_fake', id_loss_fake)
        # summary_train_lossD_ps = tf.summary.scalar('lossD/id_loss_d_fake_ex', id_loss_fake_ex)

        summary_train_lossG_id = tf.summary.scalar('lossG/id_loss_g', id_loss_syn)
        summary_train_lossG_ps = tf.summary.scalar('lossG/ps_loss_g', ps_loss_syn)
        # summary_train_lossG_id = tf.summary.scalar('lossG/id_loss_g_ex', id_loss_syn_ex)
        # summary_train_lossG_ps = tf.summary.scalar('lossG/ps_loss_g_ex', ps_loss_syn_ex)
        return Dloss, Gloss,Dloss_warmup,Gloss_warmup,output_de

    def loss_drgan_caspeal(self,labels,fakelabels,poselabels,randomposelabels,idrlogits,idf1logits,idf2logits,psrlogits,psf1logits,psf2logits):
        with tf.name_scope('Discriminator_loss'):
            id_loss_fake1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=fakelabels,logits=idf1logits))
            id_loss_fake2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=fakelabels,logits=idf2logits))
            id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,logits=idrlogits))
            # id_loss_fake = tf.reduce_mean()
            ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=poselabels,logits=psrlogits))
        with tf.name_scope('Generator_loss'):
        # Dloss = ad_loss_fake+ad_loss_real+id_loss_real+ps_loss_real
        #     ad_loss_syn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.ones_like(predict_fake),logits=predict_fake_logits))
            id_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,logits=idf1logits))
            id_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=idf2logits))
            ps_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=poselabels,logits=psf1logits))
            ps_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=randomposelabels, logits=psf2logits))
        # Gloss = ad_loss_syn+id_loss_syn+ps_loss_syn
        return id_loss_real,id_loss_fake1,id_loss_fake2,ps_loss_real,\
               id_loss_syn1,ps_loss_syn1,\
                id_loss_syn2,ps_loss_syn2

    def get_train_op(self,lossD,lossG,global_step,lr,beta1):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1)
        # netDparam = slim.get_variables("discriminator")
        # grads_and_var_D = optimizer.compute_gradients(lossD,netDparam,colocate_gradients_with_ops = True)
        # grads_D,vars_D = zip(*grads_and_var_D)
        # grads_D,_ =tf.clip_by_global_norm(grads_D,0.1)
        # train_D_op = optimizer.apply_gradients(zip(grads_D,vars_D),global_step)
        # print self.varsD
        # print netDparam
#         assert(self.varsD==netDparam)
        train_D_op = optimizer.minimize(lossD, var_list=self.varsD)
        with tf.control_dependencies([train_D_op]):
            # netG_param = slim.get_variables("generator")
            # grads_and_var_G = optimizer.compute_gradients(lossG,netG_param,colocate_gradients_with_ops = True)
            # grads_G ,var_G = zip(*grads_and_var_G)
            # grads_G , _ = tf.clip_by_global_norm(grads_G,0.1)
            # train_G_op = optimizer.apply_gradients(zip(grads_G,var_G),global_step)
            # assert(self.varsG==netG_param)
            train_G_op = optimizer.minimize(lossG, var_list=self.varsG)
        return train_G_op

    def train(self,mode):
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpus_list
        # tf.set_random_seed(20)
        with tf.Graph().as_default():
            global_step = slim.get_or_create_global_step()
            if mode == 0:#handle3D ALFW,AFW
                input_data = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.input_size, self.input_size, self.input_channel],
                                            name='input_images')  # image
                input_label = tf.placeholder(dtype=tf.int32, shape=[None], name='input_labels')  # label
                input_pose = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='input_poses')  # pose
                input_pose_re = tf.reshape(input_pose, [-1, 1, 1, 3])
                index = tf.placeholder(tf.int32, None, name='input_nums')
                # mk onehot labels
                labels = slim.one_hot_encoding(input_label, self.data_loader_train.class_nums + 1)
                fakelabel = tf.zeros_like(input_label) + self.data_loader_train.class_nums
                fakelabels = slim.one_hot_encoding(fakelabel, self.data_loader_train.class_nums + 1)
                Dloss, Gloss, Dloss_warmup,Gloss_warmup,\
                sample \
                    = self.loss_drgan_3d6w(input_data, input_pose_re, labels, fakelabels,index)

                summary_train_lossD = tf.summary.scalar('loss/total_loss_d', Dloss)
                summary_train_lossG = tf.summary.scalar('loss/total_loss_g', Gloss)

                # summary_train_image_batch = tf.summary.image('image/input',tf.expand_dims(input_data[0],0))
                summary_train_image_batch = tf.summary.image('image0/input', tf.expand_dims(input_data[0], 0))
                summary_train_image_sample = tf.summary.image('image0/decoder', tf.expand_dims(sample[0], 0))

                summary_train_image_batch = tf.summary.image('image1/input', tf.expand_dims(input_data[1], 0))
                summary_train_image_sample = tf.summary.image('image1/decoder', tf.expand_dims(sample[1], 0))

                summary_train_image_batch = tf.summary.image('image2/input', tf.expand_dims(input_data[2], 0))
                summary_train_image_sample = tf.summary.image('image2/decoder', tf.expand_dims(sample[2], 0))
            elif mode==1:#caspeal
                input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
                input_label = tf.placeholder(dtype= tf.int32,shape=[None],name='input_labels')#label
                input_pose = tf.placeholder(dtype= tf.int32,shape=[None],name='input_poses')#pose
                index = tf.placeholder(tf.int32, None,name='input_nums')
                #mk onehot labels
                labels = slim.one_hot_encoding(input_label,self.data_loader_train.class_nums+1)
                input_false_labels = tf.placeholder(dtype=tf.int32,shape=[None],name='falselabel')
                false_labels = slim.one_hot_encoding(input_false_labels,self.data_loader_train.class_nums+1)
                poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
                #加入新的随机Pose

                random_input_pose = tf.placeholder(dtype=tf.int32,shape=[None],name='random_input_pose')
                # random_input_pose=tf.cast(random_p,tf.int32)
                random_poses = slim.one_hot_encoding(random_input_pose,self.pose_c)
                noise = tf.random_uniform(shape=(index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')

                #get predict data
                pidr_softmax, pidrlogits, ppsr_logits, \
                pidf1_softmax, pidf1logits, ppsf1_logits, \
                pidf2_softmax, pidf2logits, ppsf2_logits, \
                output_de, output_en, output_de_rp=self.predict_drgan_caspeal(input_data,noise=noise,pose=poses,random_pose=random_poses)
                #comput loss
                id_loss_real, id_loss_fake1, id_loss_fake2, ps_loss_real, \
                id_loss_syn1, ps_loss_syn1, \
                id_loss_syn2, ps_loss_syn2\
                    =self.loss_drgan_caspeal(labels,false_labels,poses,random_poses,pidrlogits,pidf1logits,pidf2logits,ppsr_logits,ppsf1_logits,ppsf2_logits)


                Dloss_warmup=id_loss_real+id_loss_fake1+ps_loss_real
                Gloss_warmup=id_loss_syn1+ps_loss_syn1

                Dloss = id_loss_real+self.lr_gamma*(id_loss_fake1+id_loss_fake2)+ps_loss_real
                Gloss = self.lr_alpha*(id_loss_syn1+id_loss_syn2)+self.lr_beta*(ps_loss_syn1+ps_loss_syn2)
                #test accuracy:
                with tf.name_scope('accurary'):
                                                   reshape_R=tf.reshape(pidr_softmax, [-1, self.data_loader_train.class_nums+1])
                                                   max_r=tf.argmax(reshape_R,1)
                                                   label_true=tf.argmax(labels,1)
                                                   correct_pred_r = tf.equal(max_r,label_true)
                                                   reshape_F=tf.reshape(pidf1_softmax, [-1, self.data_loader_train.class_nums+1])
                                                   max_f=tf.argmax(reshape_F, 1)
                                                   correct_pred_f = tf.equal(max_f,label_true)
                                                   reshape_F1 = tf.reshape(pidf2_softmax, [-1, self.data_loader_train.class_nums + 1])
                                                   max_f1 = tf.argmax(reshape_F1, 1)
                                                   correct_pred_f1 = tf.equal(max_f1, label_true)
                                                   accuracy_r = tf.reduce_mean(tf.cast(correct_pred_r,tf.float32))
                                                   accuracy_f = tf.reduce_mean(tf.cast(correct_pred_f,tf.float32))
                                                   accuracy_f1 = tf.reduce_mean(tf.cast(correct_pred_f1, tf.float32))

                summary_train_accracy_r = tf.summary.scalar('accuracy/real_data',accuracy_r)
                summary_train_accracy_f = tf.summary.scalar('accuracy/fake_data',accuracy_f)
                summary_train_accracy_f = tf.summary.scalar('accuracy/fake_data_random', accuracy_f1)

                summary_train_lossD = tf.summary.scalar('losstotal/total_loss_d',Dloss)
                summary_train_lossG = tf.summary.scalar('losstotal/total_loss_g',Gloss)

                # summary_train_lossD_ad_r = tf.summary.scalar('loss/ad_loss_d_real',ad_loss_real)
                # summary_train_lossD_ad_f = tf.summary.scalar('loss/ad_loss_d_fake',ad_loss_fake)
                summary_train_lossD_id = tf.summary.scalar('lossD/id_loss_d',id_loss_real)
                summary_train_lossD_ps = tf.summary.scalar('lossD/ps_loss_d',ps_loss_real)
                summary_train_lossD_idf1 = tf.summary.scalar('lossD/id_loss_fake1',id_loss_fake1)
                summary_train_lossD_idf2 = tf.summary.scalar('lossD/id_loss_fake2',id_loss_fake2)

                summary_train_lossG_id = tf.summary.scalar('lossG/id_loss_syn1',id_loss_syn1)
                summary_train_lossG_id2 = tf.summary.scalar('lossG/id_loss_syn2',id_loss_syn2)
                summary_train_lossG_ps = tf.summary.scalar('lossG/ps_loss_syn1',ps_loss_syn1)
                summary_train_lossG_ps2 = tf.summary.scalar('lossG/ps_loss_syn2',ps_loss_syn2)

                summary_train_image_batch = tf.summary.image('image0/input',tf.expand_dims(input_data[0],0))
                summary_train_image_sample = tf.summary.image('image0/decoder',tf.expand_dims(output_de[0],0))
                summary_train_image_sample = tf.summary.image('image0/decoder_re',tf.expand_dims(output_de_rp[0],0))

                summary_train_image_batch = tf.summary.image('image1/input', tf.expand_dims(input_data[1], 0))
                summary_train_image_sample = tf.summary.image('image1/decoder', tf.expand_dims(output_de[1], 0))
                summary_train_image_sample = tf.summary.image('image1/decoder_re', tf.expand_dims(output_de_rp[1], 0))

                summary_train_image_batch = tf.summary.image('image2/input', tf.expand_dims(input_data[2], 0))
                summary_train_image_sample = tf.summary.image('image2/decoder', tf.expand_dims(output_de[2], 0))
                summary_train_image_sample = tf.summary.image('image2/decoder_re', tf.expand_dims(output_de_rp[2], 0))
            else:# 单Label + dual data，取消random_data
                input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
                input_label = tf.placeholder(dtype= tf.int32,shape=[None],name='input_labels')#label
                input_pose = tf.placeholder(dtype= tf.int32,shape=[None],name='input_poses')#pose
                index = tf.placeholder(tf.int32, None,name='input_nums')
                #mk onehot labels
                labels = slim.one_hot_encoding(input_label,self.class_nums)
                input_false_labels = tf.placeholder(dtype=tf.int32,shape=[None],name='falselabel')
                fake_labels = slim.one_hot_encoding(input_false_labels,self.class_nums)#现阶段不用
                poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
                #加入新的随机Pose
                noise = tf.random_uniform(shape=(index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
                #comput loss
                id_loss_real, id_loss_fake1, id_loss_fake2, ps_loss_real, \
                id_loss_syn1, ps_loss_syn1, \
                id_loss_syn2, ps_loss_syn2, \
                pidr_softmax, pidf1_softmax, pidf2_softmax, \
                id_preserve_f1, id_preserve_f2, \
                ad_loss_real, ad_loss_fake1, ad_loss_fake2, ad_loss_syn1, ad_loss_syn2, \
                output_de_f1, output_de_rp_f2\
                    =self.loss_gan_multipie(input_data,noise=noise,pose=poses,label=labels,fakelabel=fake_labels)

                if self.losstype==1: #use perceptual loss
                    # Dloss_warmup=id_loss_real+ps_loss_real+ad_loss_real+ad_loss_fake1
                    # Gloss_warmup=ps_loss_syn1+id_preserve_f1+ad_loss_syn1
                    Dloss_warmup=id_loss_real+ps_loss_real+ad_loss_real+ad_loss_fake1
                    Gloss_warmup=ps_loss_syn1+ad_loss_syn1+0.02*id_loss_syn1
                    Dloss = id_loss_real+ps_loss_real+ad_loss_real+self.lr_gamma*(ad_loss_fake1+ad_loss_fake2)
                    Gloss = self.lr_beta*(ps_loss_syn1+ps_loss_syn2)+self.lr_delta*(id_loss_syn1+id_loss_syn2)+self.lr_gamma*(ad_loss_syn1+id_loss_syn2)
                elif self.losstype==2:#not use the perceptual loss
                    Dloss_warmup=id_loss_real+id_loss_fake1+ps_loss_real
                    Gloss_warmup=id_loss_syn1+ps_loss_syn1

                    Dloss = id_loss_real+self.lr_gamma*(id_loss_fake1+id_loss_fake2)+ps_loss_real
                    Gloss = self.lr_alpha*(id_loss_syn1+id_loss_syn2)+self.lr_beta*(ps_loss_syn1+ps_loss_syn2)
                #test accuracy:
                with tf.name_scope('accuracy'):
                    reshape_R=tf.reshape(pidr_softmax, [-1, self.class_nums])
                    max_r=tf.argmax(reshape_R,1)
                    label_true=tf.argmax(labels,1)
                    correct_pred_r = tf.equal(max_r,label_true)
                    reshape_F=tf.reshape(pidf1_softmax, [-1, self.class_nums])
                    max_f=tf.argmax(reshape_F, 1)
                    correct_pred_f = tf.equal(max_f,label_true)
                    reshape_F1 = tf.reshape(pidf2_softmax, [-1, self.class_nums])
                    max_f1 = tf.argmax(reshape_F1, 1)
                    correct_pred_f1 = tf.equal(max_f1, label_true)
                    accuracy_r = tf.reduce_mean(tf.cast(correct_pred_r,tf.float32))
                    accuracy_f = tf.reduce_mean(tf.cast(correct_pred_f,tf.float32))
                    accuracy_f1 = tf.reduce_mean(tf.cast(correct_pred_f1, tf.float32))

                    summary_train_accracy_r = tf.summary.scalar('real_data',accuracy_r)
                    summary_train_accracy_f = tf.summary.scalar('fake_data',accuracy_f)
                    summary_train_accracy_f = tf.summary.scalar('fake_data_exchange', accuracy_f1)

                summary_train_lossD = tf.summary.scalar('losstotal/total_loss_d',Dloss)
                summary_train_lossG = tf.summary.scalar('losstotal/total_loss_g',Gloss)

            summary_train = tf.summary.merge_all()
            train_vars = tf.trainable_variables()
            self.varsG = [var for var in train_vars if 'generator' in var.name]
            self.varsD = [var for var in train_vars if 'discriminator' in var.name]

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True

            with tf.Session(config=config) as sess:

                # for learning_rate in [self.learning_rate]:
                    # embedding = tf.Variable(tf.zeros([7,96,96,3]),name='test_embedding')
                    # assignment = embedding.assign(input_data)
                    # config2=projector.ProjectorConfig()
                    # embedding_config=config2.embeddings.add()
                    # embedding_config.tensor_name=embedding.name
                    # embedding_config.sprite.image=os.path.join(self.log_dir,'sprite.png')
                    # embedding_config.sprite.single_image_dim.extend([96,96,3])
                curr_interval=0
                with tf.name_scope('train_optimizer'):
                    g_lr=tf.train.exponential_decay(self.g_lr,global_step=global_step,decay_steps=self.batch_idxs*20,decay_rate=0.9)
                    optimizer_d = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=self.beta1,
                                                       name='optimizer')
                    optimizer_g = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=self.beta1,
                                                       name='optimizer')
                    # batch norm,when trainingm the moving_mean and moving_variance need to be updated!!!!!!
                    #Gather batch normalization update operations
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):

                        train_op_d_warmup=optimizer_d.minimize(Dloss_warmup,var_list=self.varsD,colocate_gradients_with_ops=True,global_step=global_step)
                        train_op_g_warmup=optimizer_g.minimize(Gloss_warmup,var_list=self.varsG,colocate_gradients_with_ops=True)
                        train_op_d = optimizer_d.minimize(Dloss, var_list=self.varsD,colocate_gradients_with_ops=True,global_step=global_step)
                        train_op_g = optimizer_g.minimize(Gloss, var_list=self.varsG,colocate_gradients_with_ops=True)
                    try:
                        tf.global_variables_initializer().run()
                    except:
                        tf.initialize_all_variables().run()
                    saver = tf.train.Saver(max_to_keep=10)
                    step = 0
                    if self.restored:
                        saver.restore(sess, self.check_point)
                        step = int(next(re.finditer("(\d+)(?!.*\d)", self.check_point)).group(0))
                    start_time = time.time()

                    self.data_loader_train.Dual_enqueueStart()#开启训练对别
                    self.data_loader_valid.Dual_enqueueStart()#开启测试队列
                    # self.data_loader_train.enqueueStart()

                    hyperparameter=self.savename+'drgan'
                    summary_write = tf.summary.FileWriter(self.summarypath+'/'+self.version, sess.graph)
                    # projector.visualize_embeddings(summary_write,config2)
                    sample_data=None
                    sample_data_ex=None
                    loss_d=0
                    loss_g=0
                    train_summary=None
                    acc_r=None
                    acc_f=None
                    acc_f1=None
                    random_p=None
                    # global_count_int=0
                    for epoch_n in xrange(self.epoch):

                        # coord = tf.train.Coordinator()
                        # threads = tf.train.start_queue_runners(coord=coord,sess=sess)
                        for interval_i in trange(self.batch_idxs):
                            if mode==0:
                                if epoch_n<5 and self.warmup:
                                    batch_image, batch_label = self.data_loader_train.read_data_batch()
                                    batch_label = np.transpose(batch_label, [1, 0])
                                    # optimizer D
                                    _ = sess.run(train_op_d_warmup,
                                                 feed_dict={input_data: batch_image,
                                                            input_label: batch_label[:, 0],
                                                            input_pose: batch_label[:, 1:4] / 90.0,
                                                            index: batch_image.shape[0]})
                                    # #optimizer G
                                    sample_, loss_d, loss_g, train_summary = None, None, None, None
                                    for _ in xrange(self.Gloop):
                                        sample_data, loss_d, loss_g, train_summary, _ = sess.run(
                                            [sample, Dloss, Gloss, summary_train, train_op_g_warmup],
                                            feed_dict={input_data: batch_image,
                                                       input_label: batch_label[:, 0],
                                                       input_pose: batch_label[:, 1:4] / 90.0,
                                                       index: batch_image.shape[0]})
                                else:
                                    batch_image, batch_label = self.data_loader_train.read_data_batch()
                                    batch_label = np.transpose(batch_label, [1, 0])
                                    # optimizer D
                                    _ = sess.run(train_op_d,
                                                 feed_dict={input_data: batch_image,
                                                            input_label: batch_label[:, 0],
                                                            input_pose: batch_label[:, 1:4]/90.0,
                                                            index: batch_image.shape[0]})
                                    # #optimizer G
                                    sample_, loss_d, loss_g, train_summary = None, None, None, None
                                    for _ in xrange(self.Gloop):
                                        sample_data, loss_d, loss_g, train_summary,_ = sess.run(
                                            [sample, Dloss, Gloss, summary_train,train_op_g],
                                            feed_dict={input_data: batch_image,
                                                       input_label: batch_label[:, 0],
                                                       input_pose: batch_label[:, 1:4]/90.0,
                                                       index: batch_image.shape[0]})
                            elif mode==1:
                                random_p = np.random.randint(0, self.pose_c, self.batch_size)
                                #read data labels
                                batch_image,batch_label=self.data_loader_train.read_data_batch()

                                #optimizer D
                                flabels=[self.data_loader_train.class_nums] * batch_image.shape[0]
                                if epoch_n < 5:
                                    _=sess.run(train_op_d_warmup,
                                            feed_dict={input_data:batch_image,
                                                       input_label:batch_label[0],
                                                       input_pose:batch_label[1],
                                                       input_false_labels:flabels,
                                                       index: batch_image.shape[0],
                                                       random_input_pose: random_p})
                                    for _ in xrange(4):
                                        random_p = np.random.randint(0, self.pose_c, self.batch_size)
                                        _, sample_data, loss_d, loss_g, train_summary, acc_r, acc_f, acc_f1 = sess.run(
                                            [train_op_g_warmup, output_de, Dloss_warmup, Gloss_warmup, summary_train,
                                             accuracy_r, accuracy_f, accuracy_f1],
                                        feed_dict={input_data: batch_image,
                                                   input_label: batch_label[0],
                                                   input_pose: batch_label[1],
                                                   input_false_labels:flabels,
                                                   index: batch_image.shape[0],
                                                   random_input_pose: random_p})
                                else:

                                    _=sess.run(train_op_d,
                                            feed_dict={input_data:batch_image,
                                                       input_label:batch_label[0],
                                                       input_pose:batch_label[1],
                                                       input_false_labels:flabels,
                                                       index: batch_image.shape[0],
                                                       random_input_pose:random_p})
                                    for _ in xrange(self.Gloop):
                                        random_p = np.random.randint(0, self.pose_c, self.batch_size)
                                        _, sample_data, loss_d, loss_g, train_summary, acc_r, acc_f, acc_f1 = sess.run(
                                            [train_op_g, output_de, Dloss, Gloss, summary_train,
                                             accuracy_r, accuracy_f, accuracy_f1],
                                        feed_dict={input_data: batch_image,
                                                   input_label: batch_label[0],
                                                   input_pose: batch_label[1],
                                                   input_false_labels:flabels,
                                                   index: batch_image.shape[0],
                                                   random_input_pose:random_p})
                                    print 'INFO : acc_r:%0.4f,acc_f:%0.4f,acc_f1:%0.4f'%(acc_r,acc_f,acc_f1)
                            else:
                                batch_image=np.zeros([self.batch_size*self.gpus_count,self.input_size,self.input_size,self.input_channel],np.float32)
                                batch_label=np.zeros([self.data_loader_train.labels_nums,self.batch_size*self.gpus_count],np.float32)
                                for b_i in xrange(self.gpus_count):
                                    batch_image[b_i*self.batch_size:(b_i+1)*self.batch_size,:,:,:],batch_label[:,b_i*self.batch_size:(b_i+1)*self.batch_size]=self.data_loader_train.read_data_batch()
                                #optimizer D
                                flabels=[self.data_loader_train.class_nums] * batch_image.shape[0]
                                if step < int(5*self.batch_idxs) and self.warmup:
                                    _=sess.run(train_op_d_warmup,
                                            feed_dict={input_data:batch_image,
                                                       input_label:batch_label[0],
                                                       input_pose:batch_label[1],
                                                       input_false_labels:flabels,
                                                       index: batch_image.shape[0]})
                                    for _ in xrange(self.Gloop):
                                        _, sample_data,sample_data_ex, loss_d, loss_g, train_summary, acc_r, acc_f, acc_f1 = sess.run(
                                            [train_op_g_warmup, output_de_f1,output_de_rp_f2, Dloss_warmup, Gloss_warmup, summary_train,
                                             accuracy_r, accuracy_f, accuracy_f1],
                                        feed_dict={input_data: batch_image,
                                                   input_label: batch_label[0],
                                                   input_pose: batch_label[1],
                                                   input_false_labels:flabels,
                                                   index: batch_image.shape[0]})
                                        print 'INFO : acc_r:%0.4f,acc_f:%0.4f,acc_f1:%0.4f' % (acc_r, acc_f, acc_f1)
                                        logging.info('INFO : acc_r:%0.4f,acc_f:%0.4f,acc_f1:%0.4f' % (acc_r, acc_f, acc_f1))
                                else:
                                    self.Gloop=2
                                    _=sess.run(train_op_d,
                                            feed_dict={input_data:batch_image,
                                                       input_label:batch_label[0],
                                                       input_pose:batch_label[1],
                                                       input_false_labels:flabels,
                                                       index: batch_image.shape[0]})
                                    for _ in xrange(self.Gloop):
                                        _, sample_data,sample_data_ex, loss_d, loss_g, train_summary, acc_r, acc_f, acc_f1 = sess.run(
                                            [train_op_g, output_de_f1,output_de_rp_f2, Dloss, Gloss, summary_train,
                                             accuracy_r, accuracy_f, accuracy_f1],
                                        feed_dict={input_data: batch_image,
                                                   input_label: batch_label[0],
                                                   input_pose: batch_label[1],
                                                   input_false_labels:flabels,
                                                   index: batch_image.shape[0]})
                                        print 'INFO : acc_r:%0.4f,acc_f:%0.4f,acc_f1:%0.4f'%(acc_r,acc_f,acc_f1)
                                        logging.info('INFO : acc_r:%0.4f,acc_f:%0.4f,acc_f1:%0.4f' % (acc_r, acc_f, acc_f1))
                            summary_write.add_summary(train_summary,global_step=step)

                            step+=1
                            print 'Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f, d_loss:%.4f, g_loss:%.4f'\
                            %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss_d,loss_g)
                            tmp_batch=None
                            if (curr_interval)%int(self.sample_interval*self.batch_idxs)==0:
                                if mode==2:
                                    utils.write_batch(self.result_path,0,sample_data,batch_image,epoch_n,interval_i,othersample=sample_data_ex)
                                else:
                                    utils.write_batch(self.result_path,0,sample_data,batch_image,epoch_n,interval_i)
                                sample_batch=np.zeros([self.test_batch_size*self.gpus_count,self.input_size,self.input_size,self.input_channel],np.float32)
                                label_batch = np.zeros([self.data_loader_train.labels_nums, self.test_batch_size * self.gpus_count], np.float32)
                                for s_i in xrange(self.gpus_count):
                                    sample_batch[s_i*self.pose_c:(s_i+1)*self.pose_c,:,:,:] ,label_batch[:,s_i*self.pose_c:(s_i+1)*self.pose_c] =self.data_loader_valid.oneperson_allpose(s_i)#得到一个人所有的图片
                                sample_count = sample_batch.shape[0]
                                # flabels = [self.data_loader_train.class_nums] * batch_image.shape[0]
                                if (curr_interval)%int(self.sample_pose_interval * self.batch_idxs)==0:

                                    if mode == 0:
                                        label_batch = [sample_count] * sample_count
                                        pose_batch = np.zeros_like(batch_label[:sample_count, 1:4], dtype=np.int)
                                        sample_data = sess.run(sample,
                                                               feed_dict={
                                                                   input_data: sample_batch,
                                                                   input_label: label_batch,
                                                                   input_pose: pose_batch,
                                                                   index: sample_count})
                                        utils.write_batch(self.result_path, 2, sample_data, sample_batch, epoch_n,
                                                          interval_i)

                                    elif mode==1:
                                        for idx in xrange(sample_count):

                                            label_batch = [sample_count] * self.pose_c#没用
                                            pose_batch = range(0, self.pose_c)
                                            tmp_batch = np.tile(sample_batch[idx],(self.pose_c,1,1,1)).\
                                                reshape(self.pose_c,sample_batch.shape[1],sample_batch.shape[2],sample_batch.shape[3])
                                            sample_data= sess.run(output_de,
                                                                               feed_dict={
                                                                                   input_data: tmp_batch,
                                                                                   input_label:label_batch,
                                                                                   input_pose: pose_batch,
                                                                                   index: self.pose_c,
                                                                                   random_input_pose: random_p})
                                            utils.write_batch(self.result_path,2,sample_data,tmp_batch,epoch_n,interval_i,sample_idx=idx)
                                    else:
                                        if (curr_interval) % int(self.test_interval * self.batch_idxs ) == 0:
                                            for idx in xrange(sample_count):#将数据集中的同一个人所有!!!角度！！！照片都跑一次
                                                tppn=self.test_batch_size
                                                label_batch_sub = [sample_count] * tppn#没用凑齐8 为了后面的split
                                                pose_batch = range(0, tppn)
                                                tmp_batch = np.tile(sample_batch[idx], (tppn, 1, 1, 1)). \
                                                    reshape(tppn, sample_batch.shape[1], sample_batch.shape[2],
                                                            sample_batch.shape[3])
                                                sample_data = sess.run(output_de_f1,
                                                                       feed_dict={
                                                                           input_data: tmp_batch,
                                                                           input_label: label_batch_sub,
                                                                           input_pose: pose_batch,
                                                                           input_false_labels:label_batch_sub,
                                                                           index: tppn})
                                                utils.write_batch(self.result_path, 2, sample_data, tmp_batch, epoch_n,
                                                                  interval_i, sample_idx=idx)
                                            if self.ifsave and curr_interval != 0 :
                                                saver.save(sess,
                                                           os.path.join(self.check_point_path, self.model_name),
                                                           global_step=step)
                                                print '*' * 20 + 'save model successed!!!!~~~~'
                                        # else:#将一个bath的数据互换Label跑一次

                                        # sample_8data=np.zeros([8,96,96,3],np.float32)
                                        # sample_8data[0:7]=sample_batch[0:7]
                                        # sample8_label_batch=np.zeros([8])
                                        idlabel_batch=[0]*sample_count
                                        sample_data, sample_data_ex= sess.run(
                                            [output_de_f1,output_de_rp_f2],
                                            feed_dict={input_data: sample_batch,
                                                       input_label: idlabel_batch,
                                                       input_pose: label_batch[1],
                                                       input_false_labels: idlabel_batch,
                                                       index: sample_count})#falselable无所谓了
                                        utils.write_batch(self.result_path, 1, sample_data, sample_batch, epoch_n,
                                                          interval_i,othersample=sample_data_ex,)

                            curr_interval+=1
                    # coord.request_stop()
                    # coord.join()





