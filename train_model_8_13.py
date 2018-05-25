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
from PIL import Image
# import scipy.misc
# from tensorflow.contrib.tensorboard.plugins import projector
import utils
import  logging
import Net_8_13 as Nets
from resnet_yd import resnet_yd
from tqdm import trange

TOTAL_VARIATION_SMOOTHING = 1e2

class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,lr_alpha=0.5,lr_beta=0.5,lr_gamma=1,version='0',
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,test_interval=1,
                 data_loader_train=None,data_loader_valid=None,epoch=10000,log_dir='logdir',d_learning_rate=0.0002,g_learning_rate=0.0002,beta1=0.5,delta=0.2,
                 test_batch_size=10,gpus_list=None,write_type=0,model_name='dr_gan',noise_z=50,pose_c=7,random_seed=25,sample_pose_interval=10,warmup_rate=1,
                 loss_type=0,Gloop=3,savepath='savename',imagepath='image',logfile='file.log',summary_dir='summary',warmup=True,discribe=''):
        self.warmup_rate=warmup_rate
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
        self.class_nums=self.data_loader_train.class_nums+1
        self.random_seed=random_seed
        self.test_slerp_count=3
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


    def predict_drgan_multipie(self, batch_data, noise=None, pose=None,pose_reverse=None,reuse=False,noise_reverse=None):
        #        with tf.variable_scope('drgan'):
        # with tf.name_scope('generator_encoder_decoder'):
        output_en = Nets.netG_encoder_gamma(batch_data,reuse=reuse)


        pose_reshape = tf.reshape(pose, [-1, 1, 1, self.pose_c])
        pose_float = tf.cast(pose_reshape, dtype=tf.float32)
        pose_add_noise=tf.concat([pose_float,noise],3)
        sample_add_pn = tf.concat([output_en,pose_add_noise ], 3)
        output_de = Nets.netG_deconder_gamma(sample_add_pn, self.output_channel,reuse=reuse)

        #pose reverse and concat pose和noise 分开
        pose_reverse_reshape = tf.reshape(pose_reverse, [-1, 1, 1, self.pose_c])
        pose_reverse_float=tf.cast(pose_reverse_reshape, dtype=tf.float32)
        # pose_add_noise_reverse=tf.concat([pose_reverse_float,noise],3)

        pose_add_noise_reverse = tf.concat([pose_reverse_float, noise], 3)
        #pose和 noise绑定
        sample_add_pn_ex = tf.concat([output_en, pose_add_noise_reverse], 3)
        output_de_rp = Nets.netG_deconder_gamma(sample_add_pn_ex, self.output_channel, reuse=True)

        # batch_data_grey=tf.image.rgb_to_grayscale(batch_data[:,:,:,::-1])
        # newsize=tf.constant([128,128],dtype=tf.int32)
        # batch_data_resize=tf.image.resize_bilinear(batch_data_grey,newsize)
        # batch_data_resize=tf.image.resize_nearest_neighbor()
        pidr_softmax, pidrlogits,pidcontent\
            = \
            resnet_yd(batch_data[:,:,:,::-1],reuse=reuse)

        ppsr_logits, prcontent \
            = \
            Nets.Custom_netD_discriminator_psloss(batch_data, posenum=self.pose_c,reuse=reuse)
        # _,reallogits=Nets.Custom_netD_discriminator_adloss(batch_data,reuse=reuse)
        _, reallogits = Nets.Custom_netD_discriminator_idloss(batch_data,class_nums=self.class_nums,reuse=reuse)

        #syn1
        # output_data_fake1_grey=tf.image.rgb_to_grayscale(output_de[:,:,:,::-1])
        # output_data_fake1_resize = tf.image.resize_bilinear(output_data_fake1_grey, newsize)
        # output_data_fake1_resize=tf.image.resize_nearest_neighbor(output_data_fake1_grey,newsize)
        pidf1_softmax, pidf1logits,pidf1content \
            = \
            resnet_yd(output_de[:,:,:,::-1],reuse=True)
        ppsf1_logits, pf1content \
            = \
            Nets.Custom_netD_discriminator_psloss(output_de, posenum=self.pose_c, reuse=True)
        # _, fake1logits = Nets.Custom_netD_discriminator_adloss(output_de,reuse=True)
        _, fake1logits = Nets.Custom_netD_discriminator_idloss(output_de, class_nums=self.class_nums, reuse=True)

        #syn2
        # output_data_fake2_grey=tf.image.rgb_to_grayscale(output_de_rp[:,:,:,::-1])
        # output_data_fake2_resize=tf.image.resize_bilinear(output_data_fake2_grey,newsize)
        pidf2_softmax, pidf2logits,pidf2content \
            = \
            resnet_yd(output_de_rp[:,:,:,::-1],reuse=True)
        ppsf2_logits, pf2content \
            = \
            Nets.Custom_netD_discriminator_psloss(output_de_rp, posenum=self.pose_c, reuse=True)
        # _, fake2logits = Nets.Custom_netD_discriminator_adloss(output_de_rp,reuse=True)
        _, fake2logits = Nets.Custom_netD_discriminator_idloss(output_de_rp, class_nums=self.class_nums, reuse=True)

        print 'idlogits',fake2logits.shape.as_list()

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


    def multi_predict_drgan_multipie(self,input_data,noise,pose,pose_reverse=None,noise_reverse=None):
        input_data_split = tf.split(input_data,self.gpus_count,0)
        noise_split = tf.split(noise,self.gpus_count,0)
        pose_split = tf.split(pose,self.gpus_count,0)
        pose_reverse_split = tf.split(pose_reverse,self.gpus_count,0)
        noise_reverse_split = tf.split(noise_reverse,self.gpus_count,0)
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
                                    = self.predict_drgan_multipie(input_data_split[i], noise=noise_split[i], pose=pose_split[i],pose_reverse=pose_reverse_split[i],noise_reverse=noise_reverse_split[i])
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

    def loss_gan_multipie(self, input_data, noise=None, pose=None,label=None,pose_reverse=None,fakelabel=None,noise_reverse=None):
        if self.multigpus:
            pidr_softmax, pidrlogits, ppsr_logits, \
            pidf1_softmax, pidf1logits, ppsf1_logits, \
            pidf2_softmax, pidf2logits, ppsf2_logits, \
            output_de, output_en, output_de_rp, \
            pidrcontent, pidf1content, pidf2content, \
            reallogits, fake1logits, fake2logits\
                = self.multi_predict_drgan_multipie(input_data, noise=noise, pose=pose,pose_reverse=pose_reverse,noise_reverse=noise_reverse)
        else:
            pidr_softmax, pidrlogits, ppsr_logits, \
            pidf1_softmax, pidf1logits, ppsf1_logits, \
            pidf2_softmax, pidf2logits, ppsf2_logits, \
            output_de, output_en, output_de_rp, \
            pidrcontent, pidf1content, pidf2content, \
            reallogits, fake1logits, fake2logits\
                = self.predict_drgan_multipie(input_data, noise=noise, pose=pose,pose_reverse=pose_reverse,noise_reverse=noise_reverse)
        self.outputdecoder_f1=output_de
        self.outputdecoder_f2=output_de_rp


        with tf.name_scope('D_loss'):
            self.id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=reallogits))
            self.id_loss_fake1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel, logits=fake1logits))
            self.id_loss_fake2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel, logits=fake2logits))
            # pose loss
            self.ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=pose, logits=ppsr_logits))
            #adversarial
            # self.ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.ones_like(reallogits),logits=reallogits
            # ))
            # self.ad_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.zeros_like(fake1logits),logits=fake1logits
            # ))
            # self.ad_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.zeros_like(fake2logits),logits=fake2logits
            # ))

        with tf.name_scope('G_loss'):
            #predict loss
            self.id_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=fake1logits))
            self.id_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=fake2logits))

            #ps loss
            input_pose_ex = tf.concat(tf.split(pose, 2, axis=0)[::-1], axis=0)
            self.ps_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=pose, logits=ppsf1_logits))
            self.ps_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=input_pose_ex, logits=ppsf2_logits))

            #identity preserved loss

            # id_preserve_f1=tf.reduce_mean(tf.squared_difference(pidrcontent,pidf1content))
            self.id_preserve_f1 = tf.reduce_mean(tf.abs(tf.subtract(pidrcontent, pidf1content)))
            pidrcontent_ex = tf.concat(tf.split(pidrcontent, 2, axis=0)[::-1], axis=0)
            # id_preserve_f2=tf.reduce_mean(tf.squared_difference(pidrcontent_ex,pidf2content))
            self.id_preserve_f2 = tf.reduce_mean(tf.abs(tf.subtract(pidrcontent_ex, pidf2content)))

            #adversarial
            # self.ad_loss_syn1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.ones_like(fake1logits),logits=fake1logits
            # ))
            # self.ad_loss_syn2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.ones_like(fake2logits),logits=fake2logits
            # ))
            shape=input_data.shape.as_list()
            # total variation denoising
            tv_y_size = _tensor_size(output_de[0, 1:, :, :])*self.batch_size*self.gpus_count
            tv_x_size = _tensor_size(output_de[0, :, 1:, :])*self.batch_size*self.gpus_count
            self.tv_loss_f1 = TOTAL_VARIATION_SMOOTHING * 2 * (
                (tf.nn.l2_loss(output_de[:, 1:, :, :] - output_de[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(output_de[:, :, 1:, :] - output_de[:, :, :shape[2] - 1, :]) /
                 tv_x_size))
            self.tv_loss_f2 = TOTAL_VARIATION_SMOOTHING * 2 * (
                (tf.nn.l2_loss(output_de_rp[:, 1:, :, :] - output_de_rp[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(output_de_rp[:, :, 1:, :] - output_de_rp[:, :, :shape[2] - 1, :]) /
                 tv_x_size))
            #piexl
            input_data_ex = tf.concat(tf.split(input_data, 2, axis=0)[::-1], axis=0)
            self.pixel_loss_f1 = tf.reduce_mean(tf.abs(tf.subtract(input_data,output_de)))
            self.pixel_loss_f2 = tf.reduce_mean(tf.abs(tf.subtract(input_data_ex,output_de_rp)))
        with tf.name_scope('loss_d'):
            # tf.summary.scalar('id_loss_d', id_loss_real)
            tf.summary.scalar('ps_loss_d', self.ps_loss_real)
            # summary_train_lossD_idf1 = tf.summary.scalar('id_loss_fake', id_loss_fake)
            # summary_train_lossD_idf2 = tf.summary.scalar('id_loss_fake_ex', id_loss_fake_ex)
            tf.summary.scalar('ad_loss_real', self.id_loss_real)
            tf.summary.scalar('ad_loss_fake1', self.id_loss_fake1)
            tf.summary.scalar('ad_loss_fake2', self.id_loss_fake2)
        with tf.name_scope('loss_g'):
            # tf.summary.scalar('id_loss_syn1', id_loss_syn)
            # tf.summary.scalar('id_loss_syn2', id_loss_syn_ex)
            tf.summary.scalar('idpresv_loss_syn1', self.id_preserve_f1)
            tf.summary.scalar('idpresv_loss_syn2', self.id_preserve_f2)
            tf.summary.scalar('ps_loss_syn1', self.ps_loss_syn1)
            tf.summary.scalar('ps_loss_syn2', self.ps_loss_syn2)
            tf.summary.scalar('ad_loss_syn1', self.id_loss_syn1)
            tf.summary.scalar('ad_loss_syn2', self.id_loss_syn2)
            tf.summary.scalar('tv_loss_syn1', self.tv_loss_f1)
            tf.summary.scalar('tv_loss_syn2', self.tv_loss_f2)
            tf.summary.scalar('pixel_loss_syn1', self.pixel_loss_f1)
            tf.summary.scalar('pixel_loss_syn2', self.pixel_loss_f2)
        with tf.name_scope ('error_reconstruction'):
            self.error=tf.reduce_mean(tf.squared_difference(input_data,output_de))
            input_data_ex=tf.concat(tf.split(input_data, 2, axis=0)[::-1], axis=0)
            self.error_ex=tf.reduce_mean(tf.squared_difference(input_data_ex,output_de_rp))
            tf.summary.scalar('error', self.error)
            tf.summary.scalar('error_ex',self.error_ex)
        if True:
            summary_train_image_batch = tf.summary.image('image0/input', tf.expand_dims(input_data[0][:,:,::-1], 0))
            summary_train_image_sample = tf.summary.image('image0/decoder', tf.expand_dims(output_de[0][:,:,::-1], 0))
            summary_train_image_sample = tf.summary.image('image0/decoder_re', tf.expand_dims(output_de_rp[0][:,:,::-1], 0))

            summary_train_image_batch = tf.summary.image('image1/input', tf.expand_dims(input_data[1][:,:,::-1], 0))
            summary_train_image_sample = tf.summary.image('image1/decoder', tf.expand_dims(output_de[1][:,:,::-1], 0))
            summary_train_image_sample = tf.summary.image('image1/decoder_re', tf.expand_dims(output_de_rp[1][:,:,::-1], 0))

            summary_train_image_batch = tf.summary.image('image2/input', tf.expand_dims(input_data[2][:,:,::-1],0))
            summary_train_image_sample = tf.summary.image('image2/decoder', tf.expand_dims(output_de[2][:,:,::-1], 0))
            summary_train_image_sample = tf.summary.image('image2/decoder_re', tf.expand_dims(output_de_rp[2][:,:,::-1], 0))
        # test accuracy:
        # with tf.name_scope('accuracy-pose'):
        #
        #     p_reshape_R = tf.reshape(ppsr_logits, [-1, self.pose_c])
        #     p_max_r = tf.argmax(p_reshape_R, 1)
        #     p_label_true = tf.argmax(pose, 1)
        #     p_label_true_reverse = tf.argmax(pose_reverse, 1)
        #     p_correct_pred_r = tf.equal(p_max_r, p_label_true)
        #     p_reshape_F = tf.reshape(ppsf1_logits, [-1, self.pose_c])
        #     p_max_f = tf.argmax(p_reshape_F, 1)
        #     p_correct_pred_f = tf.equal(p_max_f, p_label_true)
        #     p_reshape_F1 = tf.reshape(ppsf2_logits, [-1, self.pose_c])
        #     p_max_f1 = tf.argmax(p_reshape_F1, 1)
        #     p_correct_pred_f1 = tf.equal(p_max_f1, p_label_true_reverse)
        #     p_accuracy_r = tf.reduce_mean(tf.cast(p_correct_pred_r, tf.float32))
        #     p_accuracy_f1 = tf.reduce_mean(tf.cast(p_correct_pred_f, tf.float32))
        #     p_accuracy_f2 = tf.reduce_mean(tf.cast(p_correct_pred_f1, tf.float32))
        #
        #     summary_train_accracy_r = tf.summary.scalar('real_data', p_accuracy_r)
        #     summary_train_accracy_f = tf.summary.scalar('fake_data', p_accuracy_f1)
        #     summary_train_accracy_f = tf.summary.scalar('fake_data_exchange', p_accuracy_f2)
        # with tf.name_scope('accuracy-identity'):
        #     reshape_R = tf.reshape(pidrlogits, [-1, self.class_nums])
        #     max_r = tf.argmax(reshape_R, 1)
        #     label_true = tf.argmax(label, 1)
        #     correct_pred_r = tf.equal(max_r, label_true)
        #     reshape_F = tf.reshape(pidf1logits, [-1, self.class_nums])
        #     max_f = tf.argmax(reshape_F, 1)
        #     correct_pred_f = tf.equal(max_f, label_true)
        #     reshape_F1 = tf.reshape(pidf2logits, [-1, self.class_nums])
        #     max_f1 = tf.argmax(reshape_F1, 1)
        #     correct_pred_f1 = tf.equal(max_f1, label_true)
        #     accuracy_r = tf.reduce_mean(tf.cast(correct_pred_r, tf.float32))
        #     accuracy_f1 = tf.reduce_mean(tf.cast(correct_pred_f, tf.float32))
        #     accuracy_f2 = tf.reduce_mean(tf.cast(correct_pred_f1, tf.float32))
        #
        #     summary_train_accracy_r = tf.summary.scalar('real_data', accuracy_r)
        #     summary_train_accracy_f = tf.summary.scalar('fake_data', accuracy_f1)
        #     summary_train_accracy_f = tf.summary.scalar('fake_data_exchange', accuracy_f2)
    #K-L divergence
    # latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)

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
            # 单Label + dual data，取消random_data
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
            input_label = tf.placeholder(dtype= tf.int64,shape=[None],name='input_labels')#label
            input_pose = tf.placeholder(dtype= tf.int64,shape=[None],name='input_poses')#pose
            index = tf.placeholder(tf.int32, None,name='input_nums')
            #mk onehot labels
            input_false_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='falselabel')
            fake_labels = slim.one_hot_encoding(input_false_labels, self.class_nums)  # 现阶段不用

            labels = slim.one_hot_encoding(input_label,self.class_nums)
            poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
            pose_reverse = tf.concat(tf.split(poses, 2, axis=0)[::-1], axis=0)
            #加入新的随机Pose
            noise = tf.random_uniform(shape=(index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
            noise_reverse = tf.concat(tf.split(noise, 2, axis=0)[::-1], axis=0)
            #comput loss

            self.loss_gan_multipie(input_data,noise=noise,pose=poses,label=labels,pose_reverse=pose_reverse,fakelabel=fake_labels,noise_reverse=noise_reverse)


            # Dloss_warmup=id_loss_real+ps_loss_real+ad_loss_real+ad_loss_fake1
            # Gloss_warmup=ps_loss_syn1+id_preserve_f1+ad_loss_syn1
            Dloss_warmup=self.id_loss_real+0.5*(self.id_loss_fake1+self.id_loss_fake2)+\
                         self.ps_loss_real

            Gloss_warmup=0.5*(self.id_loss_syn1+self.id_loss_syn2)+ \
                         0.5 * (self.ps_loss_syn1 + self.ps_loss_syn2)

            Dloss = 0.5*self.ps_loss_real+\
                    2*(self.id_loss_real+0.5*(self.id_loss_fake1+self.id_loss_fake2))
            Gloss = 0.5 * (self.ps_loss_syn1+self.ps_loss_syn2)+\
                    0.003*(self.id_loss_syn1+self.id_loss_syn2)+\
                    0.003*(self.id_preserve_f1+self.id_preserve_f2)+\
                    20*(self.pixel_loss_f1+self.pixel_loss_f2)
            #0.0001*(self.tv_loss_f1+self.tv_loss_f2)+\
            #pre 1e-3 adv 1e-3 id_p 3e-3 pixel 1 tv 1e-4
            with tf.name_scope('test'):
                output_en_test = Nets.netG_encoder_gamma(input_data,reuse=True)

                code=tf.get_variable('slerpcode',shape=(self.test_slerp_count,1,1,519),dtype=tf.float32)

                # pose_reshape_test = tf.reshape(poses, [-1, 1, 1, self.pose_c])
                # pose_float_test = tf.cast(pose_reshape_test, dtype=tf.float32)
                # pose_add_noise_test = tf.concat([pose_float_test, noise], 3)

                noise_3=tf.random_uniform(shape=(self.test_slerp_count, 1, 1, self.noise_z), minval=-1, maxval=1, dtype=tf.float32,
                                  name='test_noise')
                sample_add_pn_test = tf.concat([code, noise_3], 3)
                output_de_test = Nets.netG_deconder_gamma(sample_add_pn_test, self.output_channel,reuse=True)

            with tf.name_scope('testpose_identity'):
                _, _, _, \
                _, _, _, \
                _, _, _, \
                outputdecoder_t_f1, output_en_t, outputdecoder_t_f2, \
                _, _, _, \
                _, _, _ = self.predict_drgan_multipie(input_data,noise=noise,pose=poses,pose_reverse=pose_reverse,noise_reverse=noise_reverse
                                                                                   ,reuse=True)
                error = tf.reduce_mean(tf.squared_difference(input_data, outputdecoder_t_f1))
                input_data_test_ex=tf.concat(tf.split(input_data, 2, axis=0)[::-1], axis=0)
                error_ex = tf.reduce_mean(tf.squared_difference(input_data_test_ex, outputdecoder_t_f2))
                summary_train_lossG_error = tf.summary.scalar('error', error)
                summary_train_lossG_error_ex = tf.summary.scalar('error_ex', error_ex)



            summary_train_lossD = tf.summary.scalar('losstotal/total_loss_warmd',Dloss_warmup)
            summary_train_lossG = tf.summary.scalar('losstotal/total_loss_warmg',Gloss_warmup)
            summary_train_lossD = tf.summary.scalar('losstotal/total_loss_d',Dloss)
            summary_train_lossG = tf.summary.scalar('losstotal/total_loss_g',Gloss)

            summary_train = tf.summary.merge_all( )
            train_vars = tf.trainable_variables()
            self.varsG = [var for var in train_vars if 'generator' in var.name]
            self.varsD = [var for var in train_vars if 'discriminator' in var.name]
            self.fc_add = [var for var in train_vars if 'recognation_fc_forGAN' in var.name]
            self.varsD=self.varsD+self.fc_add
            self.varD_light_9 = [var for var in train_vars if 'resnet_yd' in var.name]
            self.var_total=self.varsG+self.varsD+self.varD_light_9
            # self.varD_light_recognition =  [ var for var in train_vars if 'recognition_soft' in var.name]
            # self.varsD_light = [var for var in train_vars if 'light']

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True

            with tf.Session(config=config) as sess:
                with tf.name_scope('train_optimizer'):
                    # g_lr=tf.train.exponential_decay(self.g_lr,global_step=global_step,decay_steps=4*self.batch_idxs,decay_rate=0.99)
                    optimizer_d = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=self.beta1,
                                                       name='optimizer_d')
                    optimizer_g = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=self.beta1,
                                                       name='optimizer_g')
                    # optimizer_light = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9,name='optimizer_light')
                    # optimizer_recog = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9,
                    #                                              name='optimizer_light_recog')
                    # batch norm,when trainingm the moving_mean and moving_variance need to be updated!!!!!!
                    #Gather batch normal    ization update operations
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):

                        train_op_d_warmup=optimizer_d.minimize(Dloss_warmup,var_list=self.varsD,colocate_gradients_with_ops=True,global_step=global_step)
                        train_op_g_warmup=optimizer_g.minimize(Gloss_warmup,var_list=self.varsG,colocate_gradients_with_ops=True)
                        train_op_d = optimizer_d.minimize(Dloss, var_list=self.varsD,colocate_gradients_with_ops=True,global_step=global_step)
                        train_op_g = optimizer_g.minimize(Gloss, var_list=self.varsG,colocate_gradients_with_ops=True)
                        # train_op_d_lightcnn = optimizer_light.minimize(Dloss,var_list=self.varD_light_9,colocate_gradients_with_ops=True)
                        # optimizer_recog = optimizer_recog.minimize(Dloss,var_list=self.varD_light_recognition,colocate_gradients_with_ops=True)

                try:
                    tf.global_variables_initializer().run()
                except:
                    tf.initialize_all_variables().run()
                # print sess.run(self.varsG)
                # print sess.run(self.varsD)
                # print sess.run(self.varD_light_9)
                # print sess.run(self.varD_light_recognition)
                saver = tf.train.Saver(max_to_keep=20)
                saver_restore_lightcnn = tf.train.Saver(self.varD_light_9)#加载局部参数
                # saver_restore_generator = tf.train.Saver(self.varsG)
                step = 0
                curr_interval = 0
                if self.restored:
                    saver.restore(sess,self.check_point)
                    # saver_restore_lightcnn.restore(sess, self.check_point)
                    # saver_restore_generator.restore(sess,'./checkpoint/DR_MultiPIE-21466')
                    # step = int(next(re.finditer("(\d+)(?!.*\d)", self.check_point)).group(0))
                start_time = time.time()

                self.data_loader_train.Dual_enqueueStart()#开启训练对别
                self.data_loader_valid.Dual_enqueueStart()#开启测试队列
                # self.data_loader_train.enqueueStart()

                # hyperparameter=self.savename+'drgan'
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
                avg_error=0
                for epoch_n in xrange(self.epoch):
                    for interval_i in trange(self.batch_idxs):

                        batch_image=np.zeros([self.batch_size*self.gpus_count,self.input_size,self.input_size,self.input_channel],np.float32)
                        batch_label=np.zeros([self.data_loader_train.labels_nums,self.batch_size*self.gpus_count],np.float32)
                        for b_i in xrange(self.gpus_count):
                            batch_image[b_i*self.batch_size:(b_i+1)*self.batch_size,:,:,:],batch_label[:,b_i*self.batch_size:(b_i+1)*self.batch_size]=self.data_loader_train.read_data_batch()
                        flabels=[self.data_loader_train.class_nums] * batch_image.shape[0]

                        if step < int(self.warmup_rate*self.batch_idxs) and self.warmup:
                            #D
                            _ =sess.run(train_op_d_warmup,
                                    feed_dict={input_data:batch_image,
                                               input_label:batch_label[0],
                                               input_pose:batch_label[1],
                                               input_false_labels:flabels,
                                               index: batch_image.shape[0]})
                            #G
                            for _ in xrange(self.Gloop):
                                _, sample_data,sample_data_ex, loss_d, loss_g, train_summary = sess.run(
                                    [train_op_g_warmup, self.outputdecoder_f1,self.outputdecoder_f2, Dloss_warmup, Gloss_warmup, summary_train],
                                feed_dict={input_data: batch_image,
                                           input_label: batch_label[0],
                                           input_pose: batch_label[1],
                                           input_false_labels: flabels,
                                           index: batch_image.shape[0]})
                            # print 'INFO : identity acc_r:%0.4f,acc:%0.4f,acc:%0.4f \n INFO: pose acc_r:%0.4f,acc:%0.4f,acc:%0.4f ' % (acc_r, acc_f1, acc_f2,acc_pr,acc_pf1,acc_pf2)
                            # logging.info('INFO : identity acc_r:%0.4f,acc:%0.4f,acc:%0.4f \n INFO: pose acc_r:%0.4f,acc:%0.4f,acc:%0.4f ' % (acc_r, acc_f1, acc_f2,acc_pr,acc_pf1,acc_pf2))
                        else:
                            #D

                            _,error1,error2=sess.run([train_op_d,self.error,self.error_ex],
                                    feed_dict={input_data:batch_image,
                                               input_label:batch_label[0],
                                               input_pose:batch_label[1],
                                               input_false_labels: flabels,
                                               index: batch_image.shape[0]})
                            if step==0:
                                avg_error = (error2 + error1) / 2.
                            else:
                                avg_error=(error2+error1)/4.+avg_error/2.

                            #G
                            for _ in xrange(self.Gloop):
                                _, sample_data,sample_data_ex, loss_d, loss_g, train_summary = sess.run(
                                    [train_op_g, self.outputdecoder_f1,self.outputdecoder_f2, Dloss, Gloss, summary_train],
                                feed_dict={input_data: batch_image,
                                           input_label: batch_label[0],
                                           input_pose: batch_label[1],
                                           input_false_labels: flabels,
                                           index: batch_image.shape[0]})
                            # print 'INFO : identity acc_r:%0.4f,acc:%0.4f,acc:%0.4f \n INFO: pose acc_r:%0.4f,acc:%0.4f,acc:%0.4f ' % (
                            # acc_r, acc_f1, acc_f2, acc_pr, acc_pf1, acc_pf2)
                            # logging.info('INFO : identity acc_r:%0.4f,acc:%0.4f,acc:%0.4f \n INFO: pose acc_r:%0.4f,acc:%0.4f,acc:%0.4f ' % (
                            #     acc_r, acc_f1, acc_f2, acc_pr, acc_pf1, acc_pf2))
                        summary_write.add_summary(train_summary,global_step=step)

                        step+=1
                        logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, d_loss:%.4f, g_loss:%.4f'\
                        %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss_d,loss_g))
                        print 'Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, d_loss:%.4f, g_loss:%.4f'\
                        %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss_d,loss_g)
                        tmp_batch=None
                        if (curr_interval)%int(self.sample_interval*self.batch_idxs)==0:
                            #记录训练数据
                            utils.write_batch(self.result_path,0,sample_data,batch_image,epoch_n,interval_i,othersample=sample_data_ex,ifmerge=True)
                            sample_batch=np.zeros([self.test_batch_size,self.input_size,self.input_size,self.input_channel],np.float32)
                            label_batch = np.zeros([self.data_loader_train.labels_nums, self.test_batch_size], np.float32)
                            #加载测试batch
                            for s_i in xrange(1):
                                sample_batch[s_i*self.pose_c:(s_i+1)*self.pose_c,:,:,:] ,label_batch[:,s_i*self.pose_c:(s_i+1)*self.pose_c] =self.data_loader_valid.oneperson_allpose(s_i)#得到一个人所有的图片
                            sample_count = sample_batch.shape[0]
                            flabels = [self.data_loader_train.class_nums] * sample_batch.shape[0]
                            #identity-preserved 测试
                            if (curr_interval)%int(self.sample_pose_interval * self.batch_idxs)==0:
                                idlabel_batch=[0]*sample_count
                                sample_data, sample_data_ex,error1,error2= sess.run(
                                    [outputdecoder_t_f1,outputdecoder_t_f2,error,error_ex],
                                    feed_dict={input_data: sample_batch,
                                               input_label: idlabel_batch,
                                               input_pose: label_batch[1],
                                               input_false_labels: flabels,
                                               index: sample_batch.shape[0]})#falselable无所谓了
                                utils.write_batch(self.result_path, 1, sample_data, sample_batch, epoch_n,
                                                  interval_i,othersample=sample_data_ex,ifmerge=True)

                                print 'reconstruction error: {}、{}/avgerror{}'.format(error1,error2,avg_error)
                                # pose - invariance测试
                                for idx in xrange(sample_count):  # 将数据集中的同一个人所有!!!角度！！！照片都跑一次
                                    tppn = self.test_batch_size
                                    label_batch_sub = [sample_count] * tppn  # 没用凑齐8 为了后面的split
                                    pose_batch = range(0, tppn)
                                    tmp_batch = np.tile(sample_batch[idx], (tppn, 1, 1, 1)). \
                                        reshape(tppn, sample_batch.shape[1], sample_batch.shape[2],
                                                sample_batch.shape[3])
                                    flabels = [self.data_loader_train.class_nums] * tmp_batch.shape[0]
                                    sample_data,sample_data_ex = sess.run([outputdecoder_t_f1,outputdecoder_t_f2],
                                                           feed_dict={
                                                               input_data: tmp_batch,
                                                               input_label: label_batch_sub,
                                                               input_pose: pose_batch,
                                                               input_false_labels: flabels,
                                                               index: tmp_batch.shape[0]})
                                    utils.write_batch(self.result_path, 2, sample_data, tmp_batch, epoch_n,
                                                      interval_i, sample_idx=idx,othersample=sample_data_ex,ifmerge=True)
                        # slerp
                        if (curr_interval) % int(self.test_interval * self.batch_idxs) == 0:
                            # 球面差值法测试
                            # for _ in xrange(3):#测试三次
                            # print 'slerp test!!'
                            index_select_1 = np.random.randint(0, batch_image.shape[0], self.test_slerp_count)
                            index_select_2 = np.random.randint(0, batch_image.shape[0], self.test_slerp_count)
                            flabels = [self.data_loader_train.class_nums] * batch_image[index_select_1].shape[0]
                            de_code_1 = sess.run(output_en_test,
                                                 feed_dict={input_data: batch_image[index_select_1],
                                                            input_label: batch_label[0][index_select_1],
                                                            input_pose: batch_label[1][index_select_1],
                                                            input_false_labels: flabels,
                                                            index: batch_image[index_select_1].shape[0]})
                            de_code_2 = sess.run(output_en_test,
                                                 feed_dict={input_data: batch_image[index_select_2],
                                                            input_label: batch_label[0][
                                                                index_select_2],
                                                            input_pose: batch_label[1][
                                                                index_select_2],
                                                            input_false_labels: flabels,
                                                            index: batch_image[index_select_2].shape[0]})
                            pose_1 = np.asarray(batch_label[1][index_select_1], np.int32)
                            pose_2 = np.asarray(batch_label[1][index_select_2], np.int32)
                            b_1 = np.zeros((pose_1.size, self.pose_c), dtype=np.float32)
                            b_2 = np.zeros_like(b_1, dtype=np.float32)
                            b_1[np.arange(pose_1.size), pose_1] = 1
                            b_2[np.arange(pose_2.size), pose_2] = 1

                            de_code_1 = np.concatenate([np.reshape(de_code_1, [-1, 512]), b_1], axis=1)
                            de_code_2 = np.concatenate([np.reshape(de_code_2, [-1, 512]), b_2], axis=1)
                            decodes = []
                            for idx, ratio in enumerate(np.linspace(0, 1, 10)):
                                z = np.stack([utils.slerp(ratio, r1, r2) for r1, r2 in
                                              zip(de_code_1, de_code_2)])
                                z = np.reshape(z, [-1, 1, 1, 519])
                                # print z.shape
                                flabels = [self.data_loader_train.class_nums] * batch_image.shape[0]
                                z_decode = sess.run(output_de_test,
                                                    feed_dict={code: z,
                                                               input_data: batch_image,
                                                               input_label: batch_label[0],
                                                               input_pose: batch_label[1],
                                                               input_false_labels: flabels,
                                                               index: batch_image.shape[0]})
                                decodes.append(z_decode)

                            decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
                            for idx, img in enumerate(decodes):
                                img = np.concatenate([[batch_image[index_select_1[idx]]], img,
                                                      [batch_image[index_select_2[idx]]]], 0)
                                img = utils.inverse_transform(img)[:, :, :, ::-1]
                                utils.save_image(img, os.path.join('./{}'.format(self.result_path),
                                                                   'test{:08}_interp_G_{:01}.png'.format(step, idx)),
                                                 nrow=10 + 2)

                            if self.ifsave and curr_interval != 0:
                                saver.save(sess,
                                           os.path.join(self.check_point_path, self.model_name),
                                           global_step=step)
                                print '*' * 20 + 'save model successed!!!!~~~~'
                        curr_interval+=1


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)






