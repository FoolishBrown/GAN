#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
# import skimage
# import datetime
import tensorflow.contrib.slim as slim
import os
import sys
import cv2
import re
import time
from PIL import Image
# import scipy.misc
# from tensorflow.contrib.tensorboard.plugins import projector
import utils
import  logging
import nets_frontalization as nets
from resnet_yd import resnet_yd
from tqdm import trange
import synthesis
import inference as test_phase
sys.path.insert(0,'./tf_inception_v1')
from inception_resnet_v1 import inference
import test_main_lfw
TOTAL_VARIATION_SMOOTHING = 1e2

class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,version='0',check_point_fr=None,
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=1,
                 data_loader_train=None,data_loader_valid=None,epoch=10,log_dir='logdir',d_learning_rate=0.0002,g_learning_rate=0.0002,g_learning_rate_fr=0.0002,beta_fr=0.95,beta1=0.5,
                 test_batch_size=10,gpus_list=None,model_name='dr_gan',noise_z=50,random_seed=25,
                 Gloop=3,savepath='savename',imagepath='image',logfile='file.log',summary_dir='summary',discribe='',gpu='7'):
        self.g_lr_fr=g_learning_rate_fr
        self.beta1_fr=beta_fr
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
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.epoch = epoch
        #dir
        self.log_dir = log_dir
        self.savename=savepath
        self.result_path = imagepath
        if restored:
            self.check_point=check_point
            self.check_point_path=os.path.join(os.path.dirname(logfile),'checkpoint')
        else:
            self.check_point=check_point
            # self.check_point_path=check_point
            self.check_point_path = os.path.join(os.path.dirname(logfile), 'checkpoint')
        self.logfile=logfile
        self.summarypath=summary_dir

        self.gpus_list=gpus_list
        self.model_name=model_name

        self.noise_z=noise_z
        self.ifsave=ifsave
        self.g_loop=Gloop
        self.class_nums=self.data_loader_train.class_nums
        self.random_seed=random_seed
        self.test_slerp_count=3
        #save loss and vars
        self.g_loss=None
        self.d_loss=None
        self.varsg=None
        self.varsd=None
        self.gpu=gpu
        self.version=version
        self.mkdir_result(self.log_dir)#save mode dir
        self.mkdir_result(self.summarypath)#save summary
        self.mkdir_result(self.savename)#save result dir
        self.mkdir_result(self.result_path)#save image dir
        self.mkdir_result(self.check_point_path)#checkpoint dir
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',datefmt='%a,%d %b %Y %H:%M:%S',filename=self.logfile,filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        self.gpus_arr=np.asarray(self.gpus_list.split(','),np.int32)
        print 'use gpu nums:',len(self.gpus_arr)
        self.gpus_count=len(self.gpus_arr)
        if self.gpus_count>1:
            self.multigpus=True
        else:
            self.multigpus=False
        self.batch_idxs = self.data_loader_train.batch_idxs / self.gpus_count
        #初始化模型
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus_list
        self._init_model()
        self._init_validation_model()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=20)
        saver_init = tf.train.Saver(self.init_vars)  # 加载局部参数
        saver_inception=tf.train.Saver(self.inception_vars)

        saver_inception.restore(self.sess,check_point_fr)
        if self.restored:
            # self.saver.restore(self.sess,self.check_point)
            saver_init.restore(self.sess,self.check_point)
        self.summary_write = tf.summary.FileWriter(self.summarypath + '/' + self.version+'_'+self.gpus_list, self.sess.graph)
        self.data_loader_valid.Dual_enqueueStart(frontalization=True)#开启训练对别
        self.data_loader_train.Dual_enqueueStart(frontalization=True)#开启训练对别

    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))

    def _init_model(self):
        '''
        init modle for train
        :return:
        '''

        self.global_step = slim.get_or_create_global_step()
        self.batch_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
        self.batch_label = tf.placeholder(dtype= tf.int64,shape=[None],name='input_labels')#label
        self.input_data,self.gt_input_data=tf.split(self.batch_data,2,axis=0)
        #mk onehot labels
        self.labels = slim.one_hot_encoding(self.batch_label,self.class_nums)
        #comput loss
        self._predict_drgan_multipie()
        self._loss_gan_multipie()
        self._loss_compute()
        self.summary_train = tf.summary.merge_all()
        #select var list
        train_vars = tf.trainable_variables()
        self.varsg_decoder = [var for var in train_vars if 'decoding' in var.name]
        self.varsg_merge = [var for var in train_vars if 'merging' in var.name]
        self.varsd = [var for var in train_vars if 'discriminator' in var.name]
        self.fc_add = [var for var in train_vars if 'recognition_resnet_fc' in var.name]
        self.inception=[var for var in train_vars if 'Inception-ResNet-v1' in var.name]
        self.vard_fr= [var for var in train_vars if 'resnet_yd' in var.name]
        # self.vard_fr = [var for var in train_vars if 'encoding' in var.name]
        self.varfr_all=self.vard_fr+self.fc_add
        self.inception_vars=self.inception
        self.init_vars = self.varsg_merge+self.varsg_decoder+self.vard_fr+self.varsd
        self.varsg=self.varsg_merge+self.varsg_decoder+self.varfr_all
        self._get_train_op(self.global_step)

    def _init_validation_model(self):
        '''
        init model for identity and slerpcode
        :return:
        '''
        with tf.name_scope('test'):
            _,_,self.encode_slerp= resnet_yd(self.input_data, reuse=True)
            # _,_,self.encode_slerp = nets.netG_encoder_gamma(self.input_data,self.class_nums, reuse=True)
            self.encode_slerp_z=tf.get_variable('code',[3,562])
            self.image_syn_slerp_middle = nets.netG_deconder_gamma(self.encode_slerp_z, self.output_channel, reuse=True)
            self.image_syn_slerp = nets.merge_net_16_unet(self.image_syn_slerp_middle, self.input_data, reuse=True)

    def validation(self,interval_i,epoch_n,step):
        '''
        inference
        :return:
        '''
        sample_batch = np.zeros(
            [self.batch_size * self.gpus_count, self.input_size, self.input_size, self.input_channel], np.float32)
        sample_label = np.zeros([self.data_loader_valid.labels_nums, self.batch_size * self.gpus_count], np.float32)
        for b_i in xrange(self.gpus_count):
            sample_batch[b_i * self.batch_size:(b_i + 1) * self.batch_size, :, :, :], \
            sample_label[:,b_i * self.batch_size:(b_i + 1) * self.batch_size] = self.data_loader_valid.read_data_batch()
        # identity-preserved 测试
        sample_data, encode_syn, encode_real = self.sess.run(
            [self.output_syn, self.cosine_syn, self.cosine_real],
            feed_dict={self.batch_data: sample_batch,
                       self.batch_label: sample_label[0]})
        score_identity = np.concatenate([encode_syn, encode_real], axis=0)
        sample_batch=np.split(sample_batch,2,axis=0)
        utils.write_batch(self.result_path, 1, sample_data, sample_batch[1],
                          interval_i, ifmerge=True,score_f_id=score_identity,othersample=sample_batch[0],reverse_other=False)
        logging.info('[score_identity] {:08} {}'.format(step,score_identity))

    def slerp_interpolation(self,batch_image,batch_label,epoch,interval):
        #用了patch的方法是不能做slerp—interpolation操作的
        # 球面差值法测试
        index_select = np.random.randint(0, batch_image.shape[0], self.test_slerp_count*2)
        index_select_sub=np.split(index_select,2,axis=0)
        encode_512 = self.sess.run(self.encode_slerp,
                             feed_dict={self.input_data: batch_image[index_select]})
        encode_50=np.random.uniform(high=1, low=-1, size=[self.test_slerp_count*2,self.noise_z])
        encode_562=np.concatenate([encode_512,encode_50],axis=1)
        encode_sub = np.split(encode_562,2,axis=0)
        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([utils.slerp(ratio, r1, r2) for r1, r2 in
                          zip(encode_sub[0], encode_sub[1])])
            z = np.reshape(z, [-1,562])
            z_decode = self.sess.run(self.image_syn_slerp,
                                feed_dict={self.input_data: batch_image[index_select],
                                    self.encode_slerp_z: z})
            decodes.append(z_decode)
        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        index_sub=np.split(index_select,2,axis=0)
        index_sub=np.split(index_sub[0],2,axis=0)
        for idx, img in enumerate(decodes):
            img = np.concatenate([[batch_image[index_sub[0][idx]]], img,
                                  [batch_image[index_sub[1][idx]]]], 0)
            img = utils.inverse_transform(img)[:, :, :, ::-1]
            utils.save_image(img, os.path.join('./{}'.format(self.result_path),
                                               'test{:08}_interp_G.png'.format(interval )), nrow=10 + 2)

    def train(self):
        '''
        train
        :return:
        '''
        start_time = time.time()

        curr_interval=0
        for epoch_n in xrange(self.epoch):
            for interval_i in trange(self.batch_idxs):
                batch_image = np.zeros(
                    [self.batch_size * self.gpus_count, self.input_size, self.input_size, self.input_channel],
                    np.float32)
                batch_label = np.zeros([self.data_loader_train.labels_nums, self.batch_size * self.gpus_count],
                                       np.float32)
                for b_i in xrange(self.gpus_count):
                    batch_image[b_i*self.batch_size:(b_i+1)*self.batch_size,:,:,:],batch_label[:,b_i*self.batch_size:(b_i+1)*self.batch_size]=self.data_loader_train.read_data_batch()
                #D
                _ ,loss_d=self.sess.run([self.train_d_op,self.d_loss],
                        feed_dict={self.batch_data:batch_image,
                                   self.batch_label:batch_label[0]})
                #G
                for _ in xrange(self.g_loop):
                    _ = self.sess.run(self.train_g_op,
                    feed_dict={self.batch_data: batch_image,
                               self.batch_label: batch_label[0]})
                sample_data, loss_fr, loss_g, train_summary,\
                data,step, \
                encode_real, encode_syn\
                    = self.sess.run(
                    [self.output_syn,self.g_loss_fr,self.g_loss, self.summary_train,
                    self.input_data,self.global_step,self.cosine_real,self.cosine_syn],
                    feed_dict={self.batch_data: batch_image,
                               self.batch_label: batch_label[0]})
                self.summary_write.add_summary(train_summary,global_step=step)

                logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, d_loss:%.4f, g_loss:%.4f,lossfr:%.4f'\
                %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss_d,loss_g,loss_fr))
                if (curr_interval) % int(self.sample_interval * self.batch_idxs) == 0:
                    # 记录训练数据
                    score_train = np.concatenate([encode_syn, encode_real], axis=0)
                    logging.info('[score_train] {:08} {}'.format(step, score_train))
                    batch_image = np.split(batch_image, 2, axis=0)
                    print sample_data.shape
                    utils.write_batch(self.result_path, 0, sample_data, batch_image[1], curr_interval,
                                      othersample=batch_image[0], reverse_other=False, ifmerge=True,
                                      score_f_id=score_train)
                    self.validation(curr_interval, epoch_n, step)
                    # self.slerp_interpolation(batch_image[1],batch_label,epoch_n,curr_interval)
                    if self.ifsave:
                        modelname = self.model_name + '-' + str(curr_interval)
                        self.saver.save(self.sess,
                                        os.path.join(self.check_point_path, self.model_name),
                                        global_step=curr_interval)
                        # save_path='/world/data-gpu-58/wangyuequan/data_sunkejia/lfw_synthesis_temp/lfw_synthesis//'+self.version+self.gpus_list+'/'
                        # self.mkdir_result(save_path)
                        # save_path='/world/data-gpu-90/rex/lfw/data/lfw_lightcnn_96_rgb/'
                        test_phase.main(str(curr_interval),os.path.join(self.check_point_path,self.model_name),self.gpu)
                        # synthesis.main(os.path.join(self.check_point_path, modelname),save_path+'/synlfw'+str(curr_interval)+'/',self.gpu)
                        print '*' * 20 + 'synthesis image finished!!!~~~~'
                        print '*' * 20 + 'save model successed!!!!~~~~'
                curr_interval+=1

    def _get_train_op(self,global_step):
        '''
        梯度计算
        :param global_step: 迭代次数
        :return:
        '''
        optimizer_d = tf.train.AdamOptimizer(learning_rate=self.d_lr,beta1=self.beta1,name='optimizer_d')
        grads_and_var_d = optimizer_d.compute_gradients(self.d_loss,self.varsd,colocate_gradients_with_ops = True)
        grads_d,vars_d = zip(*grads_and_var_d)
        grads_d,_ =tf.clip_by_global_norm(grads_d,0.1)
        self.train_d_op = optimizer_d.apply_gradients(zip(grads_d,vars_d),global_step)
        optimizer_g = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=self.beta1,name='optimizer_g')
        self.train_g_op_fr = tf.train.MomentumOptimizer(learning_rate=self.g_lr_fr, momentum=self.beta1_fr, name='optimizer_g_fr').minimize(self.g_loss_fr,var_list=self.varfr_all)
        #decoder optimizer
        grads_and_var_g = optimizer_g.compute_gradients(self.g_loss,self.varsg,colocate_gradients_with_ops = True)
        grads_g ,var_g = zip(*grads_and_var_g)
        grads_g , _ = tf.clip_by_global_norm(grads_g,0.1)
        self.train_g_op = optimizer_g.apply_gradients(zip(grads_g,var_g))
        return global_step

    def _predict_drgan_multipie(self,reuse=False):
        '''
        网络训练
        :param reuse: True | False netG_encoder_gamma
        :return:inference_recognition
        '''
        self.softresult, self.logits_cl, self.encode_fr=nets.inference_recognition(self.batch_data, classnums=self.class_nums, reuse=reuse)
        noise = tf.random_uniform(shape=(self.batch_size,self.noise_z), minval=-1, maxval=1, dtype=tf.float32,
                                  name='input_noise')
        self.encode_add_z=tf.concat([self.encode_fr,noise],1)
        self.output_syn_middle = nets.netG_deconder_gamma(self.encode_add_z, self.output_channel,reuse=reuse)
        self.output_syn_middle_profile,self.output_syn_middle_front=tf.split(self.output_syn_middle,2,axis=0)
        self.output_syn_front= nets.merge_net_16_unet(self.output_syn_middle,self.batch_data,reuse=reuse)
        self.identity_real,_ = inference(self.batch_data,keep_prob=1,phase_train=False)
        self.identity_syn,_ = inference(self.output_syn_front,keep_prob=1,phase_train=False,reuse=True)
        self.output_syn, self.output_gt_syn = tf.split(self.output_syn_front, 2, axis=0)
        self.real_logits = nets.Custom_netD_discriminator_adloss(self.gt_input_data,reuse=reuse)
        self.fake_logits = nets.Custom_netD_discriminator_adloss(self.output_syn, reuse=True)
        self.profile_content, self.front_content = tf.split(self.encode_fr, 2, axis=0)
        # 生成图像的features
        self.syn_softmax,self.fake_logits_all, self.syn_encode \
            = resnet_yd(self.output_syn_front,reuse=True)
        self.syn_content, self.syn_front_content = tf.split(self.syn_encode, 2, axis=0)
        # 计算cosine距离
        self.cosine_real = tf.divide(tf.reduce_sum(tf.multiply(self.profile_content, self.front_content), axis=1),
                                     tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.front_content), axis=1)),
                                                 tf.sqrt(tf.reduce_sum(tf.square(self.profile_content), axis=1))))
        self.cosine_syn = tf.divide(tf.reduce_sum(tf.multiply(self.syn_content, self.syn_front_content), axis=1),
                                    tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.syn_content), axis=1)),
                                                tf.sqrt(tf.reduce_sum(tf.square(self.syn_front_content), axis=1))))

    def _loss_gan_multipie(self):
        '''
        loss 计算
        :return:
        '''
        with tf.name_scope('D_loss'):
            #adversarial
            self.ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.real_logits),logits=self.real_logits
            ))
            self.ad_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.fake_logits),logits=self.fake_logits
            ))
        with tf.name_scope('G_loss'):
            #real loss
            self.pre_loss_real=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels,logits=self.logits_cl
            ))
            #syn loss
            # self.pre_loss_syn=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=self.labels,logits=self.syn_logits
            # ))
            #identity preserved loss
            self.id_loss_psd = tf.reduce_mean(tf.abs(tf.subtract( self.syn_content,self.front_content)))
            #identity preserved loss
            self.id_loss_preserved = tf.reduce_mean(tf.abs(tf.subtract(self.identity_syn,self.identity_real)))
            #adversarial
            self.ad_loss_syn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.fake_logits),logits=self.fake_logits
            ))
            # total variation denoisin
            shape = self.batch_data.shape.as_list()
            tv_y_size = _tensor_size(self.output_syn[0, 1:, :, :])*self.batch_size*self.gpus_count
            tv_x_size = _tensor_size(self.output_syn[0, :, 1:, :])*self.batch_size*self.gpus_count
            self.tv_loss_syn = TOTAL_VARIATION_SMOOTHING * 2 * (
                (tf.nn.l2_loss(self.output_syn[:, 1:, :, :] - self.output_syn[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(self.output_syn[:, :, 1:, :] - self.output_syn[:, :, :shape[2] - 1, :]) /
                 tv_x_size))
            #piexl
            self.pix_loss_syn_middle=tf.reduce_mean(tf.abs(tf.subtract(self.output_syn_middle_profile,self.gt_input_data)))
            self.pix_loss_syn_final = tf.reduce_mean(tf.abs(tf.subtract(self.output_syn,self.gt_input_data)))
            #symmetry
            self.output_syn_half_left,self.output_syn_half_right=tf.split(self.output_syn,2,axis=2)
            self.sym_loss_syn = tf.reduce_mean(tf.abs(tf.subtract(self.output_syn_half_left,self.output_syn_half_right[:,:,::-1,:])))

        with tf.name_scope('loss_d'):
            tf.summary.scalar('ad_loss_real', self.ad_loss_real)
            tf.summary.scalar('ad_loss_fake1', self.ad_loss_fake)
        with tf.name_scope('loss_g'):
            tf.summary.scalar('tv_loss_syn', self.tv_loss_syn)
            tf.summary.scalar('sym_loss_syn', self.sym_loss_syn)
            tf.summary.scalar('ad_loss_syn2', self.ad_loss_syn)
            tf.summary.scalar('pix_loss_syn_middle', self.pix_loss_syn_middle)
            tf.summary.scalar('pix_loss_syn_front',self.pix_loss_syn_final)
        with tf.name_scope('distance'):
            tf.summary.scalar('cosine_real', tf.reduce_mean(self.cosine_real))
            tf.summary.scalar('cosine_syn', tf.reduce_mean(self.cosine_syn))

    def _loss_compute(self):#G:D 2:1
        '''
        loss 加权
        :return:
        '''
        self.d_loss = 0.001*(self.ad_loss_real+self.ad_loss_fake)
        self.g_loss = 0.001*self.ad_loss_syn+self.pix_loss_syn_final+self.pix_loss_syn_middle*0.5+\
                    self.tv_loss_syn*0.0001+0.3*self.sym_loss_syn+0.003*self.pre_loss_real+self.id_loss_preserved*0.003
        self.g_loss_fr=self.pre_loss_real
        tf.summary.scalar('losstotal/total_loss_d', self.d_loss)
        tf.summary.scalar('losstotal/total_loss_g', self.g_loss)
        tf.summary.scalar('losstotal/total_fr_loss',self.g_loss_fr)

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)






