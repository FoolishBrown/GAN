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
import nets_frontalization as nets
# from resnet_yd import resnet_yd
# sys.path.insert(0,'./tf_inception_v1')
# from inception_resnet_v1 import inference
from tqdm import trange

TOTAL_VARIATION_SMOOTHING = 1e2

class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,lr_alpha=0.5,lr_beta=0.5,lr_gamma=1,version='0',
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,test_interval=1,
                 data_loader_train=None,data_loader_valid=None,epoch=10000,log_dir='logdir',d_learning_rate=0.0002,g_learning_rate=0.0002,beta1=0.5,delta=0.2,
                 test_batch_size=10,gpus_list=None,write_type=0,model_name='dr_gan',noise_z=50,pose_c=7,random_seed=25,sample_pose_interval=10,warmup_rate=1,
                 loss_type=0,Gloop=3,savepath='savename',imagepath='image',logfile='file.log',summary_dir='summary',warmup=True,discribe='',light_c=20):
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
        self.light_c=light_c
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
        self.version=version
        self.mkdir_result(self.log_dir)#save mode dir
        self.mkdir_result(self.summarypath)#save summary
        self.mkdir_result(self.savename)#save result dir
        self.mkdir_result(self.result_path)#save image dir
        self.mkdir_result(self.check_point_path)#checkpoint dir
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',datefmt='%a,%d %b %Y %H:%M:%S',filename=self.logfile,filemode='w')
        logging.info('gamma:{} alpha:{} beta:{} delta:{} poselabel:(-1,1), discribe{}'.format(lr_gamma,lr_alpha,lr_beta,delta,discribe))
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
        self.saver = tf.train.Saver(max_to_keep=10)
        saver_init = tf.train.Saver(self.init_vars)  # 加载局部参数

        if self.restored:
            # self.saver.restore(self.sess,self.check_point)
            saver_init.restore(self.sess,self.check_point)

        self.summary_write = tf.summary.FileWriter(self.summarypath + '/' + self.version+'_'+self.gpus_list, self.sess.graph)

        self.data_loader_valid.enqueueStart()  # 开启测试队列
        self.data_loader_train.enqueueStart()#开启训练对别

    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))
    #K-L divergence
    # latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)
    def _init_model(self):
        '''
        init modle for train
        :return:
        '''
        # tf.set_random_seed(20)
        # with tf.Graph().as_default():

        self.global_step = slim.get_or_create_global_step()
        self.batch_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
        self.batch_label = tf.placeholder(dtype= tf.int64,shape=[None],name='input_labels')#label
       #mk onehot labels
        self.labels = slim.one_hot_encoding(self.batch_label,self.class_nums)
        #comput loss
        self.softmax_real,self.logits,self.fc=nets.inference_recognition(self.batch_data,self.class_nums)
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels,logits=self.logits
        ))
        lo=tf.summary.scalar('train/pre_loss',self.loss)
        reshape_R = tf.reshape(self.softmax_real, [-1, self.class_nums])
        max_r = tf.argmax(reshape_R, 1)
        self.predict_rate = tf.equal(max_r, self.batch_label)
        self.accuracy_r = tf.reduce_mean(tf.cast(self.predict_rate, tf.float32))
        acc=tf.summary.scalar('train/pre_rate',self.accuracy_r )
        self.summary_train = tf.summary.merge([lo,acc])

        train_vars = tf.trainable_variables()
        self.fc_add = [var for var in train_vars if 'recognition_fc' in var.name]
        self.vard_fr= [var for var in train_vars if 'resnet_yd' in var.name]
        self.init_vars=self.vard_fr
        self.var_all=self.vard_fr+self.fc_add
        train_optimizer=tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.99,name='optimizer')
        self.train_op=train_optimizer.minimize(self.loss,var_list=self.var_all,global_step=self.global_step)

    def _init_validation_model(self):
        '''
        init model for identity and slerpcode
        :return:
        '''
        # slim.assign_from_checkpoint()
        with tf.name_scope('test'):
            self.batch_data_test = tf.placeholder(dtype=tf.float32,
                                             shape=[None, self.input_size, self.input_size, self.input_channel],
                                             name='input_images_test')  # image
            self.batch_label_test = tf.placeholder(dtype=tf.int64, shape=[None], name='input_labels_test')  # label
            self.labels_test = slim.one_hot_encoding(self.batch_label_test,self.class_nums)
            #comput loss
            self.softmax_real_test,self.logits_test,self.fc_test=nets.inference_recognition(self.batch_data_test,self.class_nums,reuse=True)
            self.loss_test=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_test,logits=self.logits_test
            ))
            loss_test=tf.summary.scalar('test/pre_loss_test',self.loss_test)
            reshape_R = tf.reshape(self.softmax_real_test, [-1, self.class_nums])
            max_r = tf.argmax(reshape_R, 1)
            self.predict_rate_test = tf.equal(max_r, self.batch_label_test)
            self.accuracy_r_test = tf.reduce_mean(tf.cast(self.predict_rate_test, tf.float32))
            pre_test=tf.summary.scalar('test/pre_rate_test', self.accuracy_r_test)
            self.summary_test = tf.summary.merge([loss_test,pre_test])
    def train(self):
        '''
        train
        :return:
        '''
        start_time = time.time()

        curr_interval=0
        for epoch_n in xrange(self.epoch):
            for interval_i in trange(self.batch_idxs):
                batch_image=np.zeros([self.batch_size*self.gpus_count,self.input_size,self.input_size,self.input_channel],np.float32)
                batch_label=np.zeros([self.data_loader_train.labels_nums,self.batch_size*self.gpus_count],np.float32)
                for b_i in xrange(self.gpus_count):
                    batch_image[b_i*self.batch_size:(b_i+1)*self.batch_size,:,:,:],batch_label[:,b_i*self.batch_size:(b_i+1)*self.batch_size]=self.data_loader_train.read_data_batch()
                # if interval_i%10:
                _,acc,  loss, train_summary,\
                step\
                    = self.sess.run(
                    [self.train_op,self.accuracy_r,self.loss, self.summary_train,self.global_step],
                    feed_dict={self.batch_data: batch_image,
                               self.batch_label: batch_label[0]})
                self.summary_write.add_summary(train_summary,global_step=step)

                logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, loss:%.4f, acc:%.4f'\
                %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss,acc))

                if (curr_interval) % int(self.test_interval * self.batch_idxs) == 0:
                    test_data,test_label = self.data_loader_valid.read_data_batch()
                    acc, loss, test_summary, \
                        = self.sess.run(
                        [self.accuracy_r_test, self.loss_test, self.summary_test],
                        feed_dict={self.batch_data: batch_image,
                               self.batch_label: batch_label[0],
                                self.batch_data_test: test_data,
                                   self.batch_label_test: test_label[0]})
                    self.summary_write.add_summary(test_summary, global_step=step)

                    logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, loss:%.4f, acc:%.4f' \
                                 % (
                                 epoch_n, self.epoch, self.gpus_list, step, (time.time() - start_time) / 3600.0, loss,
                                 acc))
                    if self.ifsave:
                        self.saver.save(self.sess,
                                   os.path.join(self.check_point_path, self.model_name),
                                   global_step=step)
                        print '*' * 20 + 'save model successed!!!!~~~~'
                curr_interval+=1







