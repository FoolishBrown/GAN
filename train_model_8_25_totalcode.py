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
import Net_8_25 as nets
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
        self.saver = tf.train.Saver(max_to_keep=20)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())

        saver_fr = tf.train.Saver(self.init_vars)  # 加载局部参数

        if self.restored:
            self.saver.restore(self.sess,self.check_point)
            # saver_fr.restore(self.sess,self.check_point)

        self.summary_write = tf.summary.FileWriter(self.summarypath + '/' + self.version+'_'+self.gpus_list, self.sess.graph)

        self.data_loader_valid.Dual_enqueueStart()  # 开启测试队列
        self.data_loader_train.Dual_enqueueStart()#开启训练对别

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
        self.input_label = tf.placeholder(dtype= tf.int64,shape=[None],name='input_labels')#label
        self.input_pose = tf.placeholder(dtype= tf.int64,shape=[None],name='input_poses')#pose
        self.index = tf.placeholder(tf.int32, None,name='input_nums')
        #mk onehot labels
        self.labels = slim.one_hot_encoding(self.input_label,self.class_nums)
        self.pose = slim.one_hot_encoding(self.input_pose,self.pose_c)#pose code pose label
        self.pose_reverse = tf.concat(tf.split(self.pose, 2, axis=0)[::-1], axis=0)
        #加入新的随机Pose
        self.noise = tf.random_uniform(shape=(self.index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
        self.noise_reverse = tf.concat(tf.split(self.noise, 2, axis=0)[::-1], axis=0)
        #comput loss
        self._predict_drgan_multipie()

        self._loss_gan_multipie()
        self._loss_compute()

        #pre 1e-3 adv 1e-3 id_p 3e-3 pixel 1 tv 1e-4

        self.summary_train = tf.summary.merge_all()
        #select var list
        train_vars = tf.trainable_variables()
        self.varsg = [var for var in train_vars if 'generator' in var.name]
        self.varsd = [var for var in train_vars if 'discriminator' in var.name]
        self.fc_add = [var for var in train_vars if 'recognation_fc' in var.name]

        self.vard_fr= [var for var in train_vars if 'resnet_yd' in var.name]
        self.init_vars=self.vard_fr+self.fc_add+self.varsg
        # self.init_vars=self.vard_fr
        # self.var_total=self.varsg+self.varsd+self.vard_fr
        # self.varsd = self.varsd+self.vard_fr+self.fc_add###finetu fr net??

        self._get_train_op(self.global_step)



            # saver_restore_lightcnn.restore(sess, self.check_point)

    def _init_validation_model(self):
        '''
        init model for identity and slerpcode
        :return:
        '''
        with tf.name_scope('test'):
            _, _, output_en_test = resnet_yd(self.batch_data, reuse=True)
            self.encode_slerp = tf.reshape(output_en_test, [-1, 1, 1, 512],name='slerp_reshape_encoder')

            self.slerp_code = tf.get_variable('slerpcode',
                                   shape=(self.test_slerp_count, 1, 1, 512 + self.pose_c + self.noise_z),
                                   dtype=tf.float32)

            self.image_syn_slerp = nets.netG_deconder_gamma(self.slerp_code, self.output_channel, reuse=True)

    def validation(self,interval_i,epoch_n,step):
        '''
        inference
        :return:
        '''

        sample_batch = np.zeros([self.test_batch_size, self.input_size, self.input_size, self.input_channel],
                                np.float32)
        label_batch = np.zeros([self.data_loader_train.labels_nums, self.test_batch_size], np.float32)
        # 加载测试batch
        # someone=np.random.randint(0,self.data_loader_valid.class_nums)
        for s_i in xrange(1):
            sample_batch[s_i * self.pose_c:(s_i + 1) * self.pose_c, :, :, :], label_batch[:, s_i * self.pose_c:(s_i + 1) * self.pose_c] = self.data_loader_valid.oneperson_allpose(s_i)  # 得到一个人所有的图片
        sample_count = sample_batch.shape[0]
        # identity-preserved 测试
        idlabel_batch = [0] * sample_count
        sample_data, sample_data_ex,encode_real,encode_syn1,encode_syn2 = self.sess.run(
            [self.output_syn1, self.output_syn2,self.output_en,self.encode_syn_1,self.encode_syn_2],
            feed_dict={self.batch_data: sample_batch,
                       self.input_label: idlabel_batch,
                       self.input_pose: label_batch[1],
                       self.index: sample_batch.shape[0]})  # falselable无所谓了
        score_identity=np.concatenate([utils.compare_pair_features(np.reshape(encode_real,[-1,512]),np.reshape(encode_syn1,[-1,512])),utils.compare_pair_features( np.reshape(encode_real,[-1,512]),np.reshape(encode_syn2,[-1,512]))],axis=0)
        utils.write_batch(self.result_path, 1, sample_data, sample_batch, epoch_n,
                          interval_i, othersample=sample_data_ex, ifmerge=True,score_f_id=score_identity)
        logging.info('[score_identity] {:08} {}'.format(step,score_identity))

        # pose - invariance测试
        for idx in xrange(sample_count):  # 将数据集中的同一个人所有!!!角度！！！照片都跑一次
            tppn = self.test_batch_size
            label_batch_sub = [sample_count] * tppn  # 没用凑齐8 为了后面的split
            pose_batch = range(0, tppn)
            tmp_batch = np.tile(sample_batch[idx], (tppn, 1, 1, 1)). \
                reshape(tppn, sample_batch.shape[1], sample_batch.shape[2],
                        sample_batch.shape[3])
            sample_data, sample_data_ex, encode_real, encode_syn1, encode_syn2 = self.sess.run([self.output_syn1, self.output_syn2,self.output_en,self.encode_syn_1,self.encode_syn_2],
                                                   feed_dict={
                                                       self.batch_data: tmp_batch,
                                                       self.input_label: label_batch_sub,
                                                       self.input_pose: pose_batch,
                                                       self.index: tmp_batch.shape[0]})
            score_pose=np.concatenate([utils.compare_pair_features(np.reshape(encode_real,[-1,512]),np.reshape(encode_syn1,[-1,512])),utils.compare_pair_features( np.reshape(encode_real,[-1,512]),np.reshape(encode_syn2,[-1,512]))],axis=0)
            utils.write_batch(self.result_path, 2, sample_data, tmp_batch, epoch_n,
                              interval_i, sample_idx=idx, othersample=sample_data_ex, ifmerge=True,score_f_id=score_pose)
            logging.info('[score_pose] {:08} {}'.format(step, score_pose))

    def slerp_interpolation(self,batch_image,batch_label,epoch,interval):
        # 球面差值法测试
        # for _ in xrange(3):#测试三次
        # print 'slerp test!!'
        index_select = np.random.randint(0, batch_image.shape[0], self.test_slerp_count*2)
        encode_512 = self.sess.run(self.encode_slerp,
                             feed_dict={self.batch_data: batch_image[index_select],
                                        self.input_label: batch_label[0][index_select],
                                        self.input_pose: batch_label[1][index_select],
                                        self.index: batch_image[index_select].shape[0]})
        pose = np.asarray(batch_label[1][index_select], np.int32)
        noise = np.random.uniform(low=-1.0, high=1.0, size=[self.test_slerp_count*2, self.noise_z])

        b_1 = np.zeros((pose.size, self.pose_c), dtype=np.float32)
        b_1[np.arange(pose.size), pose] = 1

        encode_525 = np.concatenate([np.reshape(encode_512, [-1, 512]), b_1], axis=1)
        encode_575 = np.concatenate([encode_525, noise], 1)
        encode_sub = np.split(encode_575,2,axis=0)
        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([utils.slerp(ratio, r1, r2) for r1, r2 in
                          zip(encode_sub[0], encode_sub[1])])
            z = np.reshape(z, [-1, 1, 1, 512 + self.noise_z + self.pose_c])
            z_decode = self.sess.run(self.image_syn_slerp,
                                feed_dict={self.slerp_code: z,
                                           self.batch_data: batch_image,
                                           self.input_label: batch_label[0],
                                           self.input_pose: batch_label[1],
                                           self.index: batch_image.shape[0]})
            decodes.append(z_decode)
        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        index_sub=np.split(index_select,2,axis=0)
        for idx, img in enumerate(decodes):
            img = np.concatenate([[batch_image[index_sub[0][idx]]], img,
                                  [batch_image[index_sub[1][idx]]]], 0)
            img = utils.inverse_transform(img)[:, :, :, ::-1]
            utils.save_image(img, os.path.join('./{}'.format(self.result_path),
                                               'test{:08}_interp_G_{:01}.png'.format(epoch,interval )), nrow=10 + 2)

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
                #D
                _ =self.sess.run(self.train_d_op,
                        feed_dict={self.batch_data:batch_image,
                                   self.input_label:batch_label[0],
                                   self.input_pose:batch_label[1],
                                   self.index: batch_image.shape[0]})

                #G
                for _ in xrange(self.g_loop):
                    _ = self.sess.run(self.train_g_op,
                    feed_dict={self.batch_data: batch_image,
                               self.input_label: batch_label[0],
                               self.input_pose: batch_label[1],
                               self.index: batch_image.shape[0]})
                # if interval_i%10:
                sample_data, sample_data_ex,encode_real,encode_syn1,encode_syn2 , loss_d, loss_g, train_summary,\
                data_ex,data,step\
                    = self.sess.run(
                    [self.output_syn1, self.output_syn2,self.output_en,self.encode_syn_1,self.encode_syn_2,self.d_loss, self.g_loss, self.summary_train,
                     self.input_data_ex,self.batch_data,self.global_step],
                    feed_dict={self.batch_data: batch_image,
                               self.input_label: batch_label[0],
                               self.input_pose: batch_label[1],
                               self.index: batch_image.shape[0]})
                self.summary_write.add_summary(train_summary,global_step=step)

                logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, d_loss:%.4f, g_loss:%.4f'\
                %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss_d,loss_g))

                if (curr_interval)%int(self.sample_interval*self.batch_idxs)==0:
                    #记录训练数据
                    score_train= np.concatenate([utils.compare_pair_features(np.reshape(encode_real,[-1,512]),np.reshape(encode_syn1,[-1,512])),
                                                 utils.compare_pair_features( np.reshape(encode_real,[-1,512]),np.reshape(encode_syn2,[-1,512]))],axis=0)
                    logging.info('[score_train] {:08} {}'.format(step, score_train))
                    utils.write_batch(self.result_path,0,sample_data,batch_image,epoch_n,interval_i,othersample=sample_data_ex,ifmerge=True,score_f_id=score_train)
                    self.validation(interval_i,epoch_n,step)

                # slerp
                if (curr_interval) % int(self.test_interval * self.batch_idxs) == 0:
                    self.slerp_interpolation(batch_image,batch_label,epoch_n,interval_i)
                    if self.ifsave and curr_interval != 0:
                        self.saver.save(self.sess,
                                   os.path.join(self.check_point_path, self.model_name),
                                   global_step=step)
                        print '*' * 20 + 'save model successed!!!!~~~~'
                curr_interval+=1

    def _get_train_op(self,global_step):
        '''
        梯度计算
        :param global_step: 迭代次数
        :return:
        '''
        optimizer_d = tf.train.AdamOptimizer(learning_rate=self.d_lr,beta1=self.beta1)
        grads_and_var_d = optimizer_d.compute_gradients(self.d_loss,self.varsd,colocate_gradients_with_ops = True)
        grads_d,vars_d = zip(*grads_and_var_d)
        grads_d,_ =tf.clip_by_global_norm(grads_d,0.1)
        self.train_d_op = optimizer_d.apply_gradients(zip(grads_d,vars_d),global_step)
        optimizer_g = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=self.beta1)
        grads_and_var_g = optimizer_g.compute_gradients(self.g_loss,self.varsg,colocate_gradients_with_ops = True)
        grads_g ,var_g = zip(*grads_and_var_g)
        grads_g , _ = tf.clip_by_global_norm(grads_g,0.1)
        self.train_g_op = optimizer_g.apply_gradients(zip(grads_g,var_g))
        return global_step

    def _predict_drgan_multipie(self,reuse=False):
        '''
        网络训练
        :param reuse: True | False
        :return:
        '''
        _,self.logits,output_en = nets.inference_recognition(self.batch_data,self.class_nums,reuse=reuse)
        self.output_en = tf.reshape(output_en,[-1,1,1,512])

        pose_reshape = tf.reshape(self.pose, [-1, 1, 1, self.pose_c])
        pose_float = tf.cast(pose_reshape, dtype=tf.float32)
        pose_add_noise=tf.concat([pose_float,self.noise],3)
        sample_add_pn = tf.concat([self.output_en,pose_add_noise ], 3)
        self.output_syn1 = nets.netG_deconder_gamma(sample_add_pn, self.output_channel,reuse=reuse)

        #pose reverse and concat pose和noise 分开
        pose_reverse_reshape = tf.reshape(self.pose_reverse, [-1, 1, 1, self.pose_c])
        pose_reverse_float=tf.cast(pose_reverse_reshape, dtype=tf.float32)
        pose_add_noise_reverse=tf.concat([pose_reverse_float,self.noise_reverse],3)

        #pose和 noise绑定
        sample_add_pn_ex = tf.concat([self.output_en, pose_add_noise_reverse], 3)
        self.output_syn2 = nets.netG_deconder_gamma(sample_add_pn_ex, self.output_channel, reuse=True)

        self.ppsr_logits\
            = \
            nets.Custom_netD_discriminator_psloss(self.batch_data, posenum=self.pose_c,reuse=reuse)
        self.reallogits=nets.Custom_netD_discriminator_adloss(self.batch_data,reuse=reuse)

        self.pidf1_softmax, self.pidf1logits,pidf1content \
            = \
            resnet_yd(self.output_syn1[:,:,:,::-1],reuse=True)
        self.pidf1content=tf.reshape(pidf1content,[-1,1,1,512])
        self.ppsf1_logits\
            = \
            nets.Custom_netD_discriminator_psloss(self.output_syn1, posenum=self.pose_c, reuse=True)
        self.fake1logits = nets.Custom_netD_discriminator_adloss(self.output_syn1,reuse=True)

        self.pidf2_softmax, self.pidf2logits,pidf2content \
            = \
            resnet_yd(self.output_syn2[:,:,:,::-1],reuse=True)
        self.pidf2content = tf.reshape(pidf2content, [-1, 1, 1, 512])
        self.ppsf2_logits\
            = \
            nets.Custom_netD_discriminator_psloss(self.output_syn2, posenum=self.pose_c, reuse=True)
        self.fake2logits = nets.Custom_netD_discriminator_adloss(self.output_syn2,reuse=True)


        self.encode_syn_1, self.encode_syn_2 = resnet_yd(self.output_syn1[:,:,:,::-1], reuse=True), resnet_yd(
            self.output_syn2[:,:,:,::-1], reuse=True)

    def _loss_gan_multipie(self):
        '''
        loss 计算
        :return:
        '''
        with tf.name_scope('D_loss'):
            # pose loss
            self.ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.pose, logits=self.ppsr_logits))
            #adversarial
            self.ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.reallogits),logits=self.reallogits
            ))
            self.ad_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.fake1logits),logits=self.fake1logits
            ))
            self.ad_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.fake2logits),logits=self.fake2logits
            ))
            self.split_content = tf.split(self.output_en, 2, axis=0)[::-1]
            self.constraint_loss=tf.reduce_mean(tf.abs(tf.subtract(self.split_content[0], self.split_content[1])))
            self.id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels,logits=self.logits))

        with tf.name_scope('G_loss'):
            #ps loss
            self.input_pose_ex = tf.concat(tf.split(self.pose, 2, axis=0)[::-1], axis=0)
            self.ps_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.pose, logits=self.ppsf1_logits))
            self.ps_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.input_pose_ex, logits=self.ppsf2_logits))

            #identity preserved loss
            self.id_preserve_f1 = tf.reduce_mean(tf.abs(tf.subtract(self.output_en, self.pidf1content)))
            self.id_preserve_f2 = tf.reduce_mean(tf.abs(tf.subtract(self.output_en, self.pidf2content)))
            #认为交换了位置一样有很强的相似性

            #adversarial
            self.ad_loss_syn1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.fake1logits),logits=self.fake1logits
            ))
            self.ad_loss_syn2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.fake2logits),logits=self.fake2logits
            ))
            # total variation denoising
            shape = self.batch_data.shape.as_list()
            tv_y_size = _tensor_size(self.output_syn1[0, 1:, :, :])*self.batch_size*self.gpus_count
            tv_x_size = _tensor_size(self.output_syn1[0, :, 1:, :])*self.batch_size*self.gpus_count
            self.tv_loss_f1 = TOTAL_VARIATION_SMOOTHING * 2 * (
                (tf.nn.l2_loss(self.output_syn1[:, 1:, :, :] - self.output_syn1[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(self.output_syn1[:, :, 1:, :] - self.output_syn1[:, :, :shape[2] - 1, :]) /
                 tv_x_size))
            self.tv_loss_f2 = TOTAL_VARIATION_SMOOTHING * 2 * (
                (tf.nn.l2_loss(self.output_syn2[:, 1:, :, :] - self.output_syn2[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(self.output_syn2[:, :, 1:, :] - self.output_syn2[:, :, :shape[2] - 1, :]) /
                 tv_x_size))
            #piexl
            self.input_data_ex = tf.concat(tf.split(self.batch_data, 2, axis=0)[::-1], axis=0)
            self.pixel_loss_f1 = tf.reduce_mean(tf.squared_difference(self.batch_data, self.output_syn1))
            self.pixel_loss_f2 = tf.reduce_mean(tf.squared_difference(self.input_data_ex , self.output_syn2))
        with tf.name_scope('loss_d'):
            tf.summary.scalar('ps_loss_d', self.ps_loss_real)
            tf.summary.scalar('constraint_loss',self.constraint_loss)
            tf.summary.scalar('ad_loss_real', self.ad_loss_real)
            tf.summary.scalar('ad_loss_fake1', self.ad_loss_fake1)
            tf.summary.scalar('ad_loss_fake2', self.ad_loss_fake2)
            tf.summary.scalar('id_loss_real',self.id_loss_real)
        with tf.name_scope('loss_g'):
            tf.summary.scalar('idpresv_loss_syn1', self.id_preserve_f1)
            tf.summary.scalar('idpresv_loss_syn2', self.id_preserve_f2)
            tf.summary.scalar('ps_loss_syn1', self.ps_loss_syn1)
            tf.summary.scalar('ps_loss_syn2', self.ps_loss_syn2)
            tf.summary.scalar('ad_loss_syn1', self.ad_loss_syn1)
            tf.summary.scalar('ad_loss_syn2', self.ad_loss_syn2)
            tf.summary.scalar('pixel_loss_syn1', self.pixel_loss_f1)
            tf.summary.scalar('pixel_loss_syn2', self.pixel_loss_f2)
        if True:
            tf.summary.image('image0/input1', tf.expand_dims(self.batch_data[0][:,:,::-1], 0))
            tf.summary.image('image0/input2', tf.expand_dims(self.batch_data[0+self.batch_size/2][:, :, ::-1], 0))
            tf.summary.image('image0/decoder', tf.expand_dims(self.output_syn1[0][:,:,::-1], 0))
            tf.summary.image('image0/decoder_re', tf.expand_dims(self.output_syn2[0][:,:,::-1], 0))

            tf.summary.image('image1/input1', tf.expand_dims(self.batch_data[1][:,:,::-1], 0))
            tf.summary.image('image1/input2', tf.expand_dims(self.batch_data[1+self.batch_size/2][:, :, ::-1], 0))
            tf.summary.image('image1/decoder', tf.expand_dims(self.output_syn1[1][:,:,::-1], 0))
            tf.summary.image('image1/decoder_re', tf.expand_dims(self.output_syn2[1][:,:,::-1], 0))

            tf.summary.image('image2/input1', tf.expand_dims(self.batch_data[2][:,:,::-1],0))
            tf.summary.image('image2/input2', tf.expand_dims(self.batch_data[2+self.batch_size/2][:, :, ::-1], 0))
            tf.summary.image('image2/decoder', tf.expand_dims(self.output_syn1[2][:,:,::-1], 0))
            tf.summary.image('image2/decoder_re', tf.expand_dims(self.output_syn2[2][:,:,::-1], 0))
    def _loss_compute(self):#G:D 2:1
        '''
        loss 加权
        :return:
        '''
        self.d_loss =1* (self.ad_loss_real +
                     0.5*(self.ad_loss_fake1+self.ad_loss_fake2))
        # self.d_loss= 0.1 * self.id_loss_real
        # self.g_loss = (self.lr_alpha)*20*self.pixel_loss_f1+(1-self.lr_alpha)*20*self.pixel_loss_f2
        self.g_loss =(self.lr_alpha)*(20*self.pixel_loss_f1+self.ad_loss_syn1+0.5*self.id_preserve_f1)+\
                     (1-self.lr_alpha)*(20*self.pixel_loss_f2+self.ad_loss_syn2+0.5*self.id_preserve_f2)
        tf.summary.scalar('losstotal/total_loss_d', self.d_loss)
        tf.summary.scalar('losstotal/total_loss_g', self.g_loss)

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)






