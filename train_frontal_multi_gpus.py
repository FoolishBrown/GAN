#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
# import skimage
# import datetime
import tensorflow.contrib.slim as slim
import os
import cv2
import re
import sys
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
# sys.path.insert(0,'./test_for_lfw')
import test_main_lfw
TOTAL_VARIATION_SMOOTHING = 1e2

class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,lr_alpha=0.5,lr_beta=0.5,lr_gamma=1,version='0',
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,test_interval=1,test_batch=8,
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
        self.test_batch=test_batch
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
        self.gpus_count=1
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

        if self.restored:
            # self.saver.restore(self.sess,self.check_point)
            saver_init.restore(self.sess,self.check_point)

        self.summary_write = tf.summary.FileWriter(self.summarypath + '/' + self.version+'_'+self.gpus_list, self.sess.graph)

        self.data_loader_valid.enqueueStart()  # 开启测试队列
        self.data_loader_train.Dual_enqueueStart(frontalization=True)#开启训练对别

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
        self.input_data,self.gt_input_data=tf.split(self.batch_data,2,axis=0)
        self.input_label = tf.split(self.batch_label,2,axis=0)[0]
       #mk onehot labels
        self.labels = slim.one_hot_encoding(self.input_label,self.class_nums)
        #comput loss
        self._predict_drgan_multipie()

        self._loss_gan_multipie()
        self._loss_compute()

        self.summary_train = tf.summary.merge_all()
        #select var list
        train_vars = tf.trainable_variables()
        self.varsg = [var for var in train_vars if 'generator' in var.name]
        self.varsd = [var for var in train_vars if 'discriminator' in var.name]
        self.fc_add = [var for var in train_vars if 'recognation_fc' in var.name]

        self.vard_fr= [var for var in train_vars if 'resnet_yd' in var.name]
        # self.init_vars=self.vard_fr+self.varsd+self.varsg+self.fc_add
        self.init_vars=self.vard_fr
        self.varsg+=self.vard_fr
        self.varsg+=self.fc_add
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
            _,_,self.encode_slerp = nets.inference_recognition(self.input_data,self.class_nums, reuse=True)
            self.encode_slerp_z=tf.get_variable('code',[3,562])
            self.image_syn_slerp = nets.netG_deconder_gamma(self.encode_slerp_z, self.output_channel, reuse=True)

    def validation(self,interval_i,epoch_n,step):
        '''
        inference
        :return:
        '''

        # sample_batch = np.zeros([self.test_batch_size, self.input_size, self.input_size, self.input_channel],
        #                         np.float32)
        # label_batch = np.zeros([self.data_loader_train.labels_nums, self.test_batch_size], np.float32)
        # 加载测试batch
        # someone=np.random.randint(0,self.data_loader_valid.class_nums)
        # for s_i in xrange(1):
        #     id =np.random.randint(0,self.data_loader_valid.class_nums,1)
        #     sample_batch[s_i * self.pose_c:(s_i + 1) * self.pose_c, :, :, :], label_batch[:, s_i * self.pose_c:(s_i + 1) * self.pose_c] = self.data_loader_valid.oneperson_allpose(id)  # 得到一个人所有的图片
        sample_batch = np.zeros(
            [self.batch_size * self.gpus_count, self.input_size, self.input_size, self.input_channel], np.float32)
        sample_label = np.zeros([self.data_loader_valid.labels_nums, self.batch_size * self.gpus_count], np.float32)
        for b_i in xrange(self.gpus_count):
            sample_batch[b_i * self.batch_size:(b_i + 1) * self.batch_size, :, :, :], \
            sample_label[:,b_i * self.batch_size:(b_i + 1) * self.batch_size] = self.data_loader_valid.read_data_batch()
        # identity-preserved 测试
        # idlabel_batch = [0] * sample_count
        sample_data,encode_syn,encode_real= self.sess.run(
            [self.output_syn,self.cosine_syn,self.cosine_real ],
            feed_dict={self.batch_data: sample_batch,
                       self.batch_label: sample_label[0]})
        score_identity = np.concatenate([encode_syn,encode_real],axis=0)
            # [utils.compare_pair_features(np.reshape(encode_syn, [-1, 512]), np.reshape(encode_syn2, [-1, 512])),
            #  utils.compare_pair_features(np.reshape(encode_front, [-1, 512]), np.reshape(encode_profile, [-1, 512]))],
            # axis=0)
        sample_batch=np.split(sample_batch,2,axis=0)
        utils.write_batch(self.result_path, 1, sample_data, sample_batch[1], epoch_n,
                          interval_i, ifmerge=True,score_f_id=score_identity,othersample=sample_batch[0],reverse_other=False)
        logging.info('[score_identity] {:08} {}'.format(step,score_identity))

    def slerp_interpolation(self,batch_image,batch_label,epoch,interval):
        # 球面差值法测试
        # for _ in xrange(3):#测试三次
        # print 'slerp test!!'
        index_select = np.random.randint(0, batch_image.shape[0], self.test_slerp_count*2*2)
        encode_512 = self.sess.run(self.encode_slerp,
                             feed_dict={self.batch_data: batch_image[index_select],
                                        self.input_label: batch_label[0][index_select]})
        encode_50=np.random.uniform(high=1, low=-1, size=[self.test_slerp_count*2,self.noise_z])
        encode_562=np.concatenate([encode_512,encode_50],axis=1)
        encode_sub = np.split(encode_562,2,axis=0)
        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([utils.slerp(ratio, r1, r2) for r1, r2 in
                          zip(encode_sub[0], encode_sub[1])])
            z = np.reshape(z, [-1,562])
            z_decode = self.sess.run(self.image_syn_slerp,
                                feed_dict={self.encode_slerp_z: z})
            decodes.append(z_decode)
        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        index_sub=np.split(index_select,2,axis=0)
        index_sub=np.split(index_sub[0],2,axis=0)
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
                _ ,loss_d=self.sess.run([self.train_d_op,self.d_loss],
                        feed_dict={self.batch_data:batch_image,
                                   self.batch_label:batch_label[0]})

                #G
                for _ in xrange(self.g_loop):
                    _ = self.sess.run(self.train_g_op,
                    feed_dict={self.batch_data: batch_image,
                               self.batch_label: batch_label[0]})
                # if interval_i%10:
                sample_data,  loss_g, train_summary,\
                data,step, \
                encode_real, encode_syn \
                    = self.sess.run(
                    [self.output_syn,self.g_loss, self.summary_train,
                    self.input_data,self.global_step,self.cosine_real,self.cosine_syn],
                    feed_dict={self.batch_data: batch_image,
                               self.batch_label: batch_label[0]})
                self.summary_write.add_summary(train_summary,global_step=step)

                logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, d_loss:%.4f, g_loss:%.4f'\
                %(epoch_n,self.epoch,self.gpus_list,step,(time.time()-start_time)/3600.0,loss_d,loss_g))

                if (curr_interval)%int(self.sample_interval*self.batch_idxs)==0:
                    #记录训练数据
                    score_train = np.concatenate([encode_syn,encode_real],axis=0)
                    logging.info('[score_train] {:08} {}'.format(step, score_train))
                    batch_image=np.split(batch_image,2,axis=0)
                    print sample_data.shape
                    utils.write_batch(self.result_path,0,sample_data,batch_image[1],epoch_n,interval_i,othersample=batch_image[0],reverse_other=False,ifmerge=True,score_f_id=score_train)
                    self.validation(interval_i,epoch_n,step)

                # slerp
                if (curr_interval) % int(self.test_interval * self.batch_idxs) == 0:
                    self.slerp_interpolation(batch_image[1],batch_label,epoch_n,interval_i)
                    if self.ifsave and curr_interval!=0:
                        modelname=self.model_name+'-'+str(step)

                        self.saver.save(self.sess,
                                   os.path.join(self.check_point_path, self.model_name),
                                   global_step=step)
                        synthesis.main(os.path.join(self.check_point_path,modelname))
                        print '*'*20+'synthesis image finished!!!~~~~'
                        f_write=open(self.check_point_path+'/lfwtest_info.txt','a')
                        mAP,mTd,mTh=test_main_lfw.main(modelpath=self.check_point_path,model=modelname)
                        f_write.write("{},{},{}\n".format(mAP,mTd,mTh))
                        f_write.close()
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
        with    tf.device('/gpu:1'):
            self.softresult,self.logits_all,self.encode = nets.inference_recognition(self.batch_data,classnums=self.class_nums,reuse=reuse)
            #
            self.logits=tf.split(self.logits_all,2,axis=0)[0]
            # self.logits,self.gt_logits= tf.split(self.logits,2,axis=0)
            #
            # self.encode, self.gt_encode= tf.split(self.encode, 2, axis=0)
            #
            # self.softresult, self.gt_softresult = tf.split(self.softresult, 2, axis=0)
            #syn1
            noise = tf.random_uniform(shape=(self.batch_size*self.gpus_count,self.noise_z), minval=-1, maxval=1, dtype=tf.float32,
                                  name='input_noise')
            self.encode_add_z=tf.concat([self.encode,noise],1)

            # self.gt_encode_add_z=tf.concat([self.gt_encode,noise],1)
        with tf.device('/gpu:2'):
            self.output_syn_all = nets.netG_deconder_gamma(self.encode_add_z, self.output_channel,reuse=reuse)
            self.output_syn,self.output_gt_syn=tf.split(self.output_syn_all,2,axis=0)
            # self.output_gt_syn = nets.netG_deconder_gamma(self.gt_encode_add_z,self.output_channel,reuse=True)
        # with tf.device('/gpu:2'):
            self.real_logits=nets.Custom_netD_discriminator_adloss(self.gt_input_data,reuse=reuse)

            self.fake_logits = nets.Custom_netD_discriminator_adloss(self.output_syn,reuse=True)
        with tf.device('/gpu:1'):
            #正脸图片的feature
            self.pidprofile_content, self.pidr_content = tf.split(self.encode , 2, axis=0)

            #生成图像的feature
            self.fake_softmax,self.fake_logits_all,self.fake_content\
            = nets.inference_recognition(self.output_syn_all,classnums=self.class_nums,reuse=True)
            self.pidf_content,self.fake_gt_content=tf.split(self.fake_content,2,axis=0)
            self.f_logits = tf.split(self.fake_logits_all, 2, axis=0)[0]

            #计算cosine距离
            self.cosine_real=tf.divide(tf.reduce_sum(tf.multiply(self.pidprofile_content,self.pidr_content),axis=1),
                                       tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.pidr_content),axis=1)),
                                                   tf.sqrt(tf.reduce_sum(tf.square(self.pidprofile_content),axis=1))))
            self.cosine_syn=tf.divide(tf.reduce_sum(tf.multiply(self.pidf_content,self.fake_gt_content),axis=1),
                                       tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.pidf_content),axis=1)),
                                                   tf.sqrt(tf.reduce_sum(tf.square(self.fake_gt_content),axis=1))))

            self.cosine_real_residual=tf.divide(tf.reduce_sum(tf.multiply(self.pidprofile_content[1:,:],self.pidr_content[:-1,:]),axis=1),
                                       tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.pidr_content[:-1,:]),axis=1)),
                                                   tf.sqrt(tf.reduce_sum(tf.square(self.pidprofile_content[1:,:]),axis=1))))
            self.cosine_syn_residual=tf.divide(tf.reduce_sum(tf.multiply(self.pidf_content[1:,:],self.fake_gt_content[:-1,:]),axis=1),
                                       tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.pidf_content[1:,:]),axis=1)),
                                                   tf.sqrt(tf.reduce_sum(tf.square(self.fake_gt_content[:-1,:]),axis=1))))



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
            #predict loss
            self.pre_loss_syn=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels,logits=self.logits
            ))
            #real loss
            self.pre_loss_softmax=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels,logits=self.f_logits
            ))
            self.id_loss_psd=tf.reduce_mean(tf.abs(tf.subtract(self.pidr_content, self.pidf_content)))
            #identity preserved loss
            self.id_loss_real = tf.reduce_mean(tf.abs(tf.subtract(self.pidr_content, self.pidprofile_content)))
            self.id_loss_syn= tf.reduce_mean(tf.abs(tf.subtract(self.pidf_content, self.fake_gt_content)))
            #adversarial
            self.ad_loss_syn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.fake_logits),logits=self.fake_logits
            ))
            # total variation denoising
            shape = self.batch_data.shape.as_list()
            tv_y_size = _tensor_size(self.output_syn[0, 1:, :, :])*self.batch_size*self.gpus_count
            tv_x_size = _tensor_size(self.output_syn[0, :, 1:, :])*self.batch_size*self.gpus_count
            self.tv_loss_syn = TOTAL_VARIATION_SMOOTHING * 2 * (
                (tf.nn.l2_loss(self.output_syn[:, 1:, :, :] - self.output_syn[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(self.output_syn[:, :, 1:, :] - self.output_syn[:, :, :shape[2] - 1, :]) /
                 tv_x_size))
            #piexl
            self.pix_loss_syn = tf.reduce_mean(tf.abs(tf.subtract(self.gt_input_data, self.output_syn)))
            #symmetry
            self.output_syn_half_letf,self.output_syn_half_right=tf.split(self.output_syn,2,axis=2)
            self.sym_loss_syn = tf.reduce_mean(tf.abs(tf.subtract(self.output_syn_half_letf,self.output_syn_half_right[:,:,::-1,:])))

        with tf.name_scope('loss_d'):
            tf.summary.scalar('ad_loss_real', self.ad_loss_real)
            tf.summary.scalar('ad_loss_fake1', self.ad_loss_fake)
        with tf.name_scope('loss_g'):
            tf.summary.scalar('identity_predict_loss',self.pre_loss_syn)
            tf.summary.scalar('identity_synthesis_loss',self.pre_loss_softmax)
            tf.summary.scalar('idpresv_loss_syn', self.id_loss_syn)
            tf.summary.scalar('idpresv_loss_real',self.id_loss_real)
            tf.summary.scalar('tv_loss_syn', self.tv_loss_syn)
            tf.summary.scalar('sym_loss_syn', self.sym_loss_syn)
            tf.summary.scalar('ad_loss_syn2', self.ad_loss_syn)
            tf.summary.scalar('pix_loss_syn1', self.pix_loss_syn)
        with tf.name_scope('distance'):
            tf.summary.scalar('cosine_real', tf.reduce_mean(self.cosine_real))
            tf.summary.scalar('cosine_syn', tf.reduce_mean(self.cosine_syn))
            tf.summary.scalar('cosine_real_res', tf.reduce_mean(self.cosine_real_residual))
            tf.summary.scalar('cosine_syn_res', tf.reduce_mean(self.cosine_syn_residual))

    def _loss_compute(self):#G:D 2:1
        '''
        loss 加权
        :return:
        '''
        self.d_loss = 0.3*(self.ad_loss_real+self.ad_loss_fake)
        self.g_loss = self.ad_loss_syn*0.3+self.pix_loss_syn+0.003*self.id_loss_syn+\
                    self.sym_loss_syn*0.3+self.tv_loss_syn*0.0001+\
                    self.pre_loss_syn*0.003
        tf.summary.scalar('losstotal/total_loss_d', self.d_loss)
        tf.summary.scalar('losstotal/total_loss_g', self.g_loss)

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)






