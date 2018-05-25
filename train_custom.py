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

import Nets
from tqdm import trange


class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,lr_alpha=0.5,lr_beta=0.5,lr_gamma=1,version='0',
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,
                 data_loader_train=None,data_loader_valid=None,epoch=10000,log_dir='logdir',learning_rate=0.0002,beta1=0.5,
                 test_batch_size=10,gpus_list=None,write_type=0,model_name='dr_gan',noise_z=50,pose_c=7,random_seed=25,sample_pose_interval=10,
                 loss_type=0,Gloop=3,savepath='savename',imagepath='image',logfile='file.log',summary_dir='summary'):
        self.lr_alpha=lr_alpha
        self.lr_beta = lr_beta
        self.lr_gamma = lr_gamma
        self.losstype=loss_type
        self.restored=restored
        self.learning_rate=learning_rate
        self.beta1=beta1
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_size = output_size
        self.output_channel = output_channel
        self.sample_interval = sample_interval
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.sample_pose_interval = sample_pose_interval
        self.epoch = epoch
        #dir
        self.log_dir = log_dir
        self.savename=savepath
        self.result_path = imagepath
        self.check_point=check_point
        self.logfile=logfile
        self.summarypath=summary_dir

        self.gpus_list=gpus_list
        self.model_name=model_name
        self.write_type=write_type
        self.batch_idxs=self.data_loader_train.batch_idxs
        self.noise_z=noise_z
        self.pose_c=pose_c
        self.ifsave=ifsave
        self.Gloop=Gloop

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
        self.mkdir_result(self.check_point)#checkpoint dir

    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))


    def predict_drgan(self,batch_data,pose=None):
        with tf.name_scope('generator_encoder_decoder'):
            output_en = Nets.Custom_netG_encoder(batch_data)
            shape = output_en.get_shape().as_list()
            print shape
        #----------noise
            noise = tf.random_uniform(shape=(self.batch_size, 6, 6, 50), minval=-1, maxval=1, dtype=tf.float32,name='input_noise')
            ps_noise_map=Nets.Custom_netG_pose_and_noise(output_en,shape,pose,noise)

            # pose_sp=tf.split(pose,2,axis=0)
            # pose=tf.concat(pose_sp,axis=0)
            # ps_noise_map_exchange = Nets.Custom_netG_pose_and_noise(output_en, shape, pose, noise,reuse=True)
            # pose_1=tf.zeros_like(pose)
            # ps_noise_map_1=Nets.Custom_netG_pose_and_noise(output_en,shape,pose_1,noise,reuse=True)
            # pose_2=tf.ones_like(pose)
            # ps_noise_map_2=Nets.Custom_netG_pose_and_noise(output_en,shape,pose_2,noise,reuse=True)
            # pose_3=tf.ones_like(pose)*0.5
            # ps_noise_map_3=Nets.Custom_netG_pose_and_noise(output_en,shape,pose_3,noise,reuse=True)
            # size_noise=noise.shape.as_list()[0]
            output_de = Nets.Custom_netG_decoder(output_en,ps_noise_map)
            # output_de_exchange = Nets.Custom_netG_decoder(output_en,ps_noise_map_exchange,reuse=True)
            # output_de_zeros=Nets.Custom_netG_decoder(output_en,ps_noise_map_1,reuse=True)
            # output_de_halves=Nets.Custom_netG_decoder(output_en,ps_noise_map_3,reuse=True)
            # output_de_ones=Nets.Custom_netG_decoder(output_en,ps_noise_map_2,reuse=True)
        #
        # tf.summary.histogram('input_discrimintor/inputrealdata',batch_data)
        # tf.summary.histogram('input_discrimintor/inputsyndata',output_de)
        with tf.name_scope('discriminatortotal'):
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

            # softmax_id_fake_ex, idlogits_fake_ex = \
            # Nets.Custom_netD_discriminator_idloss(output_de_exchange, class_nums=self.data_loader_train.class_nums+1, reuse=True)
            #
            # pslogits_fake_ex, content_feature_fake_ex = \
            # Nets.Custom_netD_discriminator_psloss(output_de_exchange, reuse=True)
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
                output_de
               # softmax_id_fake_ex, idlogits_fake_ex, \
               # pslogits_fake_ex, content_feature_fake_ex,\
               # output_de,output_de_exchange,new_pose

        # softmax_ad_real, adlogits_real, \
            # softmax_ad_fake, adlogits_fake, \

            # softmax_ad_fake_ex,adlogits_fake_ex,\


                # output_de_zeros,output_de_halves,output_de_ones



    def loss_drgan_optimizer(self,input_data,input_pose,label,fakelabel):
        # get predict data
        softmax_id_real, idlogits_real, \
        pslogits_real, content_feature_real, \
        softmax_id_fake, idlogits_fake, \
        pslogits_fake, content_feature_fake, \
        output_de\
            = self.predict_drgan(input_data, pose=input_pose)
        count_n=tf.cast(self.batch_size,tf.float32)
        with tf.name_scope('Discriminator_loss'):
            id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=idlogits_real))
            id_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=fakelabel, logits=idlogits_fake))
            # id_loss_fake_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=fakelabel,logits=idlogits_fake_ex))
            ps_loss_real = tf.reduce_mean(tf.squared_difference(input_pose, pslogits_real))/count_n

        # with tf.name_scope('perceptual_loss'):
        #     feature_shape = tf.shape(content_feature_real)
        #     feature_size = tf.cast(feature_shape[1] * feature_shape[2] * feature_shape[3], dtype=tf.float32)
        #     feature_reconstruction_loss = tf.reduce_sum(
        #         tf.squared_difference(content_feature_real, content_feature_fake)) / feature_size

        with tf.name_scope('Generator_loss'):
        # Dloss = ad_loss_fake+ad_loss_real+id_loss_real+ps_loss_real
            id_loss_syn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=idlogits_fake))
            # id_loss_syn_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=label,logits=idlogits_fake_ex
            # ))
            ps_loss_syn = tf.reduce_mean(tf.squared_difference(input_pose, pslogits_fake))/count_n
            # ps_loss_syn_ex = tf.reduce_mean(tf.squared_difference(input_pose,pslogits_fake_ex))

        summary_train_lossD_id = tf.summary.scalar('lossD/id_loss_d_real', id_loss_real)
        summary_train_lossD_ps = tf.summary.scalar('lossD/ps_loss_d', ps_loss_real)
        summary_train_lossD_id = tf.summary.scalar('lossD/id_loss_d_fake', id_loss_fake)
        # summary_train_lossD_ps = tf.summary.scalar('lossD/id_loss_d_fake_ex', id_loss_fake_ex)

        summary_train_lossG_id = tf.summary.scalar('lossG/id_loss_g', id_loss_syn)
        summary_train_lossG_ps = tf.summary.scalar('lossG/ps_loss_g', ps_loss_syn)
        # summary_train_lossG_id = tf.summary.scalar('lossG/id_loss_g_ex', id_loss_syn_ex)
        # summary_train_lossG_ps = tf.summary.scalar('lossG/ps_loss_g_ex', ps_loss_syn_ex)

        with tf.name_scope('compute_loss'):
            # Dloss=self.lr_gamma*id_loss_real+self.lr_alpha*(id_loss_fake+id_loss_fake_ex)+ps_loss_real
            # Gloss_warmup=self.lr_gamma*id_loss_syn+self.lr_alpha*(id_loss_syn+id_loss_syn_ex)*self.warmup+self.lr_beta*(ps_loss_syn+ps_loss_syn_ex)*self.warmup
            # Gloss=self.lr_gamma*id_loss_syn+self.lr_alpha*(id_loss_syn+id_loss_syn_ex)+self.lr_beta*(ps_loss_syn+ps_loss_syn_ex)
        # return Dloss,Gloss,Gloss_warmup,output_de,output_de_exchange
            Dloss = self.lr_gamma * id_loss_real + self.lr_alpha * id_loss_fake + self.lr_beta *
            Gloss = self.lr_gamma * id_loss_syn + self.lr_alpha * id_loss_syn + self.lr_beta * ps_loss_syn
        return Dloss, Gloss, output_de
    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpus_list
        # tf.set_random_seed(20)
        with tf.Graph().as_default():
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
            input_label = tf.placeholder(dtype= tf.int32,shape=[None],name='input_labels')#label
            input_pose = tf.placeholder(dtype= tf.float32,shape=[None,3],name='input_poses')#pose
            input_pose_re=tf.reshape(input_pose,[-1,1,1,3])
            index = tf.placeholder(tf.int32, None,name='input_nums')
            #mk onehot labels
            labels = slim.one_hot_encoding(input_label,self.data_loader_train.class_nums+1)
            fakelabel=tf.zeros_like(input_label)+self.data_loader_train.class_nums
            fakelabels=slim.one_hot_encoding(fakelabel,self.data_loader_train.class_nums+1)
            # poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
            #test accuracy:
            # with tf.name_scope('accurary'):
            #     reshape_R=tf.reshape(predict_r_id, [-1, self.data_loader_train.class_nums])
            #     max_r=tf.argmax(reshape_R,1)
            #     label_true=tf.argmax(labels,1)
            #     correct_pred_r = tf.equal(max_r,label_true)
            #     reshape_F=tf.reshape(predict_f_id, [-1, self.data_loader_train.class_nums])
            #     max_f=tf.argmax(reshape_F, 1)
            #     correct_pred_f = tf.equal(max_f,label_true)
            #     accuracy_r = tf.reduce_mean(tf.cast(correct_pred_r,tf.float32))
            #     accuracy_f = tf.reduce_mean(tf.cast(correct_pred_f,tf.float32))
            Dloss,Gloss,\
            sample\
                =self.loss_drgan_optimizer(input_data, input_pose_re, labels, fakelabels)
            # summary_train_accracy_r = tf.summary.scalar('accuracy/real_data',accuracy_r)
            # summary_train_accracy_f = tf.summary.scalar('accuracy/fake_data',accuracy_f)
            summary_train_lossD = tf.summary.scalar('loss/total_loss_d',Dloss)
            summary_train_lossG = tf.summary.scalar('loss/total_loss_g',Gloss)



            # summary_train_image_batch = tf.summary.image('image/input',tf.expand_dims(input_data[0],0))
            summary_train_image_batch = tf.summary.image('image0/input',tf.expand_dims(input_data[0],0))
            summary_train_image_sample = tf.summary.image('image0/decoder',tf.expand_dims(sample[0],0))

            summary_train_image_batch = tf.summary.image('image1/input', tf.expand_dims(input_data[1], 0))
            summary_train_image_sample = tf.summary.image('image1/decoder', tf.expand_dims(sample[1], 0))

            summary_train_image_batch = tf.summary.image('image2/input', tf.expand_dims(input_data[2], 0))
            summary_train_image_sample = tf.summary.image('image2/decoder', tf.expand_dims(sample[2], 0))
            summary_train = tf.summary.merge_all()
            # summary_train = tf.summary.merge([summary_train_lossD,summary_train_lossG,summary_train_lossD_ad_r,summary_train_lossD_ad_r,summary_train_lossD_id,
            #                                   summary_train_lossG_syn,summary_train_lossG_id])
            train_vars = tf.trainable_variables()
            self.varsG = [var for var in train_vars if 'generator' in var.name]
            self.varsD = [var for var in train_vars if 'discriminator' in var.name]

            # train_op = self.get_train_op(Dloss,Gloss,global_step,self.learning_rate,self.beta1)
            # sample_valid = self.validation_drgan(input_data, index, noise=noise[:index], pose=poses)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True

            with tf.Session(config=config) as sess:

                for learning_rate in [self.learning_rate]:
                    curr_interval=0
                    with tf.name_scope('train_optimizer'):
                        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=self.beta1,
                                                           name='optimizer')
                        # batch norm,when trainingm the moving_mean and moving_variance need to be updated!!!!!!
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            train_op_d = optimizer.minimize(Dloss, var_list=self.varsD)
                            # train_op_g_warmup = optimizer.minimize(Gloss_warmup,var_list=self.varsG)
                            train_op_g = optimizer.minimize(Gloss, var_list=self.varsG)
                    try:
                        tf.global_variables_initializer().run()
                    except:
                        tf.initialize_all_variables().run()
                    # ,keep_checkpoint_every_n_hours=3
                    saver = tf.train.Saver(max_to_keep=10)
                    step = 0
                    if self.restored:
                        saver.restore(sess, self.check_point)
                        step = int(next(re.finditer("(\d+)(?!.*\d)", self.check_point)).group(0))
                    start_time = time.time()
                    self.data_loader_train.enqueueStart()
                    hyperparameter=self.savename+'drgan'
                    summary_write = tf.summary.FileWriter(self.log_dir+'/'+self.savename, sess.graph)
                    # projector.visualize_embeddings(summary_write,config2)
                    for epoch_n in xrange(self.epoch):

                        # coord = tf.train.Coordinator()
                        # threads = tf.train.start_queue_runners(coord=coord,sess=sess)
                        for interval_i in trange(self.batch_idxs):
                            # if i%10 ==0:
                            bathstart=time.time()
                            #read data labels
                            batch_image,batch_label=self.data_loader_train.read_data_batch()
                            batch_label=np.transpose(batch_label,[1,0])
                            print batch_label.shape
                            batchend=time.time()
                            # batch_label[1][]-=3
                            print 'INFO: read a batch images use {:.4f}s'.format(batchend-bathstart)

                            print 'INFO: made noise code'
                            # batchnoise = np.random.uniform(-1, 1, size=(self.batch_size, 1,1, self.noise_z))
                            #optimizer D
                            _=sess.run(train_op_d,
                                        feed_dict={input_data:batch_image,
                                                   input_label:batch_label[:,0],
                                                   input_pose:batch_label[:,1:4],
                                                   index: batch_image.shape[0]})
                            # print 'imageszie',im.shape,im2.shape
                            # break
                            # #optimizer G
                            sample_, loss_d, loss_g, train_summary=None,None,None,None
                            for _ in xrange(self.Gloop):
                                # if epoch_n <0:
                                #     _ = sess.run(train_op_g_warmup,
                                #                  feed_dict={input_data: batch_image,
                                #                             input_label: batch_label[0],
                                #                             input_pose: batch_label[1],
                                #                             index: batch_image.shape[0]})
                                # else:
                                sample_, loss_d, loss_g, train_summary = sess.run(
                                    [sample, Dloss, Gloss, summary_train],
                                    feed_dict={input_data: batch_image,
                                           input_label: batch_label[:,0],
                                           input_pose: batch_label[:,1:4],
                                           index: batch_image.shape[0]})
                            summary_write.add_summary(train_summary,global_step=step)

                            step+=1

                            handledtime=time.time()
                            print 'INFO: handle a batch use {:.2f}!!!'.format(handledtime-batchend)


                            print '[gpu%s]Epoch [%4d/%4d] [%4d/%4d] [global_step:%d]time:%.4f, d_loss:%.8f, g_loss:%.8f'\
                            %(self.gpus_list,epoch_n,self.epoch,interval_i,self.batch_idxs,step,time.time()-start_time,loss_d,loss_g)

                            if (curr_interval)%int(self.sample_interval*self.batch_idxs)==0:
                                print 'INFO: handle batch...'
                                self.write_batch(0,sample_,batch_image,epoch_n,interval_i)

                                # sample_batch , identity =self.data_loader_train.get_same_person_batch(self.random_seed)
                                # sample_count = sample_batch.shape[0]
                                pose_batch=np.zeros_like(batch_label[:,1:4],dtype=np.int)
                                print 'INFO: writing pose code testing ...'
                                # label_batch = [identity] * self.pose_c
                                if (curr_interval)%int(self.sample_pose_interval * self.batch_idxs)==0:
                                    # pose_batch =np.arange(-(self.pose_c/2),self.pose_c/2+1)+6

                                    # for idx in xrange(sample_count):
                                        # print sample_batch.shape,sample_count
                                        # tmp_batch = np.tile(sample_batch[idx],(self.pose_c,1,1,1)).\
                                    #     reshape(self.pose_c,sample_batch.shape[1],sample_batch.shape[2],sample_batch.shape[3])
                                    sample_data_= sess.run(sample,
                                                                       feed_dict={
                                                                           input_data: batch_image,
                                                                           input_label:batch_label[:,0],
                                                                           input_pose: pose_batch,
                                                                           index: self.pose_c})

                                    self.write_batch(2,sample_data_,batch_image,epoch_n,interval_i)

                                # #save model
                                if self.ifsave:
                                    saver.save(sess, os.path.join(self.check_point, self.model_name + '.ckpt'),
                                               global_step=step)
                                    print '*'*20+'save model successed!!!!~~~~'
                            curr_interval+=1

                    # coord.request_stop()
                    # coord.join()


#------------------------------------------write img-----------------
    '''
    def inference(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus_list

        tf.set_random_seed(20)
        # with tf.Graph().as_default():
        input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel])#image
        # input_data = tf.random_crop(src_data,[-1,self.output_size,self.output_size,self.input_channel])
        input_label = tf.placeholder(dtype= tf.int32,shape=[None])#label
        input_pose = tf.placeholder(dtype= tf.int32,shape=[None])#pose
        index = tf.placeholder(tf.int32, None)
        #mk onehot labels
        labels = slim.one_hot_encoding(input_label,self.data_loader_train.class_nums)
        poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label

        noise = tf.random_uniform(shape=(index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32)
        sample_valid = self.validation_drgan(input_data, index, noise=noise[:index], pose=poses,reuse=True)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
            # ,keep_checkpoint_every_n_hours=3
            saver = tf.train.Saver()
            if self.restored:
                saver.restore(sess, self.check_point)
                sample_batch, pose_batch, identity = self.data_loader_train.get_same_person_batch(self.random_seed)
                pose_batch = np.arange(-(self.pose_c / 2), self.pose_c / 2 + 1) + 6
                sample_count = sample_batch.shape[0]
                for idx in xrange(sample_count):
                    # print sample_batch.shape,sample_count
                    tmp_batch = np.tile(sample_batch[idx], (self.pose_c, 1, 1, 1)). \
                        reshape(self.pose_c, sample_batch.shape[1], sample_batch.shape[2], sample_batch.shape[3])
                    sample_data = sess.run(sample_valid, feed_dict={
                                                           input_data: tmp_batch,
                                                           input_pose: pose_batch,
                                                           index: self.pose_c})
                    score_r_id = np.zeros([13,self.data_loader_train.class_nums])
                    score_f_id = np.zeros([13, self.data_loader_train.class_nums])

                    print score_f_id[0], np.sum(score_f_id), sum(score_f_id[0])
                    self.write_batch(False, sample_data, tmp_batch, -2,171005, score_r_id, score_f_id, identity,
                                     sample_idx=idx)
    '''
    def write_batch(self,type,sample,data,e_n,idx,score_r_id=None,score_f_id=None,identity=None,sample_idx=0):
        mainfold_h = int(np.ceil(np.sqrt(sample.shape[0])))
        # mainfold_w = int(np.floor(np.sqrt(sample.shape[0])))
        # print 'merge size:{},{}'.format(mainfold_h,mainfold_h)
        # if e_n==idx:#第一次执行的时候
        if type==0:
            self.save_images(data, [mainfold_h, mainfold_h],
                             './{}/{:02d}_{:04d}_GT_sample.png'.format(self.result_path, e_n, idx), score_r_id, identity)
            # print 'INFO: test imgwriting'
            self.save_images(sample, [mainfold_h, mainfold_h],
                             './{}/{:02d}_{:04d}_test_sample.png'.format(self.result_path, e_n, idx), score_f_id, identity)
        elif type==1:
            # print 'INFO: GT imwriting!...'
            self.save_images(data,[mainfold_h,mainfold_h],
                             './{}/{:02d}_{:04d}_GT.png'.format(self.result_path,e_n,idx),score_r_id,identity)
            # print 'INFO: test imgwriting'
            self.save_images(sample,[mainfold_h,mainfold_h],
                             './{}/{:02d}_{:04d}_test.png'.format(self.result_path,e_n,idx),score_f_id,identity)
        else:
            if not os.path.exists('./{}/{:02d}_{:04d}/'.format(self.result_path,e_n,idx)):
                os.mkdir('./{}/{:02d}_{:04d}/'.format(self.result_path,e_n,idx))
            # print 'INFO: GT for pose code writing !'
            self.save_images(data,[mainfold_h,mainfold_h],
                             './{}/{:02d}_{:04d}/_{:02d}_GT_for_posecode.png'.format(self.result_path,e_n,idx,sample_idx),
                             score_r_id,identity)
            # print 'INFO: test for pose code imwriting !...'
            self.save_images(sample, [mainfold_h, mainfold_h],
                             './{}/{:02d}_{:04d}/_{:02d}_test_for_posecode.png'.format(self.result_path, e_n, idx,sample_idx),
                             score_f_id,identity)

    def save_images(self,images,size,image_path,score,identity):
        return self.imsave(self.inverse_transform(images),size,image_path,score,identity)

    def inverse_transform(self,images):
        return (images+1.) *127.5

    def merge(self,images,size):

        # print 'image_shape:',images.shape,'size',size
        h , w = images.shape[1],images.shape[2]
        if (images.shape[3] in (3,4)):
            c = images.shape[3]
            img = np.zeros((h*size[0],w*size[1],c))
            for idx,image in enumerate(images):
                # print idx,size
                i = idx % size[1]
                j = idx // size[0]
                img[j*h:j*h+h, i*w:i*w+w,:]=image
                # cv2.imwrite('./{}/test{}.png'.format(self.result_path,idx),image)
            return img
        elif images.shape[3]==1:
            img = np.zeros((h * size[0], w * size[1]))
            for idx, image in enumerate(images):
              i = idx % size[1]
              j = idx // size[1]
              img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
            return img
        else:
            raise ValueError('in merge(images,size) images parameter '
                             'must have dimensions: HxW or HxWx3 or HxWx4')


    def imsave(self,images,size,path,score,identity):
        # image = np.squeeze(self.merge(images,size))
        # print image[0]
        # return scipy.misc.imsave(path,image)
        width = images.shape[1]
        images = self.merge(images,size)
        # sum_n=np.sum(score,1)
        if score==None:
            return cv2.imwrite(path, images[:,:,::-1])
        # print 'sum_n',sum_n
        if score!=0 and identity!=0:
            for idx in xrange(len(score)):
                i = idx % size[1]
                j = idx //size[1]
                #cv x 是h 方向
                if isinstance(identity,int):
                    score_r=score[idx][identity]
                else:
                    score_r=score[idx][identity[idx]]
                cv2.putText(images,str(round(score_r,4)),(i*width,j*width+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        # print path,images.shape
        return cv2.imwrite(path,images)

    def write_sample(self,sample,data,step):
        savepath=self.result_path+'/'+str(step)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        out_im=(sample+1)*127.5
        data = (data+1)*127.5
#        out_im=tf.to_float(out_im)

        for i in xrange(self.batch_size):
            try:
                img_real=data[i].astype(np.int)
                img=out_im[i].astype(np.int)
              #  print 'int trans success!!!'
            except:
                img=tf.to_int32(out_im[i])
                print 'to_int32 success!'
            #print img
            outname_real='/sample%i_real.jpg'%i
            outname='/sample%i.jpg'%i
            cv2.imwrite(savepath+outname_real,img_real[:,:,:])
            cv2.imwrite(savepath+outname,img[:,:,:])#::-1




