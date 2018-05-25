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

    def __init__(self,check_point='check_point',restored=False,ifsave=True,lr_alpha=0.5,lr_beta=0.5,lr_gamma=1,
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,
                 data_loader_train=None,data_loader_valid=None,epoch=10000,log_dir='logdir',learning_rate=0.0002,beta1=0.5,
                 test_batch_size=10,gpus_list=None,write_type=0,model_name='dr_gan',noise_z=50,pose_c=13,random_seed=25,sample_pose_interval=10,
                 loss_type=0,Gloop=3):
        self.lr_alpha=lr_alpha
        self.lr_beta = lr_beta
        self.losstype=loss_type
        self.check_point=check_point
        self.mkdir_result(self.check_point)
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
        self.log_dir = log_dir
        self.savename=str(self.learning_rate)+'_'+str(self.lr_alpha)+'_'+str(self.lr_beta)+'_original_new_'+str(Gloop)
        self.result_path = '{}/{}_{}'.format(self.log_dir,datetime.date.today().strftime("%Y%m%d"),self.savename)
        self.mkdir_result(self.log_dir)
        self.mkdir_result(self.result_path)
        self.gpus_list=gpus_list

        self.model_name=model_name
        self.mkdir_result(self.check_point)
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

    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))


    def predict_drgan(self,batch_data,noise=None,pose=None):
#        with tf.variable_scope('drgan'):
        with tf.name_scope('generator_encoder_decoder'):
            output_en = Nets.netG_encoder(batch_data)
        #----------noise
        # print 'noise:max{},min{}'.format(np.max(sampel_z),np.min(sampel_z))
            sample_add_z = tf.concat([output_en,noise],3)
            pose=tf.expand_dims(pose,1)
            pose=tf.expand_dims(pose,1)
            sample_add_zp = tf.concat([sample_add_z,pose],3)
        # print  sample_add_zp.shape
    #------------
            size_noise=noise.shape.as_list()[0]
        # print 'size_noise',pose.shape.as_list(),noise.shape.as_list()

            output_de, _ = Nets.netG_deconder(sample_add_zp,size_noise,self.output_channel)

        tf.summary.histogram('generator/outputencoder',output_en)
        tf.summary.histogram('generator/inputdecoder',sample_add_zp)
        tf.summary.histogram('generator/outputdecoder',output_de)

        tf.summary.histogram('input_discrimintor/inputrealdata',batch_data)
        tf.summary.histogram('input_discrimintor/inputsyndata',output_de)
        with tf.name_scope('discriminator_total'):
            predict_r,predict_r_logits,\
            predict_r_label,predict_r_label_logits,\
            predict_r_pose,predict_r_pose_logits,\
            _ = \
            Nets.netD_discriminator(batch_data,class_nums=self.data_loader_train.class_nums+1,posenum=self.pose_c)

            predict_f,predict_f_logits,\
            predict_f_label,predict_f_label_logits,\
            predict_f_pose,predict_f_pose_logits,\
            _ = \
            Nets.netD_discriminator(output_de,class_nums=self.data_loader_train.class_nums+1,posenum=self.pose_c,reuse=True)

        tf.summary.histogram('discriminator/real_d',predict_r)
        tf.summary.histogram('discriminator/fake_d',predict_f)
        tf.summary.histogram('discriminator/real_id',predict_r_label)
        tf.summary.histogram('discriminator/fake_id',predict_f_label)
        tf.summary.histogram('discriminator/real_pose',predict_r_pose)
        tf.summary.histogram('discriminator/fake_pose',predict_f_pose)

        return predict_r,predict_r_logits,\
               predict_r_label,predict_r_label_logits,\
               predict_r_pose,predict_r_pose_logits,\
               predict_f,predict_f_logits,\
               predict_f_label,predict_f_label_logits,\
               predict_f_pose,predict_f_pose_logits,\
               output_de,output_en

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

        return predict_data,predict_fake,sample,_



    def loss_drgan(self,predict_data,predict_data_logits,predict_fake,predict_fake_logits,labels=None,
                   pose_label=None,logits_r_id=None,logits_f_id=None,logits_pose_r=None,logits_pose_f=None,false_labels=None):
        with tf.name_scope('Discriminator_loss'):
            # ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            # labels=tf.ones_like(predict_data),logits=predict_data_logits))
            # ad_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            # labels=tf.zeros_like(predict_fake),logits=predict_fake_logits))
            id_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=false_labels,logits=logits_f_id
            ))
            id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,logits=logits_r_id))
        # id_loss_fake = tf.reduce_mean()
            ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=pose_label,logits=logits_pose_r))
        with tf.name_scope('Generator_loss'):
        # Dloss = ad_loss_fake+ad_loss_real+id_loss_real+ps_loss_real
        #     ad_loss_syn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.ones_like(predict_fake),logits=predict_fake_logits))
            id_loss_syn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,logits=logits_f_id))
            ps_loss_syn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=pose_label,logits=logits_pose_f))
        # Gloss = ad_loss_syn+id_loss_syn+ps_loss_syn
        return id_loss_real,id_loss_fake,ps_loss_real,\
               id_loss_syn,ps_loss_syn

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

    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpus_list
        # tf.set_random_seed(20)
        with tf.Graph().as_default():
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
            # input_data = tf.random_crop(src_data,[-1,self.output_size,self.output_size,self.input_channel])
            input_label = tf.placeholder(dtype= tf.int32,shape=[None],name='input_labels')#label
            input_pose = tf.placeholder(dtype= tf.int32,shape=[None],name='input_poses')#pose
            index = tf.placeholder(tf.int32, None,name='input_nums')
            #mk onehot labels
            labels = slim.one_hot_encoding(input_label,self.data_loader_train.class_nums+1)
            input_false_labels = tf.placeholder(dtype=tf.int32,shape=[None],name='falselabel')
            false_labels = slim.one_hot_encoding(input_false_labels,self.data_loader_train.class_nums+1)
            poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label

            noise = tf.random_uniform(shape=(index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
            # input_noise = tf.placeholder(dtype=tf.float32,shape=[None,1,1,self.noise_z])
            # global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

            # Create a variable to count the number of train() calls
            # global_step = slim.get_or_create_global_step()


            #get predict data
            predict_r,logits_r, \
            predict_r_id,logits_r_id,\
            predict_r_pose,logits_r_pose,\
            predict_f,logits_f,\
            predict_f_id,logits_f_id,\
            predict_f_pose,logits_f_pose,\
            sample,encode=self.predict_drgan(input_data,noise=noise,pose=poses)
            #comput loss
            id_loss_real,id_loss_fake, ps_loss_real,\
            id_loss_syn,ps_loss_syn\
                =self.loss_drgan(predict_r,logits_r,
                    predict_f,logits_f,
                    labels=labels,pose_label=poses,
                    logits_r_id=logits_r_id,logits_f_id=logits_f_id,
                    logits_pose_r=logits_r_pose,logits_pose_f=logits_f_pose,false_labels=false_labels)

            with tf.name_scope('compute_loss'):
                id_loss_real=self.lr_alpha*id_loss_real
                id_loss_syn=id_loss_syn*self.lr_alpha
                ps_loss_real=ps_loss_real * self.lr_beta
                ps_loss_syn=ps_loss_syn*self.lr_beta
                Dloss=id_loss_real+id_loss_fake+ps_loss_real
                Gloss=id_loss_syn+ps_loss_syn
                # Dloss=ad_loss_real+ad_loss_fake
                # Gloss=ad_loss_syn
                # if self.losstype==1:
                #     Dloss+=id_loss_real
                #     Gloss+=id_loss_syn
                # elif self.losstype==2:
                #     Dloss+=id_loss_real+ps_loss_real
                #     Gloss+=id_loss_syn+ps_loss_syn
            #test accuracy:
            with tf.name_scope('accurary'):
                reshape_R=tf.reshape(predict_r_id, [-1, self.data_loader_train.class_nums+1])
                max_r=tf.argmax(reshape_R,1)
                label_true=tf.argmax(labels,1)
                correct_pred_r = tf.equal(max_r,label_true)
                reshape_F=tf.reshape(predict_f_id, [-1, self.data_loader_train.class_nums+1])
                max_f=tf.argmax(reshape_F, 1)
                correct_pred_f = tf.equal(max_f,label_true)
                accuracy_r = tf.reduce_mean(tf.cast(correct_pred_r,tf.float32))
                accuracy_f = tf.reduce_mean(tf.cast(correct_pred_f,tf.float32))

            summary_train_accracy_r = tf.summary.scalar('accuracy/real_data',accuracy_r)
            summary_train_accracy_f = tf.summary.scalar('accuracy/fake_data',accuracy_f)
            summary_train_lossD = tf.summary.scalar('loss/total_loss_d',Dloss)
            summary_train_lossG = tf.summary.scalar('loss/total_loss_g',Gloss)

            # summary_train_lossD_ad_r = tf.summary.scalar('loss/ad_loss_d_real',ad_loss_real)
            # summary_train_lossD_ad_f = tf.summary.scalar('loss/ad_loss_d_fake',ad_loss_fake)
            summary_train_lossD_id = tf.summary.scalar('loss/id_loss_d',id_loss_real)
            summary_train_lossD_ps = tf.summary.scalar('loss/ps_loss_d',ps_loss_real)

            summary_train_lossG_syn = tf.summary.scalar('loss/id_fake_loss_d',id_loss_fake)
            summary_train_lossG_id = tf.summary.scalar('loss/id_loss_g',id_loss_syn)
            summary_train_lossG_ps = tf.summary.scalar('loss/ps_loss_g',ps_loss_syn)

            summary_train_image_batch = tf.summary.image('image/input',tf.expand_dims(input_data[0],0))
            summary_train_image_sample = tf.summary.image('image/decoder',tf.expand_dims(sample[0],0))
            summary_train = tf.summary.merge_all()
            # summary_train = tf.summary.merge([summary_train_lossD,summary_train_lossG,summary_train_lossD_ad_r,summary_train_lossD_ad_r,summary_train_lossD_id,
            #                                   summary_train_lossG_syn,summary_train_lossG_id])
            train_vars = tf.trainable_variables()
            self.varsG = [var for var in train_vars if 'generator' in var.name]
            self.varsD = [var for var in train_vars if 'discriminator' in var.name]

            # train_op = self.get_train_op(Dloss,Gloss,global_step,self.learning_rate,self.beta1)
            sample_valid = self.validation_drgan(input_data, index, noise=noise[:index], pose=poses)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True

            with tf.Session(config=config) as sess:

                for learning_rate in [self.learning_rate]:
                    # embedding = tf.Variable(tf.zeros([7,96,96,3]),name='test_embedding')
                    # assignment = embedding.assign(input_data)
                    # config2=projector.ProjectorConfig()
                    # embedding_config=config2.embeddings.add()
                    # embedding_config.tensor_name=embedding.name
                    # embedding_config.sprite.image=os.path.join(self.log_dir,'sprite.png')
                    # embedding_config.sprite.single_image_dim.extend([96,96,3])




                    curr_interval=0
                    with tf.name_scope('train_optimizer'):
                        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=self.beta1,
                                                           name='optimizer')
                        # batch norm,when trainingm the moving_mean and moving_variance need to be updated!!!!!!
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            train_op_d = optimizer.minimize(Dloss, var_list=self.varsD)
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
                            batchend=time.time()
                            print 'INFO: read a batch images use {:.4f}s'.format(batchend-bathstart)

                            print 'INFO: made noise code'
                            # batchnoise = np.random.uniform(-1, 1, size=(self.batch_size, 1,1, self.noise_z))
                            #optimizer D
                            flabels=[self.data_loader_train.class_nums] * batch_image.shape[0]
                            _=sess.run(train_op_d,
                                        feed_dict={input_data:batch_image,
                                                   input_label:batch_label[0],
                                                   input_pose:batch_label[1],
                                                   input_false_labels:flabels,
                                                   index: batch_image.shape[0]})
                            # print 'imageszie',im.shape,im2.shape
                            # break
                            # #optimizer G
                            for _ in xrange(self.Gloop):
                                _= sess.run(train_op_g,
                                    feed_dict={input_data: batch_image,
                                               input_label: batch_label[0],
                                               input_pose: batch_label[1],
                                               input_false_labels:flabels,
                                               index: batch_image.shape[0]})

                            r,r1,mr,mf,m, _, sample_, loss_d, loss_g, code , train_summary,acc_r,acc_f= sess.run(
                                [reshape_R,reshape_F,max_r,max_f,label_true,train_op_g, sample, Dloss, Gloss, encode,summary_train,
                                 accuracy_r,accuracy_f],
                                feed_dict={input_data: batch_image,
                                           input_label:batch_label[0],
                                           input_pose: batch_label[1],
                                           input_false_labels:flabels,
                                           index: batch_image.shape[0]})
                            summary_write.add_summary(train_summary,global_step=step)

                            step+=1

                            handledtime=time.time()
                            print 'INFO: handle a batch use {:.2f}!!!'.format(handledtime-batchend)


                            print 'Epoch [%4d/%4d] [%4d/%4d] [global_step:%d]time:%.4f, d_loss:%.8f, g_loss:%.8f,acc_r:%0.4f,acc_f:%0.4f'\
                            %(epoch_n,self.epoch,interval_i,self.batch_idxs,step,time.time()-start_time,loss_d,loss_g,acc_r,acc_f)
                            print 'sampele:',np.max(sample_),np.min(sample_),sample_.shape
                            print 'code',np.max(code)  ,np.min(code)
                            # print 'm',m
                            # print 'mr',mr
                            # print 'mf',mf
                            # if epoch_n%4!=0 or epoch_n==0:
                            #     continue
                            if (curr_interval)%int(self.sample_interval*self.batch_idxs)==0:
                            # if step%int(self.sample_interval)
                                # if self.write_type==0:
                                #print 'INFO: write sample...'
                                # self.write_sample(sample_,batch_image,)
                                # else:
                                print 'INFO: handle batch...'
                                self.write_batch(0,sample_,batch_image,epoch_n,interval_i,r,r1,m)

                                sample_batch , pose_batch ,identity =self.data_loader_train.get_same_person_batch(self.random_seed)
                                sample_count = sample_batch.shape[0]
                                # flabels = [self.data_loader_train.class_nums + 1] * sample_count
                                label_batch = [identity] * sample_count
                                sample_data, mr,mf,score_r_id, score_f_id = sess.run([sample,max_r,max_f,reshape_R,reshape_F],feed_dict={
                                                                              input_data: sample_batch,
                                                                              input_label:label_batch,
                                                                              input_pose: pose_batch,
                                                                              # input_false_labels:flabels,
                                                                              index : sample_count})
                                print 'score_real:','--'*20,score_r_id.shape
                                print 'INFO: write batch...'

                                # score_r_id = score_r_id.reshape(-1,self.data_loader_train.class_nums)
                                # print identity
                                # score_f_id = score_f_id.reshape(-1,self.data_loader_train.class_nums)
                                self.write_batch(1,sample_data,sample_batch,epoch_n,interval_i,score_r_id,score_f_id,identity)


                                print 'INFO: writing pose code testing ...'
                                label_batch = [identity] * self.pose_c
                                if (curr_interval)%int(self.sample_pose_interval * self.batch_idxs)==0:
                                    pose_batch =np.arange(-(self.pose_c/2),self.pose_c/2+1)+6
                                    for idx in xrange(sample_count):
                                        # print sample_batch.shape,sample_count
                                        tmp_batch = np.tile(sample_batch[idx],(self.pose_c,1,1,1)).\
                                            reshape(self.pose_c,sample_batch.shape[1],sample_batch.shape[2],sample_batch.shape[3])
                                        sample_data, score_r_id, score_f_id = sess.run([sample, reshape_R, reshape_F],
                                                                           feed_dict={
                                                                               input_data: tmp_batch,
                                                                               input_label:label_batch,
                                                                               input_pose: pose_batch,
                                                                               index: self.pose_c})
                                        # score_r_id = score_r_id.reshape(-1, self.data_loader_train.class_nums)
                                        # score_f_id = score_f_id.reshape(-1, self.data_loader_train.class_nums)

                                        print score_f_id[0],np.sum(score_f_id),sum(score_f_id[0])
                                        self.write_batch(2,sample_data,tmp_batch,epoch_n,interval_i,score_r_id,score_f_id,identity,sample_idx=idx)

                                # #save model
                                if self.ifsave:
                                    saver.save(sess,os.path.join(self.log_dir,self.model_name+self.savename+'.ckpt'),global_step=step)
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
    def write_batch(self,type,sample,data,e_n,idx,score_r_id,score_f_id,identity,sample_idx=0):
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

        # print 'sum_n',sum_n
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




