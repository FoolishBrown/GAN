#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import skimage
from tqdm import trange
import datetime
import tensorflow.contrib.slim as slim
import os
import cv2
import time
import scipy.misc

import Nets
from tqdm import trange


class Trainer(object):

    def __init__(self,check_point='check_point',restored=False,ifsave=True,
                 batch_size=64,input_size=110,input_channel=3,output_size=96,output_channel=3,sample_interval=5000,
                 data_loader_train=None,data_loader_valid=None,epoch=10000,log_dir='logdir',learning_rate=0.0002,beta1=0.5,
                 test_batch_size=10,gpus_list=None,write_type=0,model_name='dr_gan',noise_z=50,pose_c=13,random_seed=20,sample_pose_interval=10,loss_type=0):
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
        self.result_path = '{}/{}'.format(self.log_dir,datetime.date.today().strftime("%Y%m%d"))
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
        predict_r,predict_r_logits,\
        predict_r_label,predict_r_label_logits,\
        predict_r_pose,predict_r_pose_logits,\
        var = \
            Nets.netD_discriminator(batch_data,class_nums=self.data_loader_train.class_nums,posenum=self.pose_c)
        # tf.summary.histogram('discriminator/fake_pose',predict_f_pose)

        return predict_r,predict_r_logits,\
               predict_r_label,predict_r_label_logits,\
               predict_r_pose,predict_r_pose_logits,var


    def validation_drgan(self,batch_data,data_size,noise=None,pose=None):
        sample_outen=Nets.netG_encoder(batch_data,reuse=True)
        valid_add_z = tf.concat([sample_outen, noise], 3)
        pose=tf.expand_dims(pose,1)
        pose=tf.expand_dims(pose,1)
        valid_add_zp = tf.concat([valid_add_z,pose],3)
        # ------------
        sample_valid, _ = Nets.netG_deconder(valid_add_zp, data_size, self.output_channel,reuse=True)

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
        tf.set_random_seed(20)
        with tf.Graph().as_default():
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel])#image
            # input_data = tf.random_crop(src_data,[-1,self.output_size,self.output_size,self.input_channel])
            input_label = tf.placeholder(dtype= tf.int32,shape=[None])#label
            input_pose = tf.placeholder(dtype= tf.int32,shape=[None])#pose
            index = tf.placeholder(tf.int32, None)
            #mk onehot labels
            labels = slim.one_hot_encoding(input_label,self.data_loader_train.class_nums)
            poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
            # Create a variable to count the number of train() calls
            global_step = slim.get_or_create_global_step()

            #get predict data
            predict_r,logits_r, \
            predict_r_id,logits_r_id,\
            predict_r_pose,logits_r_pose,self.varsD \
                = self.predict_drgan(input_data,  pose=poses)

            #comput loss
            id_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits_r_id))
            ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=poses, logits=logits_r_pose))
            Dloss=id_loss_real+ps_loss_real
            #test accuracy:
            predict_r=tf.reshape(predict_r_id,[-1,self.data_loader_train.class_nums])
            max_id=tf.argmax(predict_r,1)
            label_id=tf.argmax(labels,1)
            correct_pred_r = tf.equal(max_id,label_id)
            correct_pred_r_pose = tf.equal(tf.argmax(tf.reshape(predict_r_pose,[-1,self.pose_c]),1),tf.argmax(poses,1))
            accuracy_id = tf.reduce_mean(tf.cast(correct_pred_r,tf.float32))
            accuracy_ps = tf.reduce_mean(tf.cast(correct_pred_r_pose,tf.float32))

            train_vars = tf.trainable_variables()
            self.varsD = [var for var in train_vars if 'discriminator' in var.name]


            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=self.beta1,name='optimizer')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_d=optimizer.minimize(Dloss,var_list=self.varsD,global_step=global_step)

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True

            with tf.Session(config=config) as sess:
                try:
                    tf.global_variables_initializer().run()
                except:
                    tf.initialize_all_variables().run()
                #,keep_checkpoint_every_n_hours=3
                saver = tf.train.Saver(max_to_keep=10)
                if self.restored:
                    saver.restore(sess,self.check_point)
                curr_interval=0
                start_time=time.time()
                self.data_loader_train.enqueueStart()
                summary_write = tf.summary.FileWriter(self.log_dir, sess.graph)
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
                        label_i,max_i,predict_id,predict_pose,p,l,step,_,acc_id,acc_ps,idloss,psloss=sess.run([label_id,max_id,predict_r_id,predict_r_pose
                            ,poses,labels,global_step,train_op_d,accuracy_id,accuracy_ps,id_loss_real,ps_loss_real],
                                    feed_dict={input_data:batch_image,
                                               input_label:batch_label[0],
                                               input_pose:batch_label[1],
                                               index: batch_image.shape[0]})
                        if epoch_n==10:
                            print batch_label[0]
                            print l
                            print batch_label[1]
                            print p
                        handledtime=time.time()
                        print 'INFO: handle a batch use {:.2f}!!!'.format(handledtime-batchend)


                        print 'Epoch [%4d/%4d] [%4d/%4d] [global_step:%d]time:%.4f, id_loss:%.4f,ps_loss:%0.4f,acc_id:%0.4f,acc_ps:%0.4f'\
                        %(epoch_n,self.epoch,interval_i,self.batch_idxs,step,time.time()-start_time,idloss,psloss,acc_id,acc_ps)
                        curr_interval+=1
#------------------------------------------write img-----------------

    def write_batch(self,type,sample,data,e_n,idx,score_r_id,score_f_id,identity,sample_idx=0):
        mainfold_h = int(np.ceil(np.sqrt(sample.shape[0])))
        if type:
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
        sum_n=np.sum(score,1)
        for idx in xrange(len(score)):
            i = idx % size[1]
            j = idx //size[1]
            #cv x 是h 方向
            score_r=score[idx][identity]/(sum_n[idx]+0.0)
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




