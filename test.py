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
import Net_8_13 as Nets
import utils
from tqdm import trange
from resnet_yd import resnet_yd

flags=tf.app.flags
# input_size = 110
# output_size = 96
# input_channel = 3
flags.DEFINE_integer('batch_size',10,'')
flags.DEFINE_integer('src_size',110,'')
flags.DEFINE_integer('output_size',96,'')
flags.DEFINE_integer('output_channel',3,'')
flags.DEFINE_boolean('flip', True, '单侧翻转')
flags.DEFINE_boolean('center_crop',True,'crop center for  validation')
flags.DEFINE_integer('label_nums',2,'label numbers')
flags.DEFINE_integer('thread_nums',6,'data read thread nums')
flags.DEFINE_integer('validation_label',331,'multipie validation')
# flags.DEFINE_string("data_label_txt",'./train_2000.txt','')
# flags.DEFINE_string("data_label_txt",'/world/data-gpu-90/rex/ms_celeb/lightcnn_144_144_clean_gray/list/train_set_part.txt','')
# flags.DEFINE_string("datapath",'/world/data-gpu-90/rex/ms_celeb/lightcnn_144_144_clean_gray/data/','')
flags.DEFINE_string("data_label_txt",'./multipie_txt/multi_subjects_flip_labeled_cut_test_oneshot_01.txt', '')
flags.DEFINE_string("datapath", '/world/data-gpu-58/wangyuequan/data_sunkejia/Multi_session/', '')
# flags.DEFINE_string("data_label_txt",'/home/sunkejia/sourcecode/pycharm/tupu_backend_server/add_label_hpidb_multilabels.txt', '')
# flags.DEFINE_string("datapath", '/world/data-gpu-58/wangyuequan/data_sunkejia/hpidb-cut-multipie/', 'data handled')
FLAGS=flags.FLAGS
class Test(object):

    def __init__(self,input_size,input_channel,gpus_list,check_point,model_id,image_data,test_count,pose=7,noise=50):
        self.pose_c=pose
        self.input_size=input_size
        self.input_channel=input_channel
        self.gpus_list=gpus_list
        self.check_point=check_point
        self.noise_z=noise
        self.image_data_reader=image_data
        self.mkdir_result(self.check_point)#savepath
        self.model_count=len(model_id)
        self.model_id=model_id
        self.test_count=test_count
        self.save_pose_identity=os.path.join(self.check_point,'p_id')
        self.mkdir_result(self.save_pose_identity)
        self.save_slerp=os.path.join(self.check_point,'slerp')
        self.mkdir_result(self.save_slerp)
    def mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir('./{}'.format(str_dir))


    def predict_drgan(self,batch_data,noise=None,pose=None):
#        with tf.variable_scope('drgan'):
#         with tf.name_scope('generator_encoder_decoder'):
        output_en = Nets.netG_encoder_gamma(batch_data)
    #----------noise
    # print 'noise:max{},min{}'.format(np.max(sampel_z),np.min(sampel_z))
        pose=tf.expand_dims(pose,1)
        pose=tf.expand_dims(pose,1)
        pose=tf.cast(pose,dtype=tf.float32)
        sample_add_z = tf.concat([output_en,pose],3)

        sample_add_zp = tf.concat([sample_add_z,noise],3)

    # print 'size_noise',pose.shape.as_list(),noise.shape.as_list()

        output_de = Nets.netG_deconder_gamma(sample_add_zp,self.input_channel)

        # pidr_softmax, pidrlogits, pidcontent \
        #     = \
            # resnet_yd(batch_data[:, :, :, ::-1], reuse=True)
        #
        # self.ppsr_logits, prcontent \
        #     = \
        #     Nets.Custom_netD_discriminator_psloss(batch_data, posenum=self.pose_c, reuse=True)
            # _, reallogits = Nets.Custom_netD_discriminator_adloss(batch_data, reuse=True)

        # pidf_softmax, pidflogits, pidfcontent \
        #     = \
        #     resnet_yd(output_de[:, :, :, ::-1], reuse=True)
        #
        # self.ppsf_logits, pfcontent \
        #     = \
        #     Nets.Custom_netD_discriminator_psloss(output_de, posenum=self.pose_c, reuse=True)
        # _, fakelogits = Nets.Custom_netD_discriminator_adloss(output_de, reuse=True)


        # with tf.name_scope('D_loss'):
            # id_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=fakelabel, logits=pidf1logits))
            # id_loss_fake_ex = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=fakelabel, logits=pidf2logits))
            # pose loss
            # self.ps_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=pose, logits=self.ppsr_logits))
            # adversarial
            # self.ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.ones_like(reallogits), logits=reallogits
            # ))
            # self.ad_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.zeros_like(pidflogits), logits=pidflogits
            # ))
            # self.ad_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.zeros_like(fake2logits), logits=fake2logits
            # ))

        # with tf.name_scope('G_loss'):
            # predict loss
            # self.id_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=label, logits=pidf1logits))
            # self.id_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=label, logits=pidf2logits))

            # ps loss
            # input_pose_ex = tf.concat(tf.split(pose, 2, axis=0)[::-1], axis=0)
            # self.ps_loss_syn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=pose, logits=self.ppsf_logits))
            # self.ps_loss_syn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #     labels=input_pose_ex, logits=ppsf2_logits))

            # identity preserved loss

            # id_preserve_f1=tf.reduce_mean(tf.squared_difference(pidrcontent,pidf1content))
            # self.id_preserve_f1 = tf.reduce_mean(tf.abs(tf.subtract(pidcontent, pidfcontent)))
            # pidrcontent_ex = tf.concat(tf.split(pidrcontent, 2, axis=0)[::-1], axis=0)
            # id_preserve_f2=tf.reduce_mean(tf.squared_difference(pidrcontent_ex,pidf2content))
            # self.id_preserve_f2 = tf.reduce_mean(tf.abs(tf.subtract(pidrcontent_ex, pidf2content)))

            # adversarial
            # self.ad_loss_syn1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.ones_like(fakelogits), logits=fakelogits
            # ))
            # self.ad_loss_syn2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.ones_like(fake2logits), logits=fake2logits

            # ))
        # print 'ad loss',self.ad_loss_real,self.ad_loss_fake1,self.ad_loss_syn1

        return output_de

    def inference(self):
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpus_list
        # tf.set_random_seed(20)
        with tf.Graph().as_default():
            input_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image
            input_pose = tf.placeholder(dtype= tf.int32,shape=[None],name='input_poses')#pose
            index = tf.placeholder(tf.int32, None,name='input_nums')
            poses = slim.one_hot_encoding(input_pose,self.pose_c)#pose code pose label
            noise = tf.random_uniform(shape=(index,1,1,self.noise_z),minval=-1,maxval=1,dtype=tf.float32,name='input_noise')
            sample=self.predict_drgan(input_data,noise=noise,pose=poses)
            # with tf.name_scope('test'):
            output_en_test = Nets.netG_encoder_gamma(input_data,reuse=True)

            code=tf.get_variable('slerpcode',[3,1,1,519],dtype=tf.float32)
            # pose_reshape_test = tf.reshape(poses, [-1, 1, 1, self.pose_c])
            # pose_float_test = tf.cast(pose_reshape_test, dtype=tf.float32)
            # pose_add_noise_test = tf.concat([pose_float_test, noise], 3)
            noise_3=tf.random_uniform(shape=(3, 1, 1, self.noise_z), minval=-1, maxval=1, dtype=tf.float32,
                              name='input_noise')
            sample_add_pn_test = tf.concat([code, noise_3], 3)
            output_de_test = Nets.netG_deconder_gamma(sample_add_pn_test, self.input_channel)


            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth =True
            train_vars = tf.trainable_variables()
            self.vars = [var for var in train_vars if 'slerpcode' not in var.name]
            with tf.Session(config=config) as sess:
                try:
                    tf.global_variables_initializer().run()
                except:
                    tf.initialize_all_variables().run()
                saver = tf.train.Saver(self.vars )
                savedata=np.zeros([self.test_count,self.pose_c,self.input_size,self.input_size,self.input_channel])
                saveimg=np.zeros([self.test_count,self.model_count,self.pose_c,self.pose_c,self.input_size,self.input_size,self.input_channel])
                self.image_data_reader.enqueueStart()
                select_count = 3
                # t, batch_label = self.image_data_reader.get_same_person_batch()
                t,l=self.image_data_reader.oneperson_allpose(0)
                index_select_1 = np.random.randint(0, t.shape[0], select_count)
                index_select_2 = np.random.randint(0, t.shape[0], select_count)
                #identity and pose invariant
                for m_i in xrange(self.model_count):#model num
                    saver.restore(sess, self.check_point+'-'+self.model_id[m_i])
                    pose_batch =range(0,self.pose_c)
                    for i in xrange(self.test_count):#sample num
                        # batch_image,label= self.image_data_reader.get_same_person_batch(i)
                        # print label
                        batch_image, label = self.image_data_reader.oneperson_allpose(i)
                        sample_count=batch_image.shape[0]
                        count=0
                        for idx in xrange(self.pose_c):
                            if label[1,idx]<=6:
                                savedata[i,count,:,:,:]=batch_image[idx]
                                count+=1
                                print batch_image.shape,sample_count,m_i,i,idx
                                tmp_batch = np.tile(batch_image[idx],(self.pose_c,1,1,1)).\
                                        reshape(self.pose_c,self.input_size,self.input_size,self.input_channel)
                                sample_data= sess.run(sample,
                                                                       feed_dict={
                                                                           input_data: tmp_batch,
                                                                           input_pose: pose_batch,
                                                                           index: self.pose_c})
                                saveimg[i,m_i,idx,:,:,:,:]=sample_data
                                # utils.write_batch(self.check_point,0,sample_data,tmp_batch,i,idx,)
                                # 球面差值法测试
                                # for _ in xrange(3):#测试三次
                                # print 'slerp test!!'
                                if isinstance(label,int):
                                    batch_label_n=np.random.randint(0,self.pose_c,batch_image.shape[0])
                                else:
                                    batch_label_n=label[1]
                                de_code_1 = sess.run(output_en_test,
                                                     feed_dict={input_data: batch_image[index_select_1],
                                                                input_pose: batch_label_n[index_select_1],
                                                                index: select_count})
                                de_code_2 = sess.run(output_en_test,
                                                     feed_dict={input_data: batch_image[index_select_2],
                                                                input_pose: batch_label_n[
                                                                    index_select_2],
                                                                index: select_count})
                                pose_1 = np.asarray(batch_label_n[index_select_1], np.int32)
                                pose_2 = np.asarray(batch_label_n[index_select_2], np.int32)
                                b_1 = np.zeros((pose_1.size, self.pose_c), dtype=np.float32)
                                b_2 = np.zeros_like(b_1, dtype=np.float32)
                                b_1[np.arange(pose_1.size), pose_1] = 1
                                b_2[np.arange(pose_2.size), pose_2] = 1
                                # print b_1,b_2
                                de_code_1 = np.concatenate([np.reshape(de_code_1, [-1, 512]), b_1], axis=1)
                                de_code_2 = np.concatenate([np.reshape(de_code_2, [-1, 512]), b_2], axis=1)
                                decodes = []
                                for idx, ratio in enumerate(np.linspace(0, 1, 10)):
                                    z = np.stack([utils.slerp(ratio, r1, r2) for r1, r2 in
                                                  zip(de_code_1, de_code_2)])
                                    z = np.reshape(z, [-1, 1, 1, 519])
                                    # print z.shape
                                    z_decode = sess.run(output_de_test,
                                                        feed_dict={code: z,
                                                                   input_data: batch_image,
                                                                   input_pose: batch_label_n,
                                                                   index: select_count})
                                    decodes.append(z_decode)

                                decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
                                for idx, img in enumerate(decodes):
                                    img = np.concatenate([[batch_image[index_select_1[idx]]], img,
                                                          [batch_image[index_select_2[idx]]]], 0)
                                    img = utils.inverse_transform(img)[:, :, :, ::-1]
                                    utils.save_image(img, os.path.join('./{}'.format(self.save_slerp),
                                                                       'test{}_interp_G_{}.png'.format(m_i, i)),
                                                     nrow=10 + 2)
                    saveimage(self.save_pose_identity,saveimg,savedata,self.model_count,self.test_count,self.pose_c,self.input_size,self.input_channel)


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



if __name__=='__main__':
    data_reader_valid = utils.ImageReader_Customize(batch_size=FLAGS.batch_size,flip=FLAGS.flip
            ,input_size=FLAGS.src_size,output_size=FLAGS.output_size,center_crop=FLAGS.center_crop,
                                              data_label_txt=FLAGS.data_label_txt,data_path=FLAGS.datapath,
                                              label_nums=FLAGS.label_nums,validtionlabel=FLAGS.validation_label)

    # model_id=['10734','16100','21466','26832','32198','42930']
    # model_id=['197479','203203','20127','211789','223237']
    # model_id=['28297','35371','37729','40087','42445','44803','47161']
    model_id=['12585','15489','18393','20329','22265','24201','26137','27105','28073','29041','30009','30977']
    t = Test(FLAGS.output_size, FLAGS.output_channel, gpus_list='3',
             check_point="./logdir_caspeal/logdir_DR_MultiPIE_shareGAN/20170826_gpu_2_vflip_multipie12.13new01/checkpoint/DR_MultiPIE_shareGAN",
             model_id=model_id, image_data=data_reader_valid, test_count=5)

    # t=Test(FLAGS.output_size,FLAGS.output_channel,gpus_list='5',check_point="./logdir_caspeal/logdir_DR_MultiPIE_10.4_derivative/20170822_gpu_3_vflip_multipie10.422/checkpoint/DR_MultiPIE_10.4_derivative",model_id=model_id,image_data=data_reader_valid,test_count=5)
    # t = Test(FLAGS.output_size, FLAGS.output_channel, gpus_list='0',
    #          check_point="./checkpoint/",
    #          model_id=model_id, image_data=data_reader_valid, test_count=1)
    t.inference()