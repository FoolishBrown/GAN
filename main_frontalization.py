# -*- coding: utf-8 -*-
from utils import ImageReader_Customize
import os
import scipy.misc
import tensorflow as tf
from train_frontalization import Trainer
import datetime

import cv2
import numpy as np

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 20, 'tarin_batch_size for one gpu')
flags.DEFINE_integer("sample_batch_size", 32, 'test,sample batch_size for one gpu,一般比pose_c相关')  # 保证可以被2、gpu数整除
flags.DEFINE_string('root_path', './logdir_caspeal', 'root path')
flags.DEFINE_integer("input_size", 96, 'train_batch_size')
flags.DEFINE_integer("input_channel", 3, 'train_channel')  # 换灰白图像
flags.DEFINE_integer("output_size", 96, 'output_size')
flags.DEFINE_integer('src_size', 110, 'crop op in this size')
flags.DEFINE_integer("output_channel", 3, 'out_put_channel')
flags.DEFINE_boolean('train', True, 'if Train or inference')
flags.DEFINE_bool('random_crop', True, 'is random crop??')
flags.DEFINE_float("beta1", 0.5, 'moment--m')
flags.DEFINE_float("d_learning_rate", 0.0001, 'base learning rate')
flags.DEFINE_float("g_learning_rate", 0.0001, 'base_learning rate')

flags.DEFINE_float("sample_pose_interval", 1, 'test identity interval')
flags.DEFINE_integer("epoch", 50, 'max train_epoch ')
'''image data reader'''
flags.DEFINE_boolean('flip', False, '单侧翻转')
flags.DEFINE_string("data_path_txt", './multipie_setting2/train_label_4.txt', '')
flags.DEFINE_string("datapath", '/world/data-gpu-58/wangyuequan/data_sunkejia/Multi_session/', '')
flags.DEFINE_string("valid_path_txt", './multipie_setting2/test_label_4.txt', '')
flags.DEFINE_string("validpath", '/world/data-gpu-58/wangyuequan/data_sunkejia/Multi_session/', '')
flags.DEFINE_integer('validation_label', 0, 'multipie validation,if validation label start from 30 ,this number wil be 30 too')
flags.DEFINE_boolean('center_crop', True, 'crop center for  validation')
flags.DEFINE_integer('label_nums', 4, 'label numbers')
flags.DEFINE_integer('thread_nums', 8, 'data read thread nums')

flags.DEFINE_string('model_name','FinalTrain_Setting2','')
flags.DEFINE_string('newtag','Setting2_fr_6631_resnet_original_setting2','model version')
#如果GD不一起训练的话，会出现彩色
flags.DEFINE_string('discribe','','version discribe')
flags.DEFINE_integer('loss_type',1,'#0:ad 1:percptual loss 2:ps idloss')
flags.DEFINE_integer('Gloop',1,'G:D,n:1')#(4,7-3,2,-0,1)
flags.DEFINE_string("gpu_list",'0',"CUDA visiabTruele device")
flags.DEFINE_string("gpu_test",'3 ',"test gpu")
flags.DEFINE_float('warmuprate',2,'warmup*batch_idx')
'''continue train'''
flags.DEFINE_boolean('restored', True, 'finetuning model')
FLAGS = flags.FLAGS
lab_dir = '{}/logdir_{}'.format(FLAGS.root_path, FLAGS.model_name)

result_save_path = '{}/{}_gpu_{}_v{}'.format(lab_dir, datetime.date.today().strftime("%Y%m%d"), FLAGS.gpu_list,
                                             FLAGS.newtag)
# check_point = '{}'.format('./checkpoint/model.ckpt-162000')
# check_point = './logdir_caspeal/logdir_Final_Setting1/20171113_gpu_1_vSetting1/checkpoint/FinalTrain—Setting1-6487'
# check_point2 = './tf_inception_v1/model-20171024-190413.ckpt-147982'
check_point2='./logdir_caspeal/logdir_finetunefr/20171115_gpu_3_vsetting2/checkpoint/finetunefr-6631'
check_point = './logdir_caspeal/logdir_frontalization/20171024_gpu_6_vMSR_Unet_test_continue/checkpoint/frontalization-176054'
log_path = '{}/{}.log'.format(result_save_path, FLAGS.model_name)
image_save_path = '{}/{}'.format(result_save_path, 'image_synthesis')
summary_save_path = '{}/summarytotal'.format(lab_dir)

flags.DEFINE_string('test_path_txt',
                    '/home/sunkejia/sourcecode/pycharm/tupu_backend_server/add_label_hpidb_multilabels.txt',
                    'validation path')


def main():
    data_reader_valid = ImageReader_Customize(batch_size=FLAGS.batch_size, flip=FLAGS.flip,
                                              output_channel=FLAGS.output_channel
                                              , input_size=FLAGS.src_size, output_size=FLAGS.output_size,
                                              center_crop=FLAGS.center_crop, data_label_txt=FLAGS.valid_path_txt,
                                              data_path=FLAGS.validpath, label_nums=FLAGS.label_nums,
                                              validtionlabel=FLAGS.validation_label,thread_nums=1)

    data_reader_train = ImageReader_Customize(batch_size=FLAGS.batch_size, flip=FLAGS.flip,
                                              output_channel=FLAGS.output_channel
                                              , input_size=FLAGS.src_size, output_size=FLAGS.output_size,
                                              center_crop=FLAGS.center_crop, thread_nums=FLAGS.thread_nums,
                                              data_label_txt=FLAGS.data_path_txt, data_path=FLAGS.datapath,
                                              label_nums=FLAGS.label_nums)

    drgan = Trainer(sample_interval=FLAGS.sample_pose_interval,restored=FLAGS.restored, batch_size=FLAGS.batch_size,
                    epoch=FLAGS.epoch, log_dir=lab_dir, d_learning_rate=FLAGS.d_learning_rate,
                    g_learning_rate=FLAGS.g_learning_rate, beta1=FLAGS.beta1,
                    test_batch_size=FLAGS.sample_batch_size, model_name=FLAGS.model_name,output_channel=FLAGS.output_channel,
                    input_size=FLAGS.input_size, input_channel=FLAGS.input_channel, data_loader_train=data_reader_train,
                    gpus_list=FLAGS.gpu_list,check_point=check_point, output_size=FLAGS.output_size,
                    version=FLAGS.newtag,Gloop=FLAGS.Gloop,gpu=FLAGS.gpu_test,check_point_fr=check_point2,
                    savepath=result_save_path, imagepath=image_save_path, logfile=log_path,
                    summary_dir=summary_save_path, discribe=FLAGS.discribe,
                    data_loader_valid=data_reader_valid)
    if FLAGS.train:
        drgan.train()

if __name__ == "__main__":
    # print tf.__version__
    print tf.__path__
    main()



