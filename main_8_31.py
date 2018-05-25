# -*- coding: utf-8 -*-
from utils import ImageReader_Customize
import os
import scipy.misc
import tensorflow as tf
from train_model_8_31 import Trainer
import datetime

import cv2
import numpy as np

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 20, 'tarin_batch_size for one gpu')
flags.DEFINE_integer("sample_batch_size", 8, 'test,sample batch_size for one gpu,一般比pose_c相关')  # 保证可以被2、gpu数整除
# flags.DEFINE_string('model_name','DR_GAN_Flip_caspeal','model name')
flags.DEFINE_string('root_path', './logdir_caspeal', 'root path')
flags.DEFINE_integer("input_size", 96, 'train_batch_size')
flags.DEFINE_integer("input_channel", 3, 'train_channel')  # 换灰白图像
flags.DEFINE_integer("output_size", 96, 'output_size')
flags.DEFINE_integer('src_size', 110, 'crop op in this size')
flags.DEFINE_integer("output_channel", 3, 'out_put_channel')
flags.DEFINE_integer('write_type', 0, '0 single image,1:merge image')
flags.DEFINE_boolean('train', True, 'if Train or inference')
flags.DEFINE_bool('random_crop', True, 'is random crop??')
flags.DEFINE_float("beta1", 0.5, 'moment--m')
flags.DEFINE_float("d_learning_rate", 0.0002, 'base learning rate')
flags.DEFINE_float("g_learning_rate", 0.0002, 'base_learning rate')
flags.DEFINE_integer('pose_code', 7, 'pose个数')
FLAGS = flags.FLAGS
flags.DEFINE_integer('mode', 2, 'GAN mode,2:mean MultiPIE')
flags.DEFINE_float("sample_pose_interval", 0.2, 'test identity interval')  # epoch<sample<test
flags.DEFINE_float("test_pose_interval", 0.5, 'test pose interval save the decode images')
flags.DEFINE_float("epoch_interval", 0.2, 'validation interval save the decode images')
flags.DEFINE_integer("epoch", 240, 'train_epoch')
flags.DEFINE_float('gamma', 0.48, 'ad')
flags.DEFINE_float('alpha', 0.48, '-')
flags.DEFINE_float('beta', 0.48, 'G中pose的加权')
flags.DEFINE_float('delta', 0.48, 'identity')
'''image data reader'''
flags.DEFINE_boolean('flip', False, '单侧翻转')
flags.DEFINE_string("data_path_txt", './multipie_txt/multi_subjects_labeled_cut_train_oneshot_01_45_label_3.txt', '')
flags.DEFINE_string("datapath", '/world/data-gpu-58/wangyuequan/data_sunkejia/Multi_session/', '')
flags.DEFINE_string("valid_path_txt", './multipie_txt/multi_subjects_labeled_cut_test_oneshot_01_45_label_3.txt', '')
flags.DEFINE_string("validpath", '/world/data-gpu-58/wangyuequan/data_sunkejia/Multi_session/', '')
flags.DEFINE_integer('validation_label', 0, 'multipie validation')
flags.DEFINE_boolean('center_crop', True, 'crop center for  validation')
flags.DEFINE_integer('label_nums', 3, 'label numbers')
flags.DEFINE_integer('thread_nums', 10, 'data read thread nums')

flags.DEFINE_string('model_name','share_gan_disentangle_light_pose','')
flags.DEFINE_string('newtag','disentangle0.5_fintune','model version')
#如果GD不一区训练的话，会出现彩色块
flags.DEFINE_string('discribe','onshot-data,pixelloss_up!通过吧ad分块，DG全都初始化,用resnet96去除pixel loss 看看greyfr 对g的影响,用了已经训练好的generator,ps+ad+percetual+identity,去掉了d中的identity对抗,修复了noise pose的绑定的问题，加入了球面差值,tv,pixel','version discribe')
flags.DEFINE_integer('loss_type',1,'#0:ad 1:percptual loss 2:ps idloss')
flags.DEFINE_integer('Gloop',5,'G:D,n:1')
flags.DEFINE_string("gpu_list",'5',"CUDA visiabTruele device")
flags.DEFINE_boolean('warmup',False,'')
flags.DEFINE_float('warmuprate',2,'warmup*batch_idx')

'''continue train'''
flags.DEFINE_boolean('restored', True, 'finetuning model')

lab_dir = '{}/logdir_{}'.format(FLAGS.root_path, FLAGS.model_name)

result_save_path = '{}/{}_gpu_{}_v{}'.format(lab_dir, datetime.date.today().strftime("%Y%m%d"), FLAGS.gpu_list,
                                             FLAGS.newtag)
# check_point='{}/checkpoint'.format(result_save_path)
# check_point = '{}'.format('./checkpoint/model.ckpt-162000')
check_point = '{}'.format('./logdir_caspeal/logdir_finetunefr/20170904_gpu_7_v0.1/checkpoint/finetunefr-5461')
# check_point='{}'.format('./logdir_caspeal/logdir_share_gan_disentangle_light_pose/20170831_gpu_6_vdisentangle0.1/checkpoint/checkpoint_save/share_gan_disentangle_light_pose-15777')
log_path = '{}/{}.log'.format(result_save_path, FLAGS.model_name)
image_save_path = '{}/{}'.format(result_save_path, 'image_synthesis')
summary_save_path = '{}/summarytotal'.format(lab_dir)

flags.DEFINE_string('test_path_txt',
                    '/home/sunkejia/sourcecode/pycharm/tupu_backend_server/add_label_hpidb_multilabels.txt',
                    'validation path')


def main():
    # data_reader_valid = ImageReader_Customize()
    data_reader_valid = ImageReader_Customize(batch_size=FLAGS.batch_size, flip=FLAGS.flip,
                                              output_channel=FLAGS.output_channel
                                              , input_size=FLAGS.src_size, output_size=FLAGS.output_size,
                                              center_crop=FLAGS.center_crop, data_label_txt=FLAGS.valid_path_txt,
                                              data_path=FLAGS.validpath, label_nums=FLAGS.label_nums,
                                              validtionlabel=FLAGS.validation_label)

    data_reader_train = ImageReader_Customize(batch_size=FLAGS.batch_size, flip=FLAGS.flip,
                                              output_channel=FLAGS.output_channel
                                              , input_size=FLAGS.src_size, output_size=FLAGS.output_size,
                                              center_crop=FLAGS.center_crop, thread_nums=FLAGS.thread_nums,
                                              data_label_txt=FLAGS.data_path_txt, data_path=FLAGS.datapath,
                                              label_nums=FLAGS.label_nums)

    drgan = Trainer(sample_interval=FLAGS.epoch_interval, sample_pose_interval=FLAGS.sample_pose_interval,
                    restored=FLAGS.restored, batch_size=FLAGS.batch_size,
                    epoch=FLAGS.epoch, log_dir=lab_dir, d_learning_rate=FLAGS.d_learning_rate,
                    g_learning_rate=FLAGS.g_learning_rate, beta1=FLAGS.beta1,
                    test_batch_size=FLAGS.sample_batch_size, model_name=FLAGS.model_name, write_type=FLAGS.write_type,
                    output_channel=FLAGS.output_channel,
                    input_size=FLAGS.input_size, input_channel=FLAGS.input_channel, data_loader_train=data_reader_train,
                    gpus_list=FLAGS.gpu_list,
                    check_point=check_point, output_size=FLAGS.output_size, loss_type=FLAGS.loss_type,
                    version=FLAGS.newtag,
                    lr_alpha=FLAGS.alpha, lr_beta=FLAGS.beta, lr_gamma=FLAGS.gamma, Gloop=FLAGS.Gloop,
                    pose_c=FLAGS.pose_code,
                    savepath=result_save_path, imagepath=image_save_path, logfile=log_path,
                    summary_dir=summary_save_path, test_interval=FLAGS.test_pose_interval,
                    warmup=FLAGS.warmup, delta=FLAGS.delta, discribe=FLAGS.discribe,
                    data_loader_valid=data_reader_valid, warmup_rate=FLAGS.warmuprate)
    # data_reader_valid.Dual_enqueueStart()
    # data_reader_train.Dual_enqueueStart()
    # x,c=data_reader_train.read_data_batch()
    # x1, c1 = data_reader_valid.read_data_batch()
    #
    if FLAGS.train:
        drgan.train()
        # if FLAGS.restored:
        #     drgan.inference()


if __name__ == "__main__":
    # print tf.__version__
    print tf.__path__
    print FLAGS.write_type
    main()



