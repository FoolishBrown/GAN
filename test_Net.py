#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:51:12 2017

@author: air
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
from tensorflow.python.ops import math_ops


def instance_norm(x, scope='instance_norm'):  # simple batch norm without moving statistics
    with tf.variable_scope(scope):
        channel = x.get_shape()[-1]
        # print channel,x.get_shape()
        beta = tf.get_variable(name='beta', shape=channel,dtype=tf.float32, initializer=tf.zeros_initializer())
        gamma = tf.get_variable(name='gamma', shape=channel,dtype=tf.float32, initializer=tf.random_normal_initializer(1, 0.02))
        mean, var = tf.nn.moments(x, [1, 2], name='moments', keep_dims=True)
        inv = gamma * math_ops.rsqrt(var + 1e-6)
        output = (x - mean) * inv + beta
    return output

def inst_norm(inputs, epsilon=1e-3, suffix=''):
    """
    Assuming TxHxWxC dimensions on the tensor, will normalize over
    the H,W dimensions. Use this before the activation layer.
    This function borrows from:
    http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Note this is similar to batch_normalization, which normalizes each
    neuron by looking at its statistics over the batch.
    :param input_:
    input tensor of NHWC format
    """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    print stat_shape
    scale = tf.get_variable('INscale'+suffix,
            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('INshift'+suffix,
            initializer=tf.zeros(stat_shape[3]))

    #batch  nrom axes=[0,1,2] 出来的结果只有1 * C,而 instanse norm 结果为 B* C
    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift


    return output

def netG_encoder(image_input,reuse=False):
    with tf.variable_scope('generator',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('encoding'):
            #目前用的是lrelu，其实应该用elu，后面注意跟换
            with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #96
                conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
                conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
                #48
                conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
                conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
                conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
                #24
                conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
                conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
                conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
                #12
                conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
                conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
                conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
                #6
                conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
                conv13= slim.conv2d(conv12,160,kernel_size,scope='conv13')
            conv14= slim.conv2d(conv13,320,kernel_size,scope='conv14',activation_fn=tf.nn.tanh,normalizer_fn=None)
            #two path -feature -W Omega
            #1
            avgpool=slim.pool(conv14,[6,6],stride=6,pooling_type="AVG",scope='avgpool')
            output=avgpool#320维的向量

            # output = tf.nn.tanh(output)
            return output

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
def netG_deconder(feature,output_channel,reuse=False):
    '''
    @brief:
        feature:1*1*320+13+50
        pose:1*1*13r
        noise:1*1*50
    '''
    with tf.variable_scope('generator',reuse=reuse):
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose],activation_fn=tf.nn.elu,normalizer_fn=slim.batch_norm,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #先将vector组织为6*6*320的tensor#slim.batch_norm
                fc1 = slim.fully_connected(feature,6*6*320,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='fc1')
                #reshape the vector[n,6,6,320]
                inputs_img=tf.reshape(fc1,[-1,6,6,320])
                # print 'inputs_img',inputs_img.shape
                #6
                deconv0 = slim.conv2d_transpose(inputs_img,160,kernel_size,scope='deconv0')
                deconv1 = slim.conv2d_transpose(deconv0,filter_num*8,kernel_size,scope='deconv1')
                #12
                deconv2 = slim.conv2d_transpose(deconv1,filter_num*8,stride=2,kernel_size=kernel_size,scope='deconv2')

                deconv3 = slim.conv2d_transpose(deconv2,filter_num*4,kernel_size,scope='deconv3')
                deconv4 = slim.conv2d_transpose(deconv3,filter_num*6,kernel_size,scope='deconv4')
                #24
                deconv5 = slim.conv2d_transpose(deconv4,filter_num*6,stride=2,kernel_size=kernel_size,scope='deconv5')
                deconv6 = slim.conv2d_transpose(deconv5,filter_num*3,kernel_size,scope='deconv6')
                deconv7 = slim.conv2d_transpose(deconv6,filter_num*4,kernel_size,scope='deconv7')
                #48
                deconv8 = slim.conv2d_transpose(deconv7,filter_num*4,stride=2,kernel_size=kernel_size,scope='deconv8')
                deconv9 = slim.conv2d_transpose(deconv8,filter_num*2,kernel_size,scope='deconv9')
                deconv10= slim.conv2d_transpose(deconv9,filter_num*2,kernel_size,scope='deconv10')
                #96
                deconv11= slim.conv2d_transpose(deconv10,filter_num*2,stride=2,kernel_size=kernel_size,scope='deconv11')
                deconv12= slim.conv2d_transpose(deconv11,filter_num*1,kernel_size,scope='deconv12')
            #为什么放到外面就好了呢？
            deconv13= slim.conv2d_transpose(deconv12,output_channel,kernel_size,activation_fn=tf.nn.tanh,normalizer_fn=None,scope='deconv13',weights_initializer=tf.contrib.layers.xavier_initializer())

            output=deconv13
        variables = tf.contrib.framework.get_variables(vs)
        return output,variables

def netD_discriminator(image_input,class_nums=None,posenum=13,reuse=False):
    with tf.variable_scope('discriminator',reuse=reuse) as vs:
        kernel_size=[3,3]
        filter_num=32
        with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
            conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
            #24
            conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
            conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
            conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
            #12
            conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
            conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
            conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
            #6
            conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
            conv13= slim.conv2d(conv12,160,kernel_size,scope='conv13')
            #two path -feature -W Omegapredict_r_label
            #avg出来之后应该是1*1*320的tensor
            conv14= slim.conv2d(conv13,320,kernel_size,scope='conv14')
            avgpool=slim.pool(conv14,[6,6],stride=6,pooling_type="AVG",scope='avgpool')

            #3个输出
            ad_logits = slim.fully_connected(avgpool,1,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='GT_Fake')
            id_logits = slim.fully_connected(avgpool,class_nums,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='Identity')
            ps_logits = slim.fully_connected(avgpool,posenum,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='Pose')
#                output=[gt_fake,identity,pose]
#             output = gt_fake
            variables = tf.contrib.framework.get_variables(vs)
            return tf.nn.softmax(ad_logits),ad_logits\
                    ,tf.nn.softmax(id_logits),id_logits\
                    ,tf.nn.softmax(ps_logits),ps_logits\
                    ,variables


def Custom_netG_encoder(image_input,reuse=False):
    with tf.variable_scope('generator',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('encoding'):
            #目前用的是lrelu，其实应该用elu，后面注意跟换
            with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #96
                conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
                conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
                #48
                conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
                conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
                conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
                #24
                conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
                conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
                conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
                #12
                conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
                conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
                conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
                #6
                conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
                conv13= slim.conv2d(conv12,filter_num*6,kernel_size,scope='conv13')
                conv14= slim.conv2d(conv13,filter_num*10,kernel_size,scope='conv14')
            conv14= slim.conv2d(conv14,512,kernel_size,scope='conv15',normalizer_fn=None)
            #two path -feature -W Omega
            #1
            # avgpool=slim.pool(conv14,[6,6],stride=6,pooling_type="AVG",scope='avgpool')
            output=conv14# n * 6 * 6 *512维的向量

            # output = tf.nn.tanh(output)
            return output
def Custom_netG_decoder(featuremap,posemap,reuse=False):
    '''
    :param feature_input: n * 6 * 6 * featurechannels
    :param reuse:
    :return:
    '''
    with tf.variable_scope('generator',reuse=reuse):
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose],activation_fn=tf.nn.elu,normalizer_fn=inst_norm,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #先将vector组织为6*6*320的tensor#slim.batch_norm
                # fc1 = slim.fully_connected(feature_input,6*6*320,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='fc1')
                #reshape the vector[n,6,6,320]
                # inputs_img=tf.reshape(fc1,[-1,6,6,320])
                # print 'inputs_img',inputs_img.shape
                sample_add_zp = tf.concat([featuremap, posemap], 3)
                #6
                deconv0 = slim.conv2d_transpose(sample_add_zp,filter_num*10,kernel_size,scope='deconv0')
                deconv1 = slim.conv2d_transpose(deconv0,filter_num*8,kernel_size,scope='deconv1')
                #12
                deconv2 = slim.conv2d_transpose(deconv1,filter_num*8,stride=2,kernel_size=kernel_size,scope='deconv2')

                deconv3 = slim.conv2d_transpose(deconv2,filter_num*4,kernel_size,scope='deconv3')
                deconv4 = slim.conv2d_transpose(deconv3,filter_num*6,kernel_size,scope='deconv4')
                #24
                deconv5 = slim.conv2d_transpose(deconv4,filter_num*6,stride=2,kernel_size=kernel_size,scope='deconv5')
                deconv6 = slim.conv2d_transpose(deconv5,filter_num*3,kernel_size,scope='deconv6')
                deconv7 = slim.conv2d_transpose(deconv6,filter_num*4,kernel_size,scope='deconv7')
                #48
                deconv8 = slim.conv2d_transpose(deconv7,filter_num*4,stride=2,kernel_size=kernel_size,scope='deconv8')
                deconv9 = slim.conv2d_transpose(deconv8,filter_num*2,kernel_size,scope='deconv9')
                deconv10= slim.conv2d_transpose(deconv9,filter_num*2,kernel_size,scope='deconv10')
                #96
                deconv11= slim.conv2d_transpose(deconv10,filter_num*2,stride=2,kernel_size=kernel_size,scope='deconv11')
                deconv12= slim.conv2d_transpose(deconv11,filter_num*1,kernel_size,scope='deconv12')
            #为什么放到外面就好了呢？
            deconv13= slim.conv2d_transpose(deconv12,3,kernel_size,activation_fn=tf.nn.tanh,normalizer_fn=None,scope='deconv13',weights_initializer=tf.contrib.layers.xavier_initializer())

            output=deconv13

            return output

def Custom_netD_discriminator_adloss(image_input,reuse=False):
    with tf.variable_scope('discriminator/adloss',reuse=reuse) as vs:
        kernel_size=[3,3]
        filter_num=32
        with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
            conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
            #24
            conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
            conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
            conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
            #12
            conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
            conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
            conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
            #6
            conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
            conv13= slim.conv2d(conv12,filter_num*6,kernel_size,scope='conv13')
            conv14= slim.conv2d(conv13,filter_num*10,kernel_size,scope='conv14')
            #two path -feature -W Omegapredict_r_label
            #avg出来之后应该是1*1*320的tensor
            #3
            conv15= slim.conv2d(conv14,filter_num*10,stride=2,kernel_size=kernel_size,scope='conv15')
            conv16= slim.conv2d(conv15,filter_num*8,kernel_size,scope='conv16')
            conv17= slim.conv2d(conv16,filter_num*12,kernel_size,scope='conv17')

            # avgpool=slim.pool(conv14,[3,3],stride=6,pooling_type="AVG",scope='avgpool')
            shape_conv17 = tf.shape(conv17)
            linear17 = tf.reshape(conv17, [shape_conv17[0],1,1, shape_conv17[1] * shape_conv17[2] * shape_conv17[3]],name='reshape_fc')

            adlogits = slim.fully_connected(linear17,1,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='ad_soft')
            # variables = tf.contrib.framework.get_variables(vs)
            return tf.nn.softmax(adlogits),adlogits

def Custom_netD_discriminator_idloss(image_input,class_nums=None,reuse=False,usecontant=False):
    '''
    08-04 取消3*3之后的卷积 用512维的fc代替，并且用dropout
    :param image_input:
    :param class_nums:
    :param reuse:
    :param usecontant:这里的contant 希望网络可以抽取feature对identity-preserved 做贡献
    :return:
    '''
    with tf.variable_scope('discriminator/idloss',reuse=reuse) as vs:
        kernel_size=[3,3]
        filter_num=32
        with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
            conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
            #24
            conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
            conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
            conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
            #12
            conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
            conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
            conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
            #6
            conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
            conv13= slim.conv2d(conv12,filter_num*6,kernel_size,scope='conv13')
            conv14= slim.conv2d(conv13,filter_num*10,kernel_size,scope='conv14')
            #two path -feature -W Omegapredict_r_label
            #avg出来之后应该是1*1*320的tensor
            #3
            conv15= slim.conv2d(conv14,filter_num*10,stride=2,kernel_size=kernel_size,scope='conv15')
            conv16= slim.conv2d(conv15,filter_num*8,kernel_size,scope='conv16')
            conv17= slim.conv2d(conv16,filter_num*12,kernel_size,scope='conv17')

            # avgpool=slim.pool(conv14,[3,3],stride=6,pooling_type="AVG",scope='avgpool')
            # shape_conv17 = tf.shape(conv17)
            # linear17 = tf.reshape(conv17, [-1,1,1, 3456],name='reshape_fc')
            conv17 = tf.reshape(slim.flatten(conv17,scope='reshape'), [-1, 1, 1, 3456])
            fc1 = slim.fully_connected(conv17,512,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='fc')
            fc_drop=slim.dropout(fc1,is_training=True,scope='dropout')
            idlogits = slim.fully_connected(fc_drop,class_nums,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='id_soft')
            # variables = tf.contrib.framework.get_variables(vs)
            if usecontant:
                return tf.nn.softmax(idlogits), idlogits,fc1
            else:
                return tf.nn.softmax(idlogits,name='softmax'),idlogits

def Custom_netD_discriminator_psloss(image_input,posenum=3,reuse=False):
    '''
    08-04修改，使用avgpool减小网络大小
    08-05修改，inst_norm 替换为batchnorm
    :param image_input:
    :param posenum:
    :param reuse:
    :return:
    '''
    with tf.variable_scope('discriminator/psloss',reuse=reuse) as vs:
        kernel_size=[3,3]
        filter_num=32
        with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
            conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
            #24
            conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
            conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
            conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
            #12
            conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
            conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
            conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
            #6
            conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
            conv13= slim.conv2d(conv12,filter_num*6,kernel_size,scope='conv13')
            conv14= slim.conv2d(conv13,filter_num*10,kernel_size,scope='conv14')
            #two path -feature -W Omegapredict_r_label
            #avg出来之后应该是1*1*320的tensor
            #3
            # conv15= slim.conv2d(conv14,filter_num*10,stride=2,kernel_size=kernel_size,scope='conv15')
            # conv16= slim.conv2d(conv15,filter_num*8,kernel_size,scope='conv16')
            # conv17= slim.conv2d(conv16,filter_num*12,kernel_size,scope='conv17')

            avgpool=slim.pool(conv14,[6,6],stride=6,pooling_type="AVG",scope='avgpool')
            # shape_conv17 = tf.shape(conv17)
            # linear17 = tf.reshape(conv17, [-1,1,1, 3456],name='reshape_fc')

            pslogits = slim.fully_connected(avgpool,posenum,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='ps_soft')
            # variables = tf.contrib.framework.get_variables(vs)
            #conv11 means content structure feature
            return pslogits,conv14

def Custom_netG_pose_and_noise(input_featuremap,shape,input_pose,noise,reuse=False):
    with tf.variable_scope('generator/concat', reuse=reuse):

        size=shape[1]*shape[2]

        linear_ps=slim.fully_connected(input_pose,size*3,scope='fps',activation_fn=None, normalizer_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),)
        ps=tf.reshape(linear_ps,[-1,6,6,3],name='rsp_input_pose')


        ps_noise=tf.concat([ps,noise],3)
        return ps_noise

####--------------
def netG_encoder_gamma(image_input,reuse=False):
    '''
    08-04 删除了line reshape层
    :param image_input:
    :param reuse:
    :return:
    '''
    with tf.variable_scope('generator',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('encoding'):
            #目前用的是lrelu，其实应该用elu，后面注意跟换
            with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #96
                conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
                conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
                #48
                conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
                conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
                conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
                #24
                conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
                conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
                conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
                #12
                conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
                conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
                conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
                #6
                conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
                conv13 = slim.conv2d(conv12, filter_num * 6, kernel_size, scope='conv13')
                conv14 = slim.conv2d(conv13, filter_num * 10, kernel_size, scope='conv14')
                # two path -feature -W Omegapredict_r_label
                # avg出来之后应该是1*1*320的tensor
                # 3
                conv15 = slim.conv2d(conv14, filter_num * 10, stride=2, kernel_size=kernel_size, scope='conv15')
                conv16 = slim.conv2d(conv15, filter_num * 8, kernel_size, scope='conv16')
                conv17 = slim.conv2d(conv16, filter_num * 12, kernel_size, scope='conv17')

                # avgpool=slim.pool(conv14,[3,3],stride=6,pooling_type="AVG",scope='avgpool')
                conv17=tf.reshape(slim.flatten(conv17),[-1,1,1,3456])
                # shape_conv17 = tf.shape(conv17)
                # print 'shape_conv17',conv17
                # linear17 = tf.reshape(conv17,
                #                       [-1, 1, 1, 3456],
                #                       name='reshape_fc')
                logits = slim.fully_connected(conv17, 512, activation_fn=None, normalizer_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                scope='ps_soft')
            output=logits#512维的向量

            # output = tf.nn.tanh(output)
            return output
def netG_deconder_gamma(feature,output_channel,reuse=False):
    '''
    08-05 instanse norm
    @brief:
        feature:1*1*320+13+50
        pose:1*1*13r
        noise:1*1*50
    '''
    with tf.variable_scope('generator',reuse=reuse):
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose],activation_fn=tf.nn.elu,normalizer_fn=slim.batch_norm,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #先将vector组织为6*6*320的tensor#slim.batch_norm
                fc1 = slim.fully_connected(feature,3*3*384,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='fc1')
                #reshape the vector[n,6,6,320]
                inputs_img=tf.reshape(fc1,[-1,3,3,384])
                # print 'inputs_img',inputs_img.shape
                #3
                deconv01 = slim.conv2d(inputs_img,filter_num * 8,kernel_size, scope='deconv01')
                deconv02 = slim.conv2d(deconv01, filter_num * 10, kernel_size, scope='deconv02')

                #6
                deconv03 = slim.conv2d_transpose(deconv02, filter_num * 10, stride=2, kernel_size=kernel_size,scope='deconv03')
                deconv0 = slim.conv2d_transpose(deconv03,filter_num*6,kernel_size,scope='deconv0')
                deconv1 = slim.conv2d_transpose(deconv0,filter_num*8,kernel_size,scope='deconv1')
                #12
                deconv2 = slim.conv2d_transpose(deconv1,filter_num*8,stride=2,kernel_size=kernel_size,scope='deconv2')
                deconv3 = slim.conv2d_transpose(deconv2,filter_num*4,kernel_size,scope='deconv3')
                deconv4 = slim.conv2d_transpose(deconv3,filter_num*6,kernel_size,scope='deconv4')
                #24
                deconv5 = slim.conv2d_transpose(deconv4,filter_num*6,stride=2,kernel_size=kernel_size,scope='deconv5')
                deconv6 = slim.conv2d_transpose(deconv5,filter_num*3,kernel_size,scope='deconv6')
                deconv7 = slim.conv2d_transpose(deconv6,filter_num*4,kernel_size,scope='deconv7')
                #48
                deconv8 = slim.conv2d_transpose(deconv7,filter_num*4,stride=2,kernel_size=kernel_size,scope='deconv8')
                deconv9 = slim.conv2d_transpose(deconv8,filter_num*2,kernel_size,scope='deconv9')
                deconv10= slim.conv2d_transpose(deconv9,filter_num*2,kernel_size,scope='deconv10')
                #96
                deconv11= slim.conv2d_transpose(deconv10,filter_num*2,stride=2,kernel_size=kernel_size,scope='deconv11')
                deconv12= slim.conv2d_transpose(deconv11,filter_num*1,kernel_size,scope='deconv12')
            #为什么放到外面就好了呢？
            deconv13= slim.conv2d_transpose(deconv12,output_channel,kernel_size,activation_fn=tf.nn.tanh,normalizer_fn=None,scope='deconv13',weights_initializer=tf.contrib.layers.xavier_initializer())

            output=deconv13
        # variables = tf.contrib.framework.get_variables(vs)
        return output