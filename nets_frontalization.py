# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
from tensorflow.python.ops import math_ops
from resnet_yd import resnet_yd
from tensorflow.contrib.layers.python.layers import initializers
import sys
sys.path.insert(0,'./tf_inception_v1')
from inception_resnet_v1 import inference

def inference_recognition(inputs,classnums,reuse=True):
    _,_,logits = resnet_yd(inputs, reuse=True)
    with tf.variable_scope('recognition_resnet_fc',reuse=reuse):
        net=slim.fully_connected(logits,classnums,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='recognition_soft')
    return tf.nn.softmax(net),net,logits

def inference_recognition_inceptionv1(inputs,classnums,reuse=False,train=True):
    logits,_ = inference(inputs, keep_prob=0.8,phase_train=train,reuse=reuse)
    with tf.variable_scope('recognition_fc',reuse=reuse):
        net=slim.fully_connected(logits,classnums,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='recognition_soft')
    return tf.nn.softmax(net),net,logits

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
####--------------
def single_mfm_grop(data, filter_num, kernel_size, layer):
    with tf.variable_scope('single_mfm_group'):
        net = slim.conv2d(data, filter_num, kernel_size, scope = ('conv%s' % layer))
        net0,net1 = tf.split(net, 2, axis=3, name = ('split%s' % layer))
        net = tf.maximum(net0, net1, name = ('mfm%s' % layer))
    return net

def double_mfm_group(data, fiter_num, sec_fiter_num, kernel_size, layer):
    with tf.variable_scope('double_mfm_group'):
        net = slim.conv2d(data, fiter_num, [1,1], scope = ('conv%s_a' % layer))
        net0,net1 = tf.split(net, 2, axis=3, name = ('split%s_a' % layer))
        net = tf.maximum(net0, net1, name = ('mfm%s_a' % layer))
        net = slim.conv2d(net, sec_fiter_num, kernel_size, scope = ('conv%s' % layer))
        net0,net1 = tf.split(net, 2, axis=3, name = ('split%s' % layer))
        net = tf.maximum(net0, net1, name = ('mfm%s' % layer))
    return net

# def inference(data,class_nums, keep_prob, phase_train=True, weight_decay=0.0001,reuse=False):
#     with slim.arg_scope([slim.conv2d], stride=1,
#                 weights_initializer=initializers.xavier_initializer(uniform=False),
#                 weights_regularizer=slim.l2_regularizer(weight_decay),
#                 activation_fn=None,
#                 padding='SAME'):
#         with slim.arg_scope([slim.fully_connected],
#                             weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                             weights_regularizer=slim.l2_regularizer(weight_decay),
#                             activation_fn=None):
#             return lightcnn_9(data, class_nums,keep_prob, is_training=phase_train,reuse=reuse)

def lightcnn_9(data, class_nums,keep_prob=0.5, is_training=True,reuse=False):
    end_points = {}

    with tf.variable_scope('lightcnn_9',reuse=reuse):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = single_mfm_grop(data, 96, [5, 5], 1)
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID',scope='pool1')
            net = double_mfm_group(net, 96, 192, [3, 3], 2)
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
            net = double_mfm_group(net, 192, 384, [3, 3], 3)
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
            net = double_mfm_group(net, 384, 256, [3, 3], 4)
            net = double_mfm_group(net, 256, 256, [3, 3], 5)
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 512, scope='fc1')
            net0,net1 = tf.split(net, 2, axis=1, name = 'split_fc1')
            maxout = tf.maximum(net0, net1, name = 'mfm_fc1')
            net = slim.dropout(maxout, keep_prob, is_training=is_training, scope='dropout')
    logits = slim.fully_connected(net,class_nums,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='recognition_soft')
    return tf.nn.softmax(logits),logits,maxout

def merge_net_16(synthesis_input,image_input,reuse=False):
    with tf.variable_scope('merging',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        merge_image=tf.concat([synthesis_input,image_input],axis=3)
        print 'merge_image',merge_image
        #目前用的是lrelu，其实应该用elu，后面注意跟换
        with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(merge_image,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*4,kernel_size,scope='conv4')
            #24
            conv5 = slim.conv2d(conv4,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv5')
            conv6 = slim.conv2d(conv5,filter_num*8,kernel_size,scope='conv6')
            #12
            conv7 = slim.conv2d(conv6, filter_num * 16, stride=2, kernel_size=kernel_size, scope='conv7')
            conv8 = slim.conv2d(conv7, filter_num * 16, kernel_size, scope='conv8')
            # 24
            deconv1 = slim.conv2d_transpose(conv8, filter_num * 8, stride=2, kernel_size=kernel_size, scope='deconv1')

            deconv2 = slim.conv2d_transpose(deconv1, filter_num * 8, kernel_size, scope='deconv2')
            # 48
            deconv3 = slim.conv2d_transpose(deconv2, filter_num * 4, stride=2, kernel_size=kernel_size, scope='deconv3')

            deconv4 = slim.conv2d_transpose(deconv3, filter_num * 4, kernel_size, scope='deconv4')
            # 96

            deconv5 = slim.conv2d_transpose(deconv4, filter_num * 2, stride=2, kernel_size=kernel_size,
                                             scope='deconv5')

            deconv6 = slim.conv2d_transpose(deconv5, filter_num * 2, kernel_size, scope='deconv6')
            # 为什么放到外面就好了呢？
            # merge6 = tf.concat([conv1, deconv6], axis=3)
            deconv7 = slim.conv2d_transpose(deconv6, filter_num * 1, kernel_size, scope='deconv7')
            deconv8 = slim.conv2d_transpose(deconv7, filter_num * 1, kernel_size, scope='deconv8')
        deconv9 = slim.conv2d_transpose(deconv8, 3, kernel_size, activation_fn=tf.nn.tanh,
                                         normalizer_fn=None, scope='deconv9',
                                         weights_initializer=tf.contrib.layers.xavier_initializer())
        return deconv9

def merge_net_16_unet(synthesis_input,image_input,reuse=False):
    with tf.variable_scope('merging',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        merge_image=tf.concat([synthesis_input,image_input],axis=3)
        print 'merge_image',merge_image
        #目前用的是lrelu，其实应该用elu，后面注意跟换
        with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(merge_image,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*4,kernel_size,scope='conv4')
            #24
            conv5 = slim.conv2d(conv4,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv5')
            conv6 = slim.conv2d(conv5,filter_num*8,kernel_size,scope='conv6')
            #12
            conv7 = slim.conv2d(conv6, filter_num * 16, stride=2, kernel_size=kernel_size, scope='conv7')
            conv8 = slim.conv2d(conv7, filter_num * 16, kernel_size, scope='conv8')
            # 24
            deconv1 = slim.conv2d_transpose(conv8, filter_num * 8, stride=2, kernel_size=kernel_size, scope='deconv1')
            merge1=tf.concat([conv6,deconv1],axis=3)
            deconv2 = slim.conv2d_transpose(merge1, filter_num * 8, kernel_size, scope='deconv2')
            # 48
            merge2=tf.concat([conv5,deconv2],axis=3)
            deconv3 = slim.conv2d_transpose(merge2, filter_num * 4, stride=2, kernel_size=kernel_size, scope='deconv3')
            merge3 = tf.concat([conv4, deconv3], axis=3)
            deconv4 = slim.conv2d_transpose(merge3, filter_num * 4, kernel_size, scope='deconv4')
            # 96
            merge4 = tf.concat([conv3, deconv4], axis=3)
            deconv5 = slim.conv2d_transpose(merge4, filter_num * 2, stride=2, kernel_size=kernel_size,
                                             scope='deconv5')
            merge5 = tf.concat([conv2, deconv5], axis=3)
            deconv6 = slim.conv2d_transpose(merge5, filter_num * 2, kernel_size, scope='deconv6')
            # 为什么放到外面就好了呢？
            merge6 = tf.concat([conv1, deconv6], axis=3)
            deconv7 = slim.conv2d_transpose(merge6, filter_num * 1, kernel_size, scope='deconv7')
            deconv8 = slim.conv2d_transpose(deconv7, filter_num * 1, kernel_size, scope='deconv8')
        deconv9 = slim.conv2d_transpose(deconv8, 3, kernel_size, activation_fn=tf.nn.tanh,
                                         normalizer_fn=None, scope='deconv9',
                                         weights_initializer=tf.contrib.layers.xavier_initializer())
        return deconv9

def netG_encoder_gamma(image_input,classnums=337,reuse=False):
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
                conv13 = slim.conv2d(conv12, filter_num * 6, kernel_size, scope='conv13')
                conv14 = slim.conv2d(conv13, filter_num * 10, kernel_size, scope='conv14')
                # two path -feature -W Omegapredict_r_label
                # avg出来之后应该是1*1*320的tensor
                # 3
                conv15 = slim.conv2d(conv14, filter_num * 10, stride=2, kernel_size=kernel_size, scope='conv15')
                conv16 = slim.conv2d(conv15, filter_num * 8, kernel_size, scope='conv16')
                conv17 = slim.conv2d(conv16, filter_num * 12, kernel_size, scope='conv17')

                conv17=slim.flatten(conv17)
            logits = slim.fully_connected(conv17, 512, activation_fn=None, normalizer_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            scope='pr_soft')
            idlogits = slim.fully_connected(logits, classnums, activation_fn=None, normalizer_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                scope='id_soft')
            output=logits#512维的向量

            return idlogits,idlogits,output
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
            with slim.arg_scope([slim.conv2d_transpose],activation_fn=tf.nn.elu,normalizer_fn=inst_norm,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
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
        return output


def Custom_netD_discriminator_adloss(image_input,reuse=False):
    with tf.variable_scope('discriminator/adloss',reuse=reuse) as vs:
        kernel_size=[3,3]
        filter_num=32
        with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            net = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            net = slim.conv2d(net,filter_num*2,kernel_size,scope='conv2')
            #48
            net = slim.conv2d(net,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
            net = slim.conv2d(net,filter_num*2,kernel_size,scope='conv4')
            net = slim.conv2d(net,filter_num*4,kernel_size,scope='conv5')
            # 24
            net = slim.conv2d(net,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
            net = slim.conv2d(net,filter_num*3,kernel_size,scope='conv7')
            net = slim.conv2d(net,filter_num*6,kernel_size,scope='conv8')
            #12
            net = slim.conv2d(net,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
            net= slim.conv2d(net,filter_num*4,kernel_size,scope='conv10')
            net= slim.conv2d(net,filter_num*8,kernel_size,scope='conv11')
            #6
            net = slim.conv2d(net,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
            net = slim.conv2d(net,filter_num*6,kernel_size,scope='conv13')
            net = slim.conv2d(net,filter_num*10,kernel_size,scope='conv14')
            #two path -feature -W Omegapredict_r_label
            #avg出来之后应该是1*1*320的tensor
            #3
            net = slim.conv2d(net,filter_num*10,stride=2,kernel_size=kernel_size,scope='conv15')
            net = slim.conv2d(net,filter_num*8,kernel_size,scope='conv16')
            net = slim.conv2d(net,filter_num*12,kernel_size,scope='conv17')
            # net = slim.conv2d(net,filter_num*10,[1,1],scope='NiN1')
            # net = slim.conv2d(net,1, [1, 1], scope='NiN2',activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02))

            avgpool=slim.pool(net,[3,3],stride=3,pooling_type="AVG",scope='avgpool')
            adlogits = slim.fully_connected(slim.flatten(avgpool),1,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='ad_soft')

            return adlogits#1*1

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

def merge_net_16_unet_v2(synthesis_input,image_input,reuse=False):
    with tf.variable_scope('merging',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        merge_image=tf.concat([synthesis_input,image_input],axis=3)
        print 'merge_image',merge_image
        #目前用的是lrelu，其实应该用elu，后面注意跟换
        with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #96
            conv1 = slim.conv2d(merge_image,filter_num*2,kernel_size,normalizer_fn=None,scope='conv1')
            conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
            #48
            conv3 = slim.conv2d(conv2,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv3')
            conv4 = slim.conv2d(conv3,filter_num*4,kernel_size,scope='conv4')
            #24
            conv5 = slim.conv2d(conv4,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv5')
            conv6 = slim.conv2d(conv5,filter_num*8,kernel_size,scope='conv6')
            #12
            conv7 = slim.conv2d(conv6, filter_num * 16, stride=2, kernel_size=kernel_size, scope='conv7')
            conv8 = slim.conv2d(conv7, filter_num * 16, kernel_size, scope='conv8')
            # 24
            deconv1 = slim.conv2d_transpose(conv8, filter_num * 8, stride=2, kernel_size=kernel_size, scope='deconv1')
            merge1=tf.concat([conv6,deconv1],axis=3)
            deconv2 = slim.conv2d_transpose(merge1, filter_num * 8, kernel_size, scope='deconv2')
            # 48
            merge2=tf.concat([conv5,deconv2],axis=3)
            deconv3 = slim.conv2d_transpose(merge2, filter_num * 4, stride=2, kernel_size=kernel_size, scope='deconv3')
            merge3 = tf.concat([conv4, deconv3], axis=3)
            deconv4 = slim.conv2d_transpose(merge3, filter_num * 4, kernel_size, scope='deconv4')
            # 96
            merge4 = tf.concat([conv3, deconv4], axis=3)
            deconv5 = slim.conv2d_transpose(merge4, filter_num * 2, stride=2, kernel_size=kernel_size,
                                             scope='deconv5')
            merge5 = tf.concat([conv2, deconv5], axis=3)
            conv6 = slim.conv2d(merge5, filter_num * 2, kernel_size, scope='deconv6')
            # 为什么放到外面就好了呢？
            merge6 = tf.concat([conv1, conv6], axis=3)
            conv7 = slim.conv2d(merge6, filter_num * 2, kernel_size, scope='deconv7')
            conv8 = slim.conv2d(conv7, filter_num * 2, kernel_size, scope='deconv8')
            conv8_1 = slim.conv2d(conv8,filter_num,kernel_size,scope='deconv8_1')
            conv8_2 = slim.conv2d(conv8_1,filter_num,kernel_size,scope='deconv8_2')
        conv9 = slim.conv2d(conv8_2, 3, kernel_size, activation_fn=tf.nn.tanh,
                                         normalizer_fn=None, scope='deconv9',padding='SAME',
                                         weights_initializer=tf.contrib.layers.xavier_initializer())
        return conv9