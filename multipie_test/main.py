# -*- coding: utf-8 -*-
"""

@author: Saber
"""
import numpy as np
import os
import sys
import scipy.io as scio
import sklearn.metrics as skm
import argparse
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
#
import readimglist
import CompareFeature
import readconf


def parse_arguments(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', type=str, help='Test Section name in Config')
    return parser.parse_args(arg)


class LFWtest:
    def __init__(self, section, modelpath=None, model=None, gpulist=None, save_path=None, netpath=None, funcname=None,
                 netname=None):
        # 模型信息读取
        self.modelroot = readconf.getConfig(section, "root")
        self.model = readconf.getConfig(section, "model")
        self.dataroot = save_path
        if modelpath != None:
            self.dataroot = save_path
            self.model = model
            self.modelroot = modelpath
            os.environ['CUDA_VISIBLE_DEVICES'] = gpulist
            self.funcname = funcname
            self.netpath = netpath
            self.netname = netname
        else:
            self.gpulist = readconf.getConfig(section, 'gpu')
            # self.dataroot = readconf.getConfig(section, "imgroot")
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpulist
            self.netname = readconf.getConfig(section, "netname")
            self.funcname = readconf.getConfig(section, "funcname")
            self.netpath = readconf.getConfig(section, "netpath")
        self.size = int(readconf.getConfig(section, "size"))
        self.channel = int(readconf.getConfig(section, 'channel'))
        self.output = int(readconf.getConfig(section, "outputsize"))
        self.mode = readconf.getConfig(section, 'mode')
        sys.path.insert(0, self.netpath)
        import_net = "import {} as net".format(self.netname)
        exec import_net
        self.frfunction = getattr(net, self.funcname)
        # LFW读取
        self.left = readconf.getConfig('LFWtest', "left")
        self.right = readconf.getConfig('LFWtest', 'right')
        self.label = readconf.getConfig('LFWtest', "label")
        self.metric = readconf.getConfig('LFWtest', 'metric')
        self.distance = int(readconf.getConfig('LFWtest', 'distance'))
        self.batchsize = int(readconf.getConfig('LFWtest', 'batchsize'))
        labelfile = open(self.label, 'r')
        label_rows = labelfile.readlines()
        self.labels = []
        for label_sub in label_rows:
            self.labels.append(int(label_sub.strip('\n')))
        print   self.mode
        if self.mode == 'tf':
            self.extract = self.tf_extract
            # self.extract=getattr(self,'tf_extract')
            self.tf_inferece()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            try:
                self.sess.run(tf.global_variables_initializer())
            except:
                self.sess.run(tf.initialize_all_variables())
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, os.path.join(self.modelroot, self.model))

            # ----------------------------tf-----------

    def tf_inferece(self):
        self.batch_data = tf.placeholder(dtype=tf.float32, shape=[None, self.size, self.size, self.channel],
                                         name='input')

        _, _, self.feature = self.frfunction(self.batch_data)
        self.feature = tf.reshape(self.feature, [-1, self.output])

    def tf_extract(self, data):
        features = self.sess.run(self.feature, feed_dict={
            self.batch_data: data
        })
        return features

    # ----------------------------------------------
    def getfeatrues(self, func, filename):
        X, X_num = func(filename, size=[self.size, self.size], root=self.dataroot)
        features = np.zeros((X_num, self.output), dtype='float32')
        img_batch = []
        for i in range(X_num):
            img_batch.append(X[i])
            if len(img_batch) == self.batchsize or i == X_num - 1:
                feature_batch = self.extract(img_batch)
                print '%d images processed' % (i + 1,)
                features[i - len(img_batch) + 1:i + 1, :] = feature_batch.copy()
                img_batch = []
        features = np.asarray(features, dtype='float32')
        return features

    def setdata2model_Manual(self, func):
        '''
        @brief:func:可选read_imagelist_3channel或者read_imagelist_1channel
        对应RGB和Grey图像
        '''
        feature_left = self.getfeatrues(func, self.left)
        feature_right = self.getfeatrues(func, self.right)
        assert (feature_left.shape == feature_right.shape)
        return feature_left, feature_right

    def savemat(self, feature_left, feature_right):
        scio.savemat(self.modelroot + os.path.splitext(self.model)[0] + '_lfw.mat',
                     {'features_left': feature_left, 'features_right': feature_right, 'labels': self.labels})

    def readmat(self):
        LFWdata = scio.loadmat(self.modelroot + os.path.splitext(self.model)[0] + '_lfw.mat')
        return LFWdata['features_left'], LFWdata['features_right']

    def accuracy(self, Grey=False, save=False, exist=1):
        '''
        @brief：
        flag=0特征已经存在，可以直接读取
        save 是否保存
        grey 是不是灰度图像
        '''
        feature1 = []
        feature2 = []
        if not exist:
            if Grey:
                feature1, feature2 = self.setdata2model_Manual(readimglist.read_imagelist_1channel)
            else:
                feature1, feature2 = self.setdata2model_Manual(readimglist.read_imagelist_3channel)
            if save:
                self.savemat(feature1, feature2)
        else:
            feature1, feature2 = self.readmat()
        predicts = np.zeros((len(feature1), 1))
        predict = CompareFeature.compare_pair_features(feature1, feature2, flag=self.distance, metric=self.metric)
        for i in range(len(predict)):
            predicts[i] = predict[i]
        accuracy, threshold = self.train_kflod(predicts)
        print "10-fold accuracy is:\n{}\n".format(accuracy)
        print "10-fold threshold is:\n{}\n".format(threshold)
        print "mean threshold is:%.4f\n", np.mean(threshold)
        print "mean is:%.4f, var is:%.4f", np.mean(accuracy), np.std(accuracy)
        return np.mean(accuracy), np.std(accuracy), np.mean(threshold)

    def draw(self, predicts, title):
        fpr, tpr, thresholds = skm.roc_curve(self.labels, predicts)
        '''
        画ROC曲线图
        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic using: ' + title)
        plt.legend(loc="lower right")
        #    plt.show()
        plt.savefig(self.modelroot + '\\' + os.path.splitext(self.model)[0] + title + '.png')

    def find_best_threshold(self, thresholds, predicts, labels):
        best_threshold = best_acc = 0
        for threshold in thresholds:
            accuracy = self.eval_acc(threshold, predicts, labels)
            if accuracy >= best_acc:
                best_acc = accuracy
                best_threshold = threshold
        return best_threshold

    def eval_acc(self, threshold, predicts, labels):
        y_true = []
        y_predict = []
        for d, l in zip(predicts, labels):
            same = 0 if float(d) > threshold else 1  # 小于阈值的是同一个人
            y_predict.append(same)
            y_true.append(int(l))
        y_true = np.array(y_true)
        y_predict = np.array(y_predict)
        accuracy = accuracy_score(y_true, y_predict)
        return accuracy

    def train_kflod(self, predicts):
        print("...Computing accuracy.")
        folds = KFold(n=6000, n_folds=10, shuffle=False)
        thresholds = np.arange(-1.0, 1.0, 0.005)
        accuracy = []
        thd = []
        predicts = np.asarray(predicts)
        self.labels = np.asarray(self.labels)
        for idx, (train, test) in enumerate(folds):
            print "processing fold {}...".format(idx)
            best_thresh = self.find_best_threshold(thresholds, predicts[train], self.labels[train])
            accuracy.append(self.eval_acc(best_thresh, predicts[test], self.labels[test]))
            thd.append(best_thresh)
        return accuracy, thd


def main(modelpath=None, model=None, gpu=None, savepath=None, netpath=None, funcname=None, netname=None):
    section = "facenet_lfw"
    lfwtestdemo = LFWtest(section, modelpath=modelpath, model=model, save_path=savepath, gpulist=gpu, netpath=netpath,
                          funcname=funcname, netname=netname)
    ifsave = bool(readconf.getConfig(section, "ifsave"))
    ifexist = int(readconf.getConfig(section, "ifexist"))
    channel = int(readconf.getConfig(section, "channel"))
    if channel == 3:
        Grey = False
    else:
        Grey = True
    return lfwtestdemo.accuracy(save=ifsave, exist=ifexist, Grey=Grey)


def main_():
    result_path2 = r'/world/data-gpu-58/wangyuequan/data_sunkejia/lfw_synthesis_temp/lfw_synthesis/MultiPIE_Unet_finetun2/'
    # list_path = []
    for a, b, c in os.walk(result_path2):
        # list_path = b
        for path in b:
            if 'syn' in path:
                f_write = open('./lfwtest_info.txt', 'a')
                mAP, mTd, mTh = main(savepath=result_path2 + path, gpu='0', netpath='./', netname='resnet_yd',
                                     funcname='resnet_yd')
                f_write.write("{},{},{},{}\n".format(path, mAP, mTd, mTh))
                f_write.close()
        break


if __name__ == '__main__':
    main_()












