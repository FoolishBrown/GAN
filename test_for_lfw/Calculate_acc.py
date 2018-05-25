# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:55:32 2016

@author: Saber
"""
import numpy as np
#测试准确度，1为相同0为不同，
def calculate_accuracy_pairs(distance,labels,num,threshold=[0.2,0.9]):

    '''
    @distance：向量间的距离
    @labels：每一对pair的比例
    @num：pair的数目
    #计算识别率,
    选取阈值，计算识别率
    返回最佳识别率
    '''    
    accuracy = []
    thresholdlist=[]
    predict = np.empty((num))
    for th in threshold:
        for i in range(num):
            if distance[i] >= th:
                 predict[i] =0
            else:
                 predict[i] =1
        predict_right =0.0
        for i in range(num):
            if predict[i]==(int(labels[i])):
              predict_right = 1.0+predict_right
        current_accuracy = (predict_right/num)
        accuracy.append(current_accuracy)
        index=accuracy.index(np.max(accuracy))
    maxth=threshold[index]
    print maxth
       # print accuracy
    return np.max(accuracy),maxth

def nomalization(predicts,test_num):
    '''
    @predicts:需要正则化的 相似度
    @test_num:样本数目
    '''
    for i in range(test_num):
                predicts[i]=(predicts[i]-np.min(predicts))/(np.max(predicts)-np.min(predicts))
    return predicts
