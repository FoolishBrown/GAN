# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:53:33 2016

@author: Saber
"""
'''
对比特征值，feature_test和feature_train
'''
import sklearn.metrics.pairwise as pw
import numpy as np

def compare_pair_features(feature_test,feature_train,flag,metric='cosine'):
    '''
    @feature_test:测试样本
    @feature_train:训练样本
    @metric:From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
    These metrics support sparse matrix inputs.
    From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, 
    ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, 
    ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
    ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    @flag: 0为pairwise_distances函数，1为cosine_similarity函数
    返回值  为对应向量的距离数组
    '''
    test_num=len(feature_test)
    if flag==0:
        distance=pw.pairwise_distances(feature_test, feature_train, metric=metric)
    elif flag==1:
        distance=pw.cosine_similarity(feature_test, feature_train)
    predicts=np.empty((test_num,))
    for i in range(test_num):
          predicts[i]=distance[i][i]
    return predicts

def compare_one2N_features(feature_test,feature_train,flag,metric='cosine'):
    '''
    @feature_test:测试样本
    @feature_train:训练样本
    @metric:From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
    These metrics support sparse matrix inputs.
    From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, 
    ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, 
    ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
    ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    @flag: 0为pairwise_distances函数，1为cosine_similarity函数
    返回值  为对应向量的距离数组
    '''
    data_num=len(feature_train)#每个组中的样本数目
    print 'feature_train：',feature_train.shape
    print 'feature_train_num:',data_num
    #扩充test dimension 和本次测试data维度相同
    one_plus_feature= np.tile(feature_test,data_num)  
    one_plus_feature=np.reshape(one_plus_feature,(data_num,-1))
    print 'one_plus_feature:',one_plus_feature.shape
    #计算维度，结果应该与data样本数相同
    if flag==0:
        distance=pw.pairwise_distances(one_plus_feature, feature_train, metric=metric)
    elif flag==1:
        distance=pw.cosine_similarity(one_plus_feature, feature_train)
    print distance.shape #输出结果维度 
    predicts=np.empty((data_num,))#用来存储预测结果
    for i in range(data_num):
          predicts[i]=distance[i][i]#比对过的样本数+n来控制位移
    return predicts
def find_rank_predict(predicts_total,rank_num,flag=False):
    '''
    @flag=True 从大到小，False 从小到大
    @predicts_total：预测结果
    @rank_num：需要的排序数目
    @return index 返回在data中的位置
    '''
    rankedistance=sorted(predicts_total,reverse=flag)[:rank_num]
    predicts_total=np.ndarray.tolist(predicts_total)
#    index=np.empty((rank_num,1))
    index=[]
    for j in range(rank_num):
        index.append(predicts_total.index(rankedistance[j]))
    return index