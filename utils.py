#-*- coding:utf-8 -*-
import scipy
import os
import tensorflow as tf
from PIL import Image
import random
from skimage import io
from skimage import color
from skimage import transform
import numpy as np
import threading
import Queue
import cv2
import math
from glob import glob
import sklearn.metrics.pairwise as pw
import copy

class ImageReader_Customize(object):
    def __init__(self,data_path='',data_label_txt='',label_nums=0,separator=' ',ext='jpg',batch_size=128,
                 input_size=96,output_channel=3,output_size=96,str_type=0,random_crop=False,validtionlabel=0,
                 random_images=False,random_seed=None,thread_nums=4,center_crop=False,queue_size=24,flip=False,
                 frontal_label=6,dual_pairmode=False):
        self.validation_label=validtionlabel#当做validation集的时候一定要有的部分！！！！作为validationlabel的偏移量
        self.data_path=data_path
        self.ext=ext
        self.frontal_label=frontal_label
        self.batch_size=batch_size
        self.input_size=input_size
        self.output_size=output_size
        self.output_channel=output_channel
        self.str_type=str_type
        self.random_crop=random_crop
        self.seed=random_seed
        self.thread_nums=thread_nums
        #data vs multi-labels  list
        self.labels_nums=label_nums
        self.data_label_txt=data_label_txt
        self.multi_labels=[]
        self.separator=separator
        #flip的时候Label 有负值，将负值的Label转成abs 并且flip im
        self.flip=flip
        self.dual_pairmode=dual_pairmode


        self.center_crop=center_crop
        self.random_images=random_images
        self.q = Queue.Queue(queue_size)
        self.q_in=Queue.Queue(queue_size)

        self._buildreader()


    def _buildreader(self):
        if self.output_channel==3:
            self.is_grey=False
        else:
            self.is_grey=True

        #init imagelist
        self.imgpath_list=[]
        #mk imagelist from directory
        if self.labels_nums==0 and self.data_label_txt=='':
            print 'mkpath_list'
            self._mkpath_list()
        else:#mk imagelist from a txt list
            print 'mkpath txt'
            self.multi_labels = [[] for _ in range(self.labels_nums)]
            self._mkpath_list_fromtxt()
            self.class_nums = len(list(set(self.multi_labels[0])))
            self.class_list=sorted(list(set(self.multi_labels[0])))
            print 'class_nums',self.class_nums
            if self.labels_nums>=2:
                self.pose_nums=len(list(set(self.multi_labels[1])))
                print 'pose nums',self.pose_nums
            if self.labels_nums>=3:
                self.light_nums = len(list(set(self.multi_labels[2])))
                print 'light_nums',self.light_nums
            if self.labels_nums>=4:
                self.session_nums = len(list(set(self.multi_labels[3])))
                print 'session_nums',self.session_nums
                # self.class_nums+=self.validation_label
        self.multi_labels = np.array(self.multi_labels, int)
        print 'multi_labels',self.multi_labels.shape

        #images numbers
        self.img_nums=len(self.imgpath_list)
        # print self.img_nums
        # print self.class_nums
        self.imgpath_list=np.asarray(self.imgpath_list)
        #threading argvment
        self.lock = threading.Lock()
        self.batch_idxs=self.img_nums / self.batch_size +1
        self.sub_nums = self.img_nums / self.thread_nums + 1

        #random list
        self.random_imgpath_list=np.asarray(self.imgpath_list).copy()
        self.random_multi_labels=np.asarray(self.multi_labels).copy()
        self.task=[]
        #排序 imglist multi_labels
        self._sort_list_label()
        self.lock=threading.Lock()

        #辅助random取到尽可能多的可能
        self.identity_idx=np.zeros([self.class_nums],np.int32)
        self.identity_im_idx={}

        for id_i in xrange(self.class_nums):
            _, _, count_im = self._get_one_person_list_label(id_i)
            new_zeros=np.zeros([count_im])
            self.identity_im_idx[id_i]=new_zeros.copy()
    def _mkpath_list_fromtxt(self):
        if os.path.isfile(self.data_label_txt):
            print 'INFO: reading img list from {}'.format(self.data_label_txt)
            f_data = open(self.data_label_txt,'r')
            f_lists = f_data.readlines()
            self._split_path_label(f_lists)
        else:
            print 'ERROR: path error!{} is not a file'.format(self.data_label_txt)
    def _split_path_label(self,filelines):

        for line in filelines:
            line_l=line.strip().split(self.separator)#path label1 label2...
            try:
                assert len(line_l)>=self.labels_nums+1
            except AssertionError :
                print line
            if os.path.isfile(line_l[0]):
                path=line_l[0]
            else:
                #path
                path=self.data_path+'/'+line_l[0]
            # path=os.path.join(self.data_path,line_l[0])
            path=path.replace('\\','/')
            # print path
            if os.path.isfile(path):
                self.imgpath_list.append(path)
                #multi-labels
                for i in xrange(self.labels_nums):
                    if i==0:
                        self.multi_labels[i].append(int(line_l[i+1])-self.validation_label)
                    else:
                        self.multi_labels[i].append(line_l[i+1])
        print 'INFO: make pathlist and labellist finished!!'

    def _mkpath_list(self):
        if not os.path.exists(self.data_path):
            print 'ERROR:{} is not exist'.format(self.data_path)
            return False
        self.imgpath_list = glob('{}/*.{}'.format(self.data_path, self.ext))

    def read_path_getdata(self):

        #decoder settings
        for ext in ["jpg","jpeg","png"]:

            if self.ext == 'jpg' and self.ext =='jpeg':
                tf_decode = tf.image.decode_jpeg
            elif self.ext == 'png':
                tf_decode = tf.image.decode_png
            else:
                print 'ERROR: ext:{}'.format(self.ext)
                return False
        with Image.open(self.imgpath_list[0]) as img:
            w , h = img.size
            shape = [ h, w, self.output_channel]
        filename_queue = tf.train.string_input_producer(
            list(self.data_list),shuffle = self.random_data,seed = self.seed)
        reader = tf.WholeFileReader()
        filename, data =reader.read(filename_queue)
        image = tf_decode(data, channels = 3)
        if self.is_grey:
            image =tf.image.rbg_to_greyscale(image)
        image.set_shape(shape)
        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3*self.batch_size*self.thread_nums

        queue = tf.train.shuffle_batch([image],batch_size=self.batch_size,num_threads=self.thread_nums,
                                       capacity=capacity,min_after_dequeue=min_after_dequeue,name='synthetic_input')
        queue=tf.to_float(queue)
        return self.norm_img(queue)

    def norm_img(self, img):
        return np.array(img)/127.5 - 1
        # return (img - img.mean()) / img.std()#试一下看看

    def _im_read(self,imagepath):
        # if(self.is_grey):
        #     return scipy.misc.imread(imagepath,flatten= True).astype(np.float)
        # else:
        #     return scipy.misc.imread(imagepath).astype(np.float)
        return cv2.imread(imagepath,not self.is_grey)

    def random_data(self):
        # random data
        # import sys
        # from signal import SIGTERM
        # pid=os.fork()
        # sys.exit(0)
        # os.kill(pid,SIGTERM)
        # while len(self.task):
        #     t = self.task.pop(0)
        #     t.stopped=True
        #     t.join()
        rng_state = np.random.get_state()
        np.random.shuffle(self.random_imgpath_list)
        for i in xrange(self.labels_nums):
            np.random.set_state(rng_state)
            np.random.shuffle(self.random_multi_labels[i])
            # self.enqueueStart()

    def enqueueStart(self):
        if not self.dual_pairmode:
            self.random_data()
        print 'totall images',len(self.imgpath_list)
        for i in xrange(self.thread_nums):
            end=min((i+1)*self.sub_nums,self.img_nums)
            subdata=self.random_imgpath_list[i:end]
            sublabel=[]
            if self.labels_nums:
                sublabel=self.random_multi_labels[:,i:end]
            t = threading.Thread(target=self._enqueueSample,args=(subdata,sublabel))
            t.daemon =True
            t.start()
            self.task.append(t)

    def _enqueueSample(self,subpath,sublabel=[]):
        len_sub=len(subpath)
        while True:
            count=0
            random_start=np.random.randint(0,len_sub-self.batch_size-10)
            img_batch=np.zeros((self.batch_size,self.output_size,self.output_size,self.output_channel),dtype=np.float32)
            # label_batch=np.zeros((self.batch_size,self.labels_nums),dtype=np.int)
            label_batch=None


            # self.q.put((subpath[random_start:random_start+self.batch_size], label_batch))
            if self.dual_pairmode:
                label_batch=np.zeros((self.labels_nums,self.batch_size))
                while count < self.batch_size/2:
                    if sublabel[0][random_start+count]==sublabel[0][random_start+count+1]:
                        label_1=random_start+count
                        label_2=random_start+count+1
                        try:
                            im1=self._get_images(subpath[label_1],sublabel[1][label_1])
                            im2=self._get_images(subpath[label_2],sublabel[1][label_2])
                        except:
                            im1 = self._get_images(subpath[label_1])
                            im2 = self._get_images(subpath[label_2])
                        label_batch[0,count]=sublabel[0][label_1]
                        label_batch[1,count]=sublabel[1][label_1]
                        label_batch[0,count+(self.batch_size/2)]=sublabel[0][label_2]
                        label_batch[1,count+(self.batch_size/2)]=sublabel[1][label_2]
                        # print count
                        img_batch[count]=im1[:,:,:]
                        img_batch[count+(self.batch_size/2)]=im2[:,:,:]
                        count+=1

                    else:
                        random_start+=1
            else:
                # path_l=[]
                if not sublabel == []:
                    label_batch = sublabel[:, random_start:random_start + self.batch_size].copy()
                for path in subpath[random_start:random_start+self.batch_size]:
                    try:
                        im=self._get_images(path,label_batch[1][count])
                    except:
                        im=self._get_images(path)
                    img_batch[count]=im[:,:,:]
                    count+=1
                    # path_l.append(path)
            self.q.put((img_batch,self._get_labels(label_batch)))


    def read_data_batch(self):
        if self.labels_nums!=0:
            assert (self.img_nums==len(self.multi_labels[0]))
        return self.q.get()

    def _get_labels(self,label):
        label1=label.copy()
        if self.flip:
            return self.fliplabel(label1)
        else:
            return label1

    def _get_images(self,path,label=0):
        image = self._im_read(path)
        if self.flip:
            image=self.flipimage(image,label)
        return self._transform(image)

    def _transform(self,image):
        if self.random_crop and self.input_size!=self.output_size:
            # print 'randomcrop'
            cropped_image = self._random_crop(image)
        elif self.center_crop and self.input_size !=self.output_size:
            crop_pix = int(round(self.input_size-self.output_size)/2.0)
            cropped_image = scipy.misc.imresize(image[crop_pix:crop_pix+self.output_size,crop_pix:crop_pix+self.output_size]
                                                ,[self.output_size,self.output_size])
        else:
            cropped_image = cv2.resize(image,dsize=(self.output_size,self.output_size),interpolation=cv2.INTER_CUBIC)
        return  self.norm_img(cropped_image).reshape([self.output_size,self.output_size,self.output_channel])

    def _random_crop(self,images):#--------------------------
        if images.shape[0]>self.input_size:
            images=cv2.resize(images,dsize=(self.input_size,self.input_size),interpolation=cv2.INTER_CUBIC)
        images=images.reshape([self.input_size,self.input_size,self.output_channel])
        offsetmax=self.input_size-self.output_size
        random_w=np.random.randint(0,offsetmax)
        random_h=np.random.randint(0,offsetmax)

        return images[random_w:random_w+self.output_size,random_h:random_h+self.output_size,:]

    def _sort_list_label(self):
        '''
        @brief:将乱序的List 排序
        :return:
        '''
        #默认 axis=1 按行排序
        sort_arr=np.argsort(self.multi_labels[0],axis=0)
        self.multi_labels=self.multi_labels[:,sort_arr]
        self.imgpath_list=self.imgpath_list[sort_arr]
        # return self.multi_labels,self.imgpath_list

    def _get_one_person_list_label(self,identity,ifshuffle=False):
        #获得指定某个identity的image_path 和 labels
        # imagelist 需要是有序表才能这么操作
        start = np.where(self.multi_labels[0] == self.class_list[identity])[0][0]
        # print start
        if identity < self.class_nums - 1:
            # print np.where(self.multi_labels[0] == identity + 1)
            end = np.where(self.multi_labels[0] == self.class_list[identity+1])[0][0]
        else:
            end = self.img_nums
        # print start,end
        count = end-start
        # count = min(self.batch_size, end - start)
        if ifshuffle:
            if self.lock.acquire():
                rng_state = np.random.get_state()
                np.random.shuffle(self.imgpath_list[start:start+count])
                for i in xrange(self.labels_nums):
                    np.random.set_state(rng_state)
                    np.random.shuffle(self.multi_labels[i,start:start+count])
                self.lock.release()
        return self.imgpath_list[start:start+count],self.multi_labels[:,start:start+count],count

    def get_same_person_batch(self,idseed=None,haslabel=False):
        '''
        :param seed: randomseed 指定某一个人
        :param ifrandom: 指定是否打乱数据(暂时没用)
        :return:
        '''

        if idseed==None:
            # np.random.seed(seed)
            identity = np.random.randint(0, self.class_nums)
            image_list,label_list,count=self._get_one_person_list_label(identity)
            assert(len(image_list)==count)
            imagebatch = np.zeros([count,self.output_size,self.output_size,self.output_channel])
            labelbatch=[]
            for i in xrange(count):
                try:
                    im = self._get_images(image_list[i], label_list[1][i])
                except:
                    im = self._get_images(image_list[i])
                imagebatch[i,:,:,:] = im
            # labelbatch=self.multi_labels[i]
            labelbatch=np.asarray(label_list)
            # labelbatch=self._get_labels(labelbatch)

        else:
            # identity = np.random.randint(0, self.class_nums)
            identity=idseed
            image_list,label_list,count=self._get_one_person_list_label(idseed)
            assert (len(image_list)==count)
            imagebatch = np.zeros([count,self.output_size,self.output_size,self.output_channel])
            # labelbatch=[]
            for i in xrange(count):
                try:
                    im = self._get_images(image_list[i], label_list[1][i])
                except:
                    im = self._get_images(image_list[i])
                imagebatch[i,:,:,:] = im
            labelbatch=np.asarray(label_list)
        if haslabel:
            return imagebatch,self._get_labels(labelbatch),identity
        else:
            return imagebatch,identity

    def flipimage(self,image,label):
        # print label
        if label<0:
            return cv2.flip(image,1)#vertical
        else:
            return image

    def fliplabel(self,label):
        label[label < 0]=abs(label[label<0])
        return label

    def _mkdual_list(self,frontalization):
        '''
        @brief: 不断地生成一个batch_size/2的随机数
        :return:
        '''
        #每次生成的dual list 只是一个batch
        # if frontalization==True:
        self._statistic(frontalization)
        while True:
            identity = np.random.randint(0, self.class_nums,self.batch_size/2)
            # print self.validation_label,self.class_nums
            # print self.multi_labels[0,10:50]
            # print identity
            select_count=np.zeros((self.batch_size),np.int32)
            for i in xrange(self.batch_size/2):
                count=self.subjects_info[identity[i]].shape[1]
                select_count[i] = random.randint(0, count-1)
                # print select_count[i]
                # print light
                path, multilabel, count = self._get_one_person_list_label(identity[i])
                if frontalization:
                    light = self.subjects_info[identity[i]][1][select_count[i]]
                    sess = self.subjects_info[identity[i]][2][select_count[i]]
                    # print 'path', path[select_count[i]]
                    # print 'light',light,'sess',sess,
                    # print self.subjects_front_light[identity[i]]
                    if self.session_nums==1:
                        select_count[i + self.batch_size / 2] = self.subjects_front_light[identity[i]][0][light]
                    else:
                        if sess==4:
                            sess=1
                        elif sess==7:
                            sess=2
                        else:
                            sess=0
                        select_count[i + self.batch_size / 2] = self.subjects_front_light[identity[i]][sess][light]
                    # print 'path',path[select_count[i + self.batch_size / 2]],select_count[i + self.batch_size / 2]
                else:
                    select_count[i+self.batch_size/2]=random.randint(0,count)

            self.q_in.put((identity,select_count))
    def _statistic(self,frontalization):
        if frontalization:
            self.subjects_front_light=np.zeros([self.class_nums,self.session_nums,self.light_nums],dtype=np.int32)
        # self.subjects_counts=np.zeros([self.class_nums])
        self.subjects_info={}
        for i in xrange(self.class_nums):
            path, multilabel, count = self._get_one_person_list_label(i)
            multilabel=np.asarray(multilabel)

            self.subjects_info[i] = multilabel[1:].copy()
            # print self.subjects_info[i].shape
            flag=True
            if frontalization:
                for l_i in xrange(count):
                    if multilabel[1][l_i]==self.frontal_label:
                        if self.session_nums==1:
                            self.subjects_front_light[i, 0, multilabel[2][l_i]] = l_i
                        else:
                            print self.session_nums,multilabel[3][l_i]
                            if multilabel[3][l_i]==4:
                                # print self.session_nums,multilabel[3][l_i]
                                self.subjects_front_light[i,1, multilabel[2][l_i]] = l_i
                            elif multilabel[3][l_i]==7:
                                # print self.session_nums, multilabel[3][l_i]
                                self.subjects_front_light[i,2,multilabel[2][l_i]]=l_i
                            else:
                                self.subjects_front_light[i,0, multilabel[2][l_i]] = l_i

                        # print 'path',path[l_i]
                        flag=False
                        # print l_i
                # if flag:
                    # print path
                    # self.subjects_counts[i]=count
        print 'mk statistic subjects frontal & count finish'


    def _mkdual_batch(self):
        while True:
            identity,select_count=self.q_in.get()
            imagebatch = np.zeros([self.batch_size,self.output_size,self.output_size,self.output_channel])
            labelbatch=np.zeros([self.labels_nums,self.batch_size],dtype=np.int)
            l=[None]*self.batch_size
            # batchpath_temp=[]
            for i in xrange(self.batch_size/2):
                batchpath,batchlabel,count=self._get_one_person_list_label(identity[i])
                assert len(batchpath)==count
                # print select_count
                select_index=select_count[i]
                select_index_2=select_count[i+self.batch_size/2]
                if select_index >= count or select_index_2 >=count:
                    select_index_2=1
                    select_index=0

                try:
                    imagebatch[i,:,:,:]=self._get_images(batchpath[select_index],batchlabel[1][select_index])[:,:,:]
                    labelbatch[:,i]=batchlabel[:,select_index]
                    l[i]=batchpath[select_index]
                    l[i+self.batch_size/2]=batchpath[select_index_2]
                    # if self.lock.acquire():
                    # batchpath_temp.append(batchpath[select_index])
                    # batchpath_temp.append(batchpath[select_index_2])
                        # self.lock.release()
                    imagebatch[i+self.batch_size/2,:,:]=self._get_images(batchpath[select_index_2],batchlabel[1][select_index_2])[:,:,:]
                    labelbatch[:,i+self.batch_size/2]=batchlabel[:,select_index_2]
                except:
                    print 'ERRO:_mkdual_bath',batchpath[select_index]
            self.q.put((imagebatch,self._get_labels(labelbatch)))

    def Dual_enqueueStart(self,frontalization=False):
        t = threading.Thread(target=self._mkdual_list, args=([frontalization]))
        t.daemon = True
        t.start()
        self.task.append(t)
        for _ in xrange(self.thread_nums):
            t_out = threading.Thread(target=self._mkdual_batch,args=())
            t_out.daemon =True
            t_out.start()
            self.task.append(t_out)

    def oneperson_allpose(self,seed):
        im_batch,label_batch,identity = self.get_same_person_batch(seed,haslabel=True)
        label=[]
        index=[]
        count=im_batch.shape[0]
        for i in xrange(count):
            if label_batch[1,i] in label:
                pass
            else:
                label.append(label_batch[1,i])
                index.append(i)
        image_batch=im_batch[index,:,:,:]
        return image_batch,self._get_labels(label_batch[:,index])







def write_batch(result_path, type, sample, data, idx, score_r_id=None, score_f_id=None, identity=None, sample_idx=0,othersample=[],ifmerge=False,merge_line=12.,reverse_other=True):
    '''
    保存png
    :param result_path:
    :param type:
    :param sample:
    :param data:
    :param e_n:
    :param idx:
    :param score_r_id:
    :param score_f_id:
    :param identity:
    :param sample_idx:
    :param othersample:
    :return:
    '''
    mainfold_h = int(np.ceil(np.sqrt(sample.shape[0])))
    #if e_n==idx:第一次执行
    savepath=[]
    savename=[]
    if type == 0:#validation
        savepath.append('./{}/validation/'.format(result_path))
        savename.append(savepath[0] + '{:08}_GT.png'.format(idx))
        savename.append(savepath[0] + '{:08}_Validation.png'.format(idx))
        savename.append(savepath[0] + '{:08}_Validation_ex.png'.format(idx))
    elif type == 1:#test identity
        savepath.append('./{}/test_identity/'.format(result_path))
        savename.append(savepath[0] + '{:08}_GT.png'.format(idx))
        savename.append(savepath[0] + '{:08}_Test.png'.format(idx))
        savename.append(savepath[0] + '{:08}_Test_ex.png'.format(idx))
    elif type ==2:#pose
        savepath.append('./{}/test_pose/'.format(result_path))
        savepath.append('./{}/test_pose/_{:08}/'.format(result_path,idx))
        savename.append(savepath[1] + 'GT.png'.format(sample_idx))
        savename.append(savepath[1] + 'Test_pose.png'.format(sample_idx))
        savename.append(savepath[1] + 'Test_pose_ex.png'.format(sample_idx))
    else:
        pass
    for path in savepath:
        if not os.path.exists(path):
            os.mkdir(path)


    #存储为一个图片
    if ifmerge:

        if len(othersample):
            mainfold_h = int(np.ceil(sample.shape[0] / float(merge_line / 3.)))
            size = [mainfold_h, int(merge_line)]
            if reverse_other:
                othersample=np.concatenate(np.split(othersample,2,axis=0)[::-1],axis=0)
            save_images_merge(data,sample,othersample,size,savename[2],score_f_id)
        else:
            mainfold_h = int(np.ceil(sample.shape[0] / float(merge_line / 2.)))
            size = [mainfold_h, int(merge_line)]
            print size
            save_images_merge(data,sample,size=size,image_path=savename[2],score=score_f_id)
    else:
        # 存储为三个不同的图片
        save_images(data, [mainfold_h, mainfold_h],savename[0], score_r_id,identity)
        save_images(sample, [mainfold_h, mainfold_h],savename[1], score_f_id,identity)
        if len(othersample):
            save_images(othersample, [mainfold_h, mainfold_h],savename[2], score_f_id,identity)

def save_images_merge(image_original,image_syn_1,image_syn_2=[],size=[0,0],image_path=None,score=[]):
    if len(image_syn_2):
        return imsave_merge(inverse_transform(image_original),inverse_transform(image_syn_1),
                            inverse_transform(image_syn_2),size,image_path,score)
    else:
        print image_original.shape,image_syn_1.shape
        return imsave_merge(inverse_transform(image_original),
                            inverse_transform(image_syn_1),
                            size=size, path=image_path, score=score)
#拼接多张图片
def imsave_merge(image_original,image_syn_1,image_syn_2=[],size=[0,0],path=None,score=None):
    h, w = image_original.shape[1], image_original.shape[2]
    print 'shape',h,w
    count=image_original.shape[0]
    if len(image_syn_2):
        gap=3
        # inter=size[1]/gap
    else:
        gap=2
    inter=size[1]/gap
    if (image_original.shape[3] in (3, 4)):
        c = image_original.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        # print img.shape
        for idx, image in enumerate(image_original):
            # print idx,size
            i = idx % inter
            j = idx // inter
            i_1 = i * gap
            img[j * h:j * h + h, i_1 * w:i_1 * w + w, :] = image
            i_2 = i * gap+1

            img[j * h:j * h + h, i_2  * w:i_2 * w + w, :] = image_syn_1[idx]
            # cv2.putText(img, str(round(score[idx], 4)), ((i_2) * w, 10+(j)*h),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if gap == 3:
                i_3 = i * gap+2
                img[j * h:j * h + h, i_3 * w:i_3 * w + w, :] = image_syn_2[idx]
                # cv2.putText(img, str(round(score[idx+count], 4)), ((i_3) * w, 10+(j*h)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(path, img[:, :, :])
    elif image_original.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1],1))
        for idx, image in enumerate(image_original):
            # print idx,size
            i = idx % 4
            j = idx // 4
            i_1 = i * gap
            img[j * h:j * h + h, i_1 * w:i_1 * w + w, :] = image
            i_2 = i * gap+1

            img[j * h:j * h + h, i_2 * w:i_2 * w + w, :] = image_syn_1[idx]
            cv2.putText(img, str(round(score[idx], 4)), ((i_2) * w, 10+(j)*h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if gap==3:
                i_3 = i * gap+2
                img[j * h:j * h + h, i_3 * w:i_3 * w + w, :] = image_syn_2[idx]
                cv2.putText(img, str(round(score[idx+count], 4)), ((i_3) * w, 10+(j*h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(path, img[:, :, :])
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def save_images(images, size, image_path, score, identity):
    return imsave(inverse_transform(images), image_path,size, score, identity)

def inverse_transform(images):
    return (images + 1.) * 127.5

#将图片拼接
def merge(images, size):

    # print 'image_shape:',images.shape,'size',size
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            # print idx,size
            i = idx % size[1]
            j = idx // size[0]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
            # cv2.imwrite('./{}/test{}.png'.format(self.result_path,idx),image)
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, path, size,score=None, identity=0):
    # image = np.squeeze(self.merge(images,size))
    # print image[0]
    # return scipy.misc.imsave(path,image)
    width = images.shape[1]
    images = merge(images, size)
    # sum_n=np.sum(score,1)

    # print 'sum_n',sum_n
    if score == None:
        return cv2.imwrite(path, images[:,:,:])
    for idx in xrange(len(score)):
        i = idx % size[1]
        j = idx // size[1]
        # cv x 是h 方向
        if isinstance(identity, int):
            score_r = score[idx][identity]
        else:
            score_r = score[idx][identity[idx]]
        cv2.putText(images, str(round(score_r, 4)), (i * width, j * width + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
    # print path,images.shape
    return cv2.imwrite(path, images[:,:,:])

def write_sample(result_path,sample, data, step):
    savepath = result_path + '/' + str(step)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    out_im = (sample + 1) * 127.5
    data = (data + 1) * 127.5
    #        out_im=tf.to_float(out_im)
    l=out_im.shape[0]
    for i in xrange(l):
        img_real=None
        img=None
        try:
            img_real = data[i].astype(np.int)
            img = out_im[i].astype(np.int)
            #  print 'int trans success!!!'
        except:
            img = tf.to_int32(out_im[i])
            print 'to_int32 success!'
        # print img
        outname_real = '/sample%i_real.jpg' % i
        outname = '/sample%i.jpg' % i
        cv2.imwrite(savepath + outname_real, img_real[:, :, :])
        cv2.imwrite(savepath + outname, img[:, :, :])  #::-1

#球面差值法
def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                      normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

def compare_pair_features(feature_test,feature_train,flag=0,metric='cosine'):
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