# from utils import imageReaderForDRGAN
import time
# import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
# ig = imageReaderForDRGAN(data_path='/home/sunkejia/data/faces', thread_num=8)
# ig.enqueueStart()


avg = 0
avg1 =0
# options=tf.GPUOptions(per_process_gpu_memory_fraction=0.125)

# im = ig.get_loader_tf()
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     im_=sess.run(im)
#     print im_
# for i in range(1):
#     st = time.time()
#     # x = ig.batchReader()
#     ed = time.time()
#     avg = 0.5*avg + 0.5*(ed-st)
#     # print ed-st
#     # print ig.q.qsize()
#     # print 'avg: %f' %avg
#     time.sleep(1.5)
#     st = time.time()
#     im=ig.get_loader_tf()
#     ed = time.time()
#     sess =tf.Session()
#     img=sess.run(im)
#     print img
#     avg1 = 0.5 * avg1 + 0.5 * (ed - st)
#     print ed -st
#     print 'avg1 %f'%avg

from utils import ImageReader_Customize
import numpy as np
import cv2
# txt='/home/sunkejia/sourcecode/pycharm/tupu_backend_server/add_label_caspeal_multilabels.txt'
# txt='./multipie_txt/multi_subjects_labeled_cut_train_oneshot_01_45_label_3.txt'
txt='./multipie_txt/synthesis_msr/synthesis_msr_train_validation_refine20.txt'
# txt='./multipie_txt/imlist_totalsession_list_label_4_train_part.txt'
# txt='./multipie_txt/multi_subjects_labeled_cut_train_oneshot_01_45_label_3.txt'
# validtionlabel=331   0647j6
datapath=''
# datapath='/world/data-gpu-58/wangyuequan/data_sunkejia/Multi_session/'
ig = ImageReader_Customize(batch_size=16,thread_nums=8,data_path=datapath,input_size=110,output_size=96,output_channel=3,center_crop=True,data_label_txt=txt,label_nums=4)
ig.Dual_enqueueStart(frontalization=True)
# a=cv2.imread(datapath+'/CAS-PEAL_face/001032/MY_001032_IEU+00_PM-45_EN_A0_D0_T0_BB_M0_D0_S0.tif',0)
# print a
# ig.enqueueStart()class_list
#
# assert c==a
# cv2.imwrite('save12.jpg',c)
for t in range(10):
    st = time.time()
    # x,c= ig.oneperson_allpose(25)
#     print ig.class_nums
#     x,c=ig.get_same_person_batch()
    # output_syn_half_letf, output_syn_half_right = np.split(x, 2, axis=2)
    # print output_syn_half_right.shape
    # x,l,c=ig._get_one_person_list_label(10)
    # print c
    # x,lc=ig.get_same_person_batch(25,haslabel=True)
    # print
    # break
    x,c,p=ig.read_data_batch()
    save=np.zeros([2*96,8*96,3])
    # le=len(x)
    count =0
    size=48
    # save[i * size:(i + 1) *size , j * size:(j + 1) * size, :] = (output_syn_half_letf[count] + 1) * 127.5
    for i in range(2):
        for j in range(8):
            # save[i * size*2:(i + 1) * size*2, j * size:(j + 1) * size, :] = (output_syn_half_letf[count] + 1) * 127.5
            # save[i * size * 2:(i + 1) * size * 2, j * size+4*96:(j + 1) * size+4*96, :] = (output_syn_half_right[count][:,::-1,:] + 1) * 127.5
            save[i*96:(i+1)*96,j*96:(j+1)*96,:]=(x[count]+1)*127.5
            count+=1
            if count >=x.shape[0]:
                break
    #     break
    # print x[0].shape
    print p
    print t,'--------------'
    cv2.imwrite('save_read{}.png'.format(t),save)
    # break
    # break
    # c=np.transpose(c,[1,0])
    # print x.shape,c[:,1:4].shape
    # print max(c[1]),min(c[1])
    # print max(c[2]), min(c[2])
    # print max(c[3]), min(c[3])
    # print max(c[0]), min(c[0])

    # break
    # ig.random_data()
    # if i==5:
    #     ig.random_data()
    # x1=np.tile(x[0], (x.shape[0], 1)).reshape(x.shape)

    # tmp_batch = np.tile(x[0], (13,1,1,1)). \
    #     reshape(13, x.shape[1], x.shape[2], x.shape[3])
    # tmp=tmp_batch[0]

    # os.mkdir('./test/')
    # for idx,image in enumerate(tmp_batch):
    #     cv2.imwrite('./test/test{}.png'.format(idx),np.int32((image+1)*127.5))
    #     assert tmp.all()==image.all()
    #     tmp=image
    # print 'x[0]',x[0]
    # assert x1[0].all()==x[0].all()

    # print 'c[0]',c[0]
    # print 'c[1]',c[1]
    ed = time.time()
    # avg = 0.5*avg + 0.5*(ed-st)
    print ed-st
    # print ig.q.qsize()
    # print 'avg: %f' %avg