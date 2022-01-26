import os
import numpy as np
from PIL import Image
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import copy
import sys
# temporarily append the directory of the pycaffe wrapper (lrp_toolbox/caffe-master-lrp/python) to the PYTHONPATH
sys.path.append('/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/python')
import caffe
#analysis the different of two ne
inwid = 227
inhei = 227

my_net = caffe.Net('/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/unsupervised/deploy.prototxt', #deploy_Lq_loss
                        '/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/unsupervised/Lq_loss_iter_5000.caffemodel',
                           caffe.TEST)  



unsupervised_net = caffe.Net('/home/liang/Desktop/deploy_Lq_loss.prototxt', #deploy_Lq_loss
                        '/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/unsupervised/Lq_loss_iter_5000.caffemodel',
                           caffe.TEST)  

        
caffe_net = caffe.Net('/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/bvlc_reference_caffenet/deploy.prototxt',
                        '/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                        caffe.TEST)


#

'''
#change the parameter 
for param_name in unsupervised_net.params.keys():
    # 权重参数
    my_net.params[param_name][0]=unsupervised_net.params[param_name][0]

    # 偏置参数
    my_net.params[param_name][1] = unsupervised_net.params[param_name][1]


    print(param_name)
 
'''   
    

for param_name in unsupervised_net.params.keys():
    # 权重参数
    weight_my = my_net.params[param_name][0].data
    weight_un = unsupervised_net.params[param_name][0].data

    # 偏置参数
    bias_my = my_net.params[param_name][1].data
    bias_un = unsupervised_net.params[param_name][1].data
    print(param_name)

    print((weight_my==weight_un).all())
    print((bias_my==bias_un).all())


   
    
# ok all is the same now ,chanege the parameter of the last layer fc8


weight_my = my_net.params['fc8'][0].data
bias_my = my_net.params['fc8'][1].data


path="/media/liang/ssd/action_3/extract_feature/all_data_save_matrix/"

bias_1=open(path+"Bias1.txt")
bias_1_w=float(bias_1.readline().strip())
bias_1.close()
bias_my[0]=bias_1_w

bias_2=open(path+"Bias2.txt")
bias_2_w=float(bias_2.readline().strip())
bias_2.close()
bias_my[1]=bias_2_w

bias_3=open(path+"Bias3.txt")
bias_3_w=float(bias_3.readline().strip())
bias_3.close()
bias_my[2]=bias_3_w

my_net.params['fc8'][1].data[...]=bias_my


weight_first=list()
weight_second=list()
weight_third=list()


scale=open(path+"Scale1.txt")
scale_num=float(scale.readline().strip())
scale.close()
weight_0=open(path+"beta1.txt")
weight_0_w=weight_0.readlines()
i=0
for line in weight_0_w:
    num=float(line.strip())
    weight_my[0,i]=num/scale_num
    i=i+1
weight_0.close()

scale=open(path+"Scale2.txt")
scale_num=float(scale.readline().strip())
scale.close()
weight_0=open(path+"beta2.txt")
weight_0_w=weight_0.readlines()
i=0
for line in weight_0_w:
    num=float(line.strip())
    weight_my[1,i]=num/scale_num
    i=i+1
weight_0.close()

scale=open(path+"Scale3.txt")
scale_num=float(scale.readline().strip())
scale.close()
weight_0=open(path+"beta3.txt")
weight_0_w=weight_0.readlines()
i=0
for line in weight_0_w:
    num=float(line.strip())
    weight_my[2,i]=num/scale_num
    i=i+1
weight_0.close()



my_net.params['fc8'][0].data[...]=weight_my


my_net.save('/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/unsupervised/my_net.caffemodel')

weight_my[2,:]*scale_num



#this one can change mean_npy

import caffe
import numpy as np
IMAGENET_MEAN_LOCATION  = '../python/caffe/imagenet/ilsvrc_2012_mean.npy'


mean_npy = np.load(IMAGENET_MEAN_LOCATION) # Input numpy array    104    116 122
blob = caffe.io.array_to_blobproto(mean_npy)
mean_binproto = 'mean.binaryproto' # Output binaryproto file
with open(mean_binproto, 'wb') as f :
    f.write( blob.SerializeToString())


image_mean = [103.939, 116.779, 128.68]

channel_mean = np.zeros((3,in_hei,in_wid))
for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val



import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
print "Usage: python convert_protomean.py proto.mean out.npy"
sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open("../data/ilsvrc12/imagenet_mean.binaryproto", 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )  #for thos 104  116  122
arr[0,0,:,:]=103.939
arr[0,1,:,:]=116.779
arr[0,2,:,:]=128.68

blob = caffe.io.array_to_blobproto(arr)
mean_binproto = '../data/ilsvrc12/my_mean.binaryproto' # Output binaryproto file
with open(mean_binproto, 'wb') as f :
    f.write( blob.SerializeToString())





#test
weight_my = net.params['fc8'][0].data
bias_my = net.params['fc8'][1].data




