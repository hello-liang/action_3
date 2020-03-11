#!/usr/bin/env python   #python2 can run by terminal

#change two position model input and result output
#use caffe or caffe_net docker container more quickly
#change input list file such as test1
import numpy as np
import scipy.io as sio
import glob
import caffe
import os


# GLOBALS
SET_MODE = 'gpu' # 'cpu' to use the cpu mode
DEVICE_ID = 0 # Choose your gpu ID, if you are using gpu mode

#PRETRAINED_FILE = 'path_to_one_of_our_caffe_models.caffemodel' # IMPORTANT: You need to select a pretrained model
#Below you have an example for the Lq loss trained with dataset UCF101





#MODEL_FILE = 'path_to_the_correspoding_deploy.prototxt'  #IMPORTANT: You need to select the corresponding deploy.prototxt
#Below you have an example for the Lq loss trained with dataset UCF101
MODEL_FILE = '/opt/data/action_3/deploy_Lq_loss.prototxt'



HEIGHT = 227
WIDTH = 227


def main():

    if SET_MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE_ID)
    elif SET_MODE == 'cpu':
        caffe.set_mode_cpu()

        
    for model_use in range(1,6):


        output_dir="pretrain_result_"+str(model_use)  
        PRETRAINED_FILE='/opt/data/action_3/models_pretrained_UCF/UCF101/lq_loss/caffe_models/Lq_loss_iter_5000.caffemodel'  #pretrained UCF model

        #output_dir="result_"+str(model_use)  
        #PRETRAINED_FILE='/opt/data/action_3/model_'+str(model_use)+'/Lq_loss_iter_5000.caffemodel'  #change model_1/pretrain/....
        TEST_DATA_FILE ='/opt/data/action_3/model_'+str(model_use)+'/test'+str(model_use)+'.txt'  #from 1:5


        if(os.path.exists(output_dir)):
            os.removedirs(output_dir)
        os.mkdir(output_dir)

        net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

        # Make sure the video list file provided does not have a blank line at the end
        with open(TEST_DATA_FILE, 'r') as f:
            f_lines = f.readlines()

        video_dict = {}
        video_order = []

        for ix, line in enumerate(f_lines):
            clc_video = line.split("\t")[0].strip()
            clc_video = clc_video.split('/')[1].strip()
            frames = glob.glob('RGB_for_extract_feature/%s/*.jpg' %(clc_video))
            num_frames = len(frames)
            print num_frames
            video_dict[clc_video] = {}
            video_dict[clc_video]['frames'] = frames[0].split('/frame')[-2] + '/frame_%04d.jpg'
            video_dict[clc_video]['reshape'] = (240, 320)
            video_dict[clc_video]['num_frames'] = num_frames
            video_order.append(clc_video)

        video_dict = video_dict
        num_videos = len(video_dict.keys())

        # Set data transformer up
        shape = (1, 3, HEIGHT, WIDTH)

        transformer = caffe.io.Transformer({'data_in': shape})
        transformer.set_raw_scale('data_in', 255)
        image_mean = [103.939, 116.779, 128.68]

        channel_mean = np.zeros((3,227,227))
        for channel_index, mean_val in enumerate(image_mean):
            channel_mean[channel_index, ...] = mean_val

        transformer.set_mean('data_in', channel_mean)
        transformer.set_channel_swap('data_in', (2, 1, 0))
        transformer.set_transpose('data_in', (2, 0, 1))

        idx_list = range(0, num_videos)
        features = []
        eval_frames =[]
        for j in idx_list:
            key = video_order[j]
            video_reshape = video_dict[key]['reshape']
            num_frames = video_dict[key]['num_frames']
            frames = video_dict[key]['frames']
            video_frames = []
            video_feat = []

            for i in range(0,num_frames): # range(np.round(num_frames/20)+1):  --> Analysis each 20 frames
                idx = i + 1
                if (idx > num_frames):
                    idx = num_frames
                curr_frame = frames % idx

                data_in = caffe.io.load_image(curr_frame)

                if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
                    data_in = caffe.io.resize_image(data_in, video_reshape)

                processed_image = transformer.preprocess('data_in',data_in)
                processed_image = np.reshape(processed_image, (1,3,227,227))
                out = net.forward_all(blobs=['fc7'],data=processed_image)

                features.append(out['fc7'][0])
                video_feat.append(out['fc7'][0])

                video_frames.append(curr_frame)
                eval_frames.append(curr_frame)

                print "Frame {}/{}, done".format(idx, num_frames)

            print "Video {}: {}, done".format(j, key)
            video_feat = np.vstack(video_feat)
            video_frames = np.hstack(video_frames)

            res = dict()
            res['feat'] = video_feat
            res['frames'] = video_frames
            sio.savemat(output_dir+'/{}'.format(key),res)


if __name__ == '__main__':
    main()
