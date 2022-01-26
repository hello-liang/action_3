#!/usr/bin/env python   #python2 can run by terminal

#change two position model input and result output
#use caffe or caffe_net docker container more quickly
#change input list file such as test1
import numpy as np
import scipy.io as sio
import glob
import caffe
import os
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

from utils import load_model, cropped_imagenet_mean, transform_input, lrp_hm, normalize_color_hm, process_raw_heatmaps


# GLOBALS
SET_MODE = 'cpu' # 'cpu' to use the cpu mode
DEVICE_ID = 0 # Choose your gpu ID, if you are using gpu mode

#PRETRAINED_FILE = 'path_to_one_of_our_caffe_models.caffemodel' # IMPORTANT: You need to select a pretrained model
#Below you have an example for the Lq loss trained with dataset UCF101





#MODEL_FILE = 'path_to_the_correspoding_deploy.prototxt'  #IMPORTANT: You need to select the corresponding deploy.prototxt
#Below you have an example for the Lq loss trained with dataset UCF101
MODEL_FILE = '/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/unsupervised/deploy.prototxt'



HEIGHT = 227
WIDTH = 227


def main():

    if SET_MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE_ID)
    elif SET_MODE == 'cpu':
        caffe.set_mode_cpu()

        


    # output_dir="pretrain_result_"+str(model_use)  
    #PRETRAINED_FILE='/opt/data/action_3/models_pretrained_UCF/UCF101/lq_loss/caffe_models/Lq_loss_iter_5000.caffemodel'  #pretrained UCF model

    output_dir="/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/demonstrator/lrp_output"
    PRETRAINED_FILE='/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/models/unsupervised/my_net.caffemodel'  #change model_1/pretrain/....
    TEST_DATA_FILE ='/media/liang/ssd/lrp_toolbox_heatmap/caffe-master-lrp/demonstrator/all_data.txt'#from 1:5




    net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

    # Make sure the video list file provided does not have a blank line at the end
    with open(TEST_DATA_FILE, 'r') as f:
        f_lines = f.readlines()

    video_dict = {}
    video_order = []

    for ix, line in enumerate(f_lines):
        clc_video = line.split("\t")[0].strip()
        clc_video = clc_video.split('/')[1].strip() # if error means the last line of the file test1.txt can not is none line.
        os.mkdir(output_dir+'/%s' %(clc_video))
        frames = glob.glob('/media/liang/ssd21/action_3/extract_feature/RGB_for_extract_feature/%s/*.jpg' %(clc_video))   #or ssd21
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
    plot_mean= channel_mean   

    transformer.set_mean('data_in', channel_mean)
    transformer.set_channel_swap('data_in', (2, 1, 0))
    transformer.set_transpose('data_in', (2, 0, 1))

    idx_list = range(0, num_videos)
    for j in idx_list:   

        key = video_order[j]
        os.mknod(output_dir+'/%s' %(key)+"/score.txt")
        f=open(output_dir+'/%s' %(key)+"/score.txt","w")
        video_reshape = video_dict[key]['reshape']
        num_frames = video_dict[key]['num_frames']
        frames = video_dict[key]['frames']
        video_frames = []
        video_feat = []
        for i in range(np.round(num_frames/100)+1): # range(np.round(num_frames/20)+1):  --> Analysis each 20 frames #range(0,num_frames) analysis_all  
            idx = i + 1
            if (idx > num_frames):
                idx = num_frames
            curr_frame = frames % idx
            output_dir_image=curr_frame
            output_dir_image=output_dir_image.replace("/media/liang/ssd21/action_3/extract_feature/RGB_for_extract_feature", output_dir)
            data_in = caffe.io.load_image(curr_frame)

            if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
                data_in = caffe.io.resize_image(data_in, video_reshape)

            processed_image = transformer.preprocess('data_in',data_in)
            processed_image = np.reshape(processed_image, (1,3,227,227))
            transformed_input=processed_image
          
            net.blobs['data'].data[...] = transformed_input
          #  out = net.forward()

            out = net.forward(blobs=['fc8']) #out['prob']  
            
            f.write(str(out['fc8'][0,0])+"\t"+str(out['fc8'][0,1])+"\t"+str(out['fc8'][0,2])+'\n')   #加\n换行显示

            
            top_predictions = np.argmax(out['fc8'], axis=1) #

            ## ############# ##
            # LRP parameters: #
            ## ############# ##
            lrp_type    = 'epsilon'
            # lrp_type              | meaning of lrp_param  | uses switch_layer | description 
            # ---------------------------------------------------------------------------
            # epsilon               | epsilon               | no                | epsilon lrp
            # alphabeta             | beta                  | no                | alphabeta lrp, alpha = 1-beta
            # eps_n_flat            | epsilon               | yes               | epsilon lrp until switch_layer,   wflat lrp for all layers below
            # eps_n_wsquare         | epsilon               | yes               | epsilon lrp until switch_layer,   wsquare lrp for all layers below
            # ab_n_flat             | beta                  | yes               | alphabeta lrp until switch_layer, wflat lrp for all layers below
            # ab_n_wsquare          | beta                  | yes               | alphabeta lrp until switch_layer, wsquare lrp for all layers below
            # eps_n_ab              | (epsilon, beta)       | yes               | epsilon lrp until switch_layer, alphabeta lrp for all layers below
            # layer_dep             | (epsilon, beta)       | no                | epsilon lrp for all fully-connected layers, alphabeta lrp with alpha=1 for all convolution layerrs
            # layer_dep_n_flat      | (epsilon, beta)       | yes               | layer_dep (see above) until switch_layer, wflat lrp for all layers below
            # layer_dep_n_wsquare   | (epsilon, beta)       | yes               | layer_dep (see above) until switch-layer, wsquare lrp for all layers below
        
            # depending on lrp_type, lrp_param needs to be a scalar or a tuple (see table above). If a scalar is given to an lrp_type that expects a tuple, the default epsilon=0., alpha=0. 
            lrp_param   =   1e-10
        
            # switch_layer param only needed for the composite methods
            # the parameter depicts the first layer for which the second formula type is used.
            # interesting values for caffenet are: 0, 4, 8, 10, 12 | 15, 18, 21 (convolution layers | innerproduct layers)
            switch_layer = 13
        
            classind    =  -1              # (class index  | -1 for top_class)
        
        
            ## ################################## ##
            # Heatmap calculation and presentation #
            ## ################################## ##
        
            # LRP
            backward = lrp_hm(net, transformed_input, lrp_method=lrp_type, lrp_param=lrp_param, target_class_inds=classind, switch_layer=switch_layer)
        
            if backward is None:
                print('----------ERROR-------------')
                print('LRP result is None, check lrp_type and lrp_param for corectness')
                return
        
            sum_over_channels  = True
            normalize_heatmap  = False
        
            if lrp_type in ['deconv', 'guided_backprop']:
                sum_over_channels = False
                normalize_heatmap  = True
        
            # post-process the relevance values
            heatmaps = process_raw_heatmaps(backward, normalize=normalize_heatmap, sum_over_channels=sum_over_channels)
            num_images = 1
            for im_idx in range(num_images):
                
                if classind == -1:
                    print('top class!')
                    target_index = top_predictions[im_idx]
                else:
                    target_index = classind
                    
                    
                image_paths = curr_frame
                input_image = Image.open(image_paths) 
                # stretch input to input dimensions (only for visualization)
                input_image = input_image.resize((227, 227), Image.ANTIALIAS)

                input_image = np.array(input_image)





                stretched_input =input_image  # after process
                heatmap = heatmaps[0]
        
                # presentation
                plt.subplot(1,2,1)
                if(target_index==0):                   
                    class_name="ArmFlapping"
                elif(target_index==1):
                    class_name="HeadBanging"
                elif(target_index==2):
                    class_name="Spinning"
                else:
                    class_name="error"

                    

                    
                
                
                plt.title('Prediction: {}'.format(class_name))
                plt.imshow(stretched_input)
                plt.axis('off')
        
                # normalize heatmap for visualization
                max_abs = np.max(np.absolute(heatmap))
                norm = mpl.colors.Normalize(vmin = -max_abs, vmax = max_abs)
        
                plt.subplot(1,2,2)
        
                if lrp_type in ['epsilon', 'alphabeta', 'eps', 'ab']:
                    plt.title('{}-LRP'.format(lrp_type))
        
                if lrp_type in ['eps_n_flat', 'eps_n_square', 'std_n_ab']:
                    if lrp_type == 'eps_n_flat':
                        first_method    = 'epsilon'
                        second_method   = 'wflat'
        
                    elif lrp_type == 'eps_n_square':
                        first_method    = 'epsilon'
                        second_method   = 'wsquare'
        
                    elif lrp_type == 'std_n_ab':
                        first_method    = 'epsilon'
                        second_method   = 'alphabeta'
        
                    plt.title('LRP heatmap for class {}\nstarting with {}\n {} from layer {} on.'.format(target_index, first_method, second_method, switch_layer))
        
                if sum_over_channels:
                    # relevance values are averaged over the pixel channels, use a 1-channel colormap (seismic)
                    plt.imshow(heatmap[...,0], cmap='seismic', norm=norm, interpolation='none')
                else:
                    # 1 relevance value per color channel
                    heatmap = normalize_color_hm(heatmap)
                    plt.imshow(heatmap, interpolation = 'none')
        
                plt.axis('off')
                

                plt.savefig(output_dir_image)   # save the figure to file

                #plt.show()
        f.close()
if __name__ == '__main__':
    main()