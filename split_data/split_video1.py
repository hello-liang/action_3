#change the video and split it to frames 
#each class have a lot of video 
#this is use python2

import cv2
import os
root="/opt/data/UCF_101/"
#root_output="/media/liang/ssd1/UCF/RGB/"
root_output="/opt/data/RGB/"
listOfAction = os.listdir(root)
listOfAction=listOfAction[0:20]
for action in listOfAction:      
    listOfFiles = os.listdir(root+action)
    for entry in listOfFiles:
        #here is avi video 
        video_name=entry.split('.');
        video_name=video_name[0];  #name of video 
        vidcap = cv2.VideoCapture(root+action+'/'+entry)
        success,image = vidcap.read()
        count = 1
        success = True
        os.makedirs(root_output+action+'/'+video_name);

        while success:
          cv2.imwrite(root_output+action+'/'+video_name+"/frame_%04d.jpg" % count, image)     # save frame as JPEG file
          success,image = vidcap.read()
         # print 'Read a new frame: ', success
          count += 1
