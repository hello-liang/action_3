# 读取第一个CSV文件并包含表头

import numpy as np
import pandas as pd
import os
import scipy.io as sio

Folder_Path = '/media/liang/ssd2/action_3/extract_feature/result_use_train_test_model/'
# 要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path = '/media/liang/ssd2/action_3/extract_feature/'  # 拼接后要保存的文件路径
SaveFile_Name_1 = 'feature.csv'  # 合并后要保存的文件名
SaveFile_Name_2 = 'label.csv'  # 合并后要保存的文件名

# 修改当前工作目录
# 将该文件夹下的所有文件名存入一个列表
file_list = os.listdir(Folder_Path)

df = sio.loadmat(Folder_Path + file_list[0])  # 编码默认UTF-8，若乱码自行更改
label = df['frames']

feature = df['feat']

# 循环遍历列表中各个CSV文件名，并追加到合并后的文件
FileStart = 1
FileEnd = len(file_list)
for i in range(FileStart, FileEnd):
    df = sio.loadmat(Folder_Path + file_list[i])  # 编码默认UTF-8，若乱码自行更改
    label_1 = df['frames']
    feature_1 = df['feat']

    feature = np.vstack((feature, feature_1))
    label = np.hstack((label, label_1))


label_list = []
for label_item in label:
    if (label_item.find("ArmFlapping") != -1):
        label_list.append(0)
    elif (label_item.find("HeadBanging") != -1):
        label_list.append(1)
    elif (label_item.find("Spinning") != -1):
        label_list.append(2)

print(len(label))
print(len(label_list))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(feature)
# print(kmeans.labels_)

print(len(kmeans.labels_))

np.savetxt('/media/liang/ssd2/action_3/extract_feature/predict_label.csv', kmeans.labels_, delimiter=',')
label_predict = kmeans.labels_
kmeans.cluster_centers_

data = np.vstack((label_list, label_predict))
entropy = 0
for i in range(3):
    p_item = len(data[1, data[0, :] == i]) / float(len(data[0, :]))
    entropy_1 = 0
    for j in range(3):
        all = data[1, data[0, :] == i]  # select from 0 row
        all_item = all[all == j]
        fre = len(all_item) / float(len(all))
        entropy_item = -fre * np.log2(fre)
        entropy_1 = entropy_1 + entropy_item
    entropy = entropy + p_item * entropy_1

print(entropy)

np.log2(3)