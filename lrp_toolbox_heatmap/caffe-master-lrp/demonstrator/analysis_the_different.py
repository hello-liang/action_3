#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:31:02 2020

@author: liang
"""
#analysis the different
from scipy.io import loadmat
annots = loadmat("/media/liang/ssd21/action_3/extract_feature/result_all_train_by_all/v_ArmFlapping_01.mat")


annots.get('frames')
len(annots)