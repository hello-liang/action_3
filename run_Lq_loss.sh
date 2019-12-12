#!/bin/sh
TOOLS=~/install/caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver solver_Lq_loss.prototxt 
echo 'Done.'
