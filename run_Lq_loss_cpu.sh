#!/bin/sh
TOOLS=/opt/caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver solver_Lq_loss_cpu.prototxt 
echo 'Done.'
