
#!/bin/bash
TOOLS=/opt/caffe/build/tools


#mv  model_all/test1.txt model_all/all_data.txt model_all/input_layer_lq_loss.py  /opt/data/action_3

GLOG_logtostderr=1 $TOOLS/caffe train -solver solver_Lq_loss.prototxt 

#mv -f Lq_loss_iter_5000.caffemodel Lq_loss_iter_5000.solverstate test1.txt all_data.txt input_layer_lq_loss.py input_layer_lq_loss.pyc model_all





