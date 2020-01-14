
#!/bin/bash
TOOLS=/opt/caffe/build/tools

for c in $(seq 3 5)
do  

   mv  model_$c/test$c.txt model_$c/train$c.txt model_$c/input_layer_lq_loss.py  /opt/data/action_3

   GLOG_logtostderr=1 $TOOLS/caffe train -solver solver_Lq_loss.prototxt 

   mv -f Lq_loss_iter_5000.caffemodel Lq_loss_iter_5000.solverstate test$c.txt train$c.txt input_layer_lq_loss.py input_layer_lq_loss.pyc model_$c


done



