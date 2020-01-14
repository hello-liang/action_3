result_all_predict=0
%try a little one
% 更换path进行 pretrain结果和retrain结果更换
for model_use=1:5
path_root='/media/liang/ssd2/action_3/extract_feature/';
%path = strcat(path_root,'result_',string(model_use));
path = strcat(path_root,'pretrain_result_',string(model_use));

files = dir (strcat(path,'/','*.mat'));
L = length (files);
label_all=[];
feature_all=[];



for i=1:L
    load(strcat(path,'/',files(i).name));  
    len=size(frames);
    len=len(1);
    for j= 1:len;
        
    a=strsplit( frames(j,:),"/")  ;
    a=a(end-1);
    a=string(a);
    a=strsplit(a,"_");
    a=a(2);
    
    if(a== "ArmFlapping")
        label_all=[label_all,1];
    elseif(a== "HeadBanging")
        label_all=[label_all,2];
    else
        label_all=[label_all,3];
    end
    end  
    feature_all=[feature_all;feat];
   % process the image in here
end





label_predict = kmeans(feature_all,3,'MaxIter',100000);

% H  = calculated entropy of Y, given X (in bits)
x=transpose(label_predict);
y=(label_all);
%Estimate the conditional entropy of the stationary signal x given the stationay signal y with independent pairs (x,y) of samples

%condition_entropy  https://ww2.mathworks.cn/matlabcentral/fileexchange/35625-information-theory-toolbox
random_label=randi(3,length(label_all)    ,1);
x_1=random_label;
predict_entropy = condEntropy(x,y) %1.2207
max_entropy=log2(3) %1.5850

random_entropy = condEntropy(x_1,y); %1.5848  

x_2=label_all(randperm(length(label_all)));
 
shuffle_entropy = condEntropy(x_2,y);  %1.5475

  
  

result_all_predict=result_all_predict + predict_entropy

  
end
  

result_all_predict/5  %    1.1026


%1.3691 use pretrained model 
 


