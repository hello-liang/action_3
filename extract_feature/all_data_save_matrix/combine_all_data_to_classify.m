rng('default')
%path_root='/media/liang/ssd2/action_3/extract_feature/';
path_root='H:\action_3\extract_feature\';
path = strcat(path_root,'result_all_train_by_all');
files = dir(char(strcat(path,'\','*.mat'))); %
%files = dir(char(strcat(path,'/','*.mat'))); %

L = length (files);
label_all=[];
label_all_string=[];
label_all_everything=[];
label_all_num=[];
feature_all=[];
frames_all=[]
for i=1:L
    load(char(strcat(path,'\',files(i).name)));   %
    %load(char(strcat(path,'/',files(i).name)));   %

    len=size(frames);
    len=len(1);
    for j= 1:len       
    a=strsplit( frames(j,:),'/')  ;
    a=a(end-1);
    a=string(a);
    label_all_everything=[label_all_everything,a];
    a=strsplit(a,'_');
    a=a(2);
    label_all_string=[label_all_string,a];
    label_all_num=[label_all_num,i];
    if(a=='ArmFlapping')
        label_all=[label_all,1];
    elseif(a== 'HeadBanging')
        label_all=[label_all,2];
    else
        label_all=[label_all,3];
    end
    end  
    feature_all=[feature_all;feat];
    frames_all=[frames_all;cellstr(frames)];

   % process the image in here
end

% H  = calculated entropy of Y, given X (in bits)
label_all = transpose(label_all);

data=array2table(feature_all);
data.class=transpose(label_all_string);

%label
%medium


 video_acc=[]
 frame_acc=[]
 i_num=[]
 
 for choose=8:11
     i_num=[i_num,choose]
 
 if choose==1
     [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = trainClassifier_medium_knn(data);
 elseif choose==2
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = trainClassifier_coarse_knn(data);
 elseif choose==3
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = trainClassifier_fine_knn(data);
 elseif choose==4
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = trainClassifier_weight_knn(data);
 elseif choose==5
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = coarse_tree(data);
 elseif choose==6
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = Fine_tree(data);
 elseif choose==7
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = medium_tree(data);
elseif choose==8
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = Linear_discriminant(data);
elseif choose==9
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = quadratic_discriminant(data);
elseif choose==10
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = linear_SVM(data);
elseif choose==11
      [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = trainClassifier_cosine_knn(data);
 end
     
label_predict=validationPredictions;
frame_acc=[frame_acc,validationAccuracy];

score=validationScores;
% tabulate(predict_1)
correct_num=0;
for i=1:60
predict_1=label_predict(label_all_num==i);

true_label=label_all_string(label_all_num==i);
acc=length(predict_1(predict_1==true_label(1))) /length(predict_1);
all_label=unique(label_all_string);

acc_1=length(predict_1(predict_1==all_label(1))) /length(predict_1);
acc_2=length(predict_1(predict_1==all_label(2))) /length(predict_1);
acc_3=length(predict_1(predict_1==all_label(3))) /length(predict_1);

 A = [acc_1,acc_2,acc_3];
[M,I] = max(A);
if(all_label(I)==true_label(1))
correct_num=correct_num+1;
end

if(acc<0.5)
    i
tabulate(predict_1) %�Ƿ���һ����whether have two max check !
end 
end
video_level_accuracy=correct_num/60
video_acc=[video_acc,video_level_accuracy]

 end
 % [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = trainClassifier_coarse_knn_test(data)

 



