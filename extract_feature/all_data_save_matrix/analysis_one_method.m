%analysis  which frame can not work and use attention map 
rng('default')
path_root='/media/liang/ssd2/action_3/extract_feature/';
%path_root='T:\action_3\extract_feature\';
path = strcat(path_root,'result_all_train_by_all');
%files = dir(char(strcat(path,'\','*.mat'))); %
files = dir(char(strcat(path,'/','*.mat'))); %

L = length (files);
label_all=[];
label_all_string=[];
label_all_everything=[];
label_all_num=[];
feature_all=[];
frames_all=[];
label_frame_num=[];
for i=1:L
    %load(char(strcat(path,'\',files(i).name)));   %
    load(char(strcat(path,'/',files(i).name)));   %

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
    label_frame_num=[label_frame_num,(1:len)];

   % process the image in here
end

% H  = calculated entropy of Y, given X (in bits)
label_all = transpose(label_all);

data=array2table(feature_all);
data.class=transpose(label_all_string);

[trainedClassifier, validationAccuracy,validationPredictions,validationScores] = quadratic_discriminant(data);



score=validationScores;
label=[transpose(label_all_everything),transpose(label_frame_num)];
%analysis                          armflaping   HeadBanging   Spinning

score=score(label_all_string=="ArmFlapping",:);
label=label(label_all_string=="ArmFlapping",:);
score=score(score(:,3)>0.3,:);
label=label(score(:,3)>0.3,:);
score=score(score(:,3)<0.8,:)
label=label(score(:,3)<0.8,:)






