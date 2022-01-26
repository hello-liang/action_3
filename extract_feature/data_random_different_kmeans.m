%rng('default')
result_all_predict_hybrid=0
result_all_predict_kmeans=0
%try a little one
kmeans_plus = [];
kmeans_sample=[];
kmeans_uniform=[];
random_entropy=[];

for model_use=1:5
    
path_root='/media/liang/ssd2/action_3/extract_feature/';
%path_root='I:\action_3\extract_feature\';

path = strcat(path_root,'result_',string(model_use));

files = dir(char(strcat(path,'/','*.mat'))); %
L = length (files);
label_all=[];
feature_all=[];

for i=1:L
    load(char(strcat(path,'/',files(i).name)));   %
    len=size(frames);
    len=len(1);
    for j= 1:len
        
    a=strsplit( frames(j,:),'/')  ;
    a=a(end-1);
    a=string(a);
    a=strsplit(a,'_');
    a=a(2);
    
    if(a=='ArmFlapping')
        label_all=[label_all,1];
    elseif(a== 'HeadBanging')
        label_all=[label_all,2];
    else
        label_all=[label_all,3];
    end
    end  
    feature_all=[feature_all;feat];
   % process the image in here
end


row_shuffle=randperm(size(feature_all,1));
feature_all=feature_all(row_shuffle,:);
label_all=label_all(:,row_shuffle);






for i=1:20
[idx,KMEANS_CENTROIDS] = kmeans(feature_all,3, 'display','iter','start','plus');
x=transpose(idx);
predict_entropy = condEntropy(x,label_all) ; %1.2207
kmeans_plus=[kmeans_plus,predict_entropy];

[idx,KMEANS_CENTROIDS] = kmeans(feature_all,3, 'display','iter','start','sample');
x=transpose(idx);
predict_entropy = condEntropy(x,label_all) ; %1.2207
kmeans_sample=[kmeans_sample,predict_entropy];

[idx,KMEANS_CENTROIDS] = kmeans(feature_all,3, 'display','iter','start','uniform');
x=transpose(idx);
predict_entropy = condEntropy(x,label_all) ; %1.2207
kmeans_uniform=[kmeans_uniform,predict_entropy];


random_label=randi(3,length(label_all)    ,1);
x_1=random_label;
random_entropy =[random_entropy, condEntropy(x_1,label_all) ];

end

end

kmeans_uniform_1=mean(kmeans_uniform)% 1.3374  use margin 1

kmeans_sample_1=mean(kmeans_sample) %1.3378 use margin 1
kmeans_plus_1 = mean(kmeans_plus)   %1.3327   use margin1

random_entropy_1=mean(random_entropy)


std(kmeans_uniform)
std(kmeans_sample)
std(kmeans_plus)
std(random_entropy)




x_2=label_all(randperm(length(label_all)));
shuffle_entropy = condEntropy(x_2,label_all);  %1.5475
