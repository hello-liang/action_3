result_all_predict_hybrid=0
result_all_predict_kmeans=0
%try a little one
for model_use=1:5
path_root='T:\action_3\extract_feature\';
%path = strcat(path_root,'result_',string(model_use));
path = strcat(path_root,'pretrain_result_',string(model_use));

files = dir(strcat(path,'\','*.mat'));
L = length (files);
label_all=[];
feature_all=[];

for i=1:L
    load(strcat(path,'\',files(i).name));  
    len=size(frames);
    len=len(1);
    for j= 1:len
        
    a=strsplit( frames(j,:),'/')  ;
    a=a(end-1);
    a=string(a);
    a=strsplit(a,'_');
    a=a(2);
    
    if(a== 'ArmFlapping')
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






% H  = calculated entropy of Y, given X (in bits)



% INIT PARTICLE SWARM
centroids = 3;          % == clusters here (aka centroids)
dimensions = 1024;         % how many dimensions in each centroid
particles = 20;         % how many particles in the swarm, aka how many solutions
iterations = 50;        % iterations of the optimization alg.
simtime=0.01;           % simulation delay btw each iteration
dataset_subset = 2;     % for the IRIS dataset, change this value from 0 to 2
write_video = false;    % enable to grab the output picture and save a video
hybrid_pso = true;     % enable/disable hybrid_pso
manual_init = false;    % enable/disable manual initialization (only for dimensions={2,3})


meas=feature_all;

dataset_size = size (meas);

% EXECUTE K-MEANS

[idx,KMEANS_CENTROIDS] = kmeans(meas,centroids, 'dist','sqEuclidean', 'display','iter','start','uniform','onlinephase','off');
label_predict_kmeans=idx


% GLOBAL PARAMETERS (the paper reports this values 0.72;1.49;1.49)
w  = 0.72; %INERTIA
c1 = 1.49; %COGNITIVE
c2 = 1.49; %SOCIAL




% SETTING UP PSO DATA STRUCTURES
swarm_vel = rand(centroids,dimensions,particles)*0.1;
swarm_pos = rand(centroids,dimensions,particles);
swarm_best = zeros(centroids,dimensions);
c = zeros(dataset_size(1),particles);%!!!!!!save classï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?ï¼?
ranges = max(meas)-min(meas); %%scale
swarm_pos = swarm_pos .* repmat(ranges,centroids,1,particles) + repmat(min(meas),centroids,1,particles);
swarm_fitness(1:particles)=Inf;

% KMEANS_INIT
if hybrid_pso
    swarm_pos(:,:,1) = KMEANS_CENTROIDS;
end



for iteration=1:iterations
      
    %CALCULATE EUCLIDEAN DISTANCES TO ALL CENTROIDS
    distances=zeros(dataset_size(1),centroids,particles); 
    for particle=1:particles
        for centroid=1:centroids
            distance=zeros(dataset_size(1),1);
            for data_vector=1:dataset_size(1)
                %meas(data_vector,:)
                distance(data_vector,1)=norm(swarm_pos(centroid,:,particle)-meas(data_vector,:));
            end
            distances(:,centroid,particle)=distance;
        end
    end
    
    %ASSIGN MEASURES with CLUSTERS    
    for particle=1:particles
        [value, index] = min(distances(:,:,particle),[],2);  %åˆ™ min(A,[],2) æ˜¯åŒ…å?«æ¯?ä¸€è¡Œçš„æœ€å°?å€¼çš„åˆ—å?‘é‡?ã€‚ %loop to find 
        % for particle 1 :20 find the sample belong to which class so each
        % partices have one way to class ,find the most fitness one
        % !because this is 3 dimention metrix ,so the second one is class
        % have 3 class so choose it to got index to identify this sample
        % belong to which class!
        c(:,particle) = index;
    end


 
    %CALCULATE GLOBAL FITNESS and LOCAL FITNESS:=swarm_fitness
    average_fitness = zeros(particles,1);
    for particle=1:particles
        for centroid = 1 : centroids
            if any(c(:,particle) == centroid)
                local_fitness=mean(distances(c(:,particle)==centroid,centroid,particle));
                average_fitness(particle,1) = average_fitness(particle,1) + local_fitness;
            end
        end
        average_fitness(particle,1) = average_fitness(particle,1) / centroids;
        if (average_fitness(particle,1) < swarm_fitness(particle))
            swarm_fitness(particle) = average_fitness(particle,1);
            swarm_best(:,:,particle) = swarm_pos(:,:,particle);     %LOCAL BEST FITNESS
        end
    end    
    [global_fitness, index] = min(swarm_fitness);       %GLOBAL BEST FITNESS
    swarm_overall_pose = swarm_pos(:,:,index);          %GLOBAL BEST POSITION
          
    % SAMPLE r1 AND r2 FROM UNIFORM DISTRIBUTION [0..1]
    r1 = rand;
    r2 = rand;
    
    % UPDATE CLUSTER CENTROIDS
    for particle=1:particles        
        inertia = w * swarm_vel(:,:,particle);
        cognitive = c1 * r1 * (swarm_best(:,:,particle)-swarm_pos(:,:,particle));
        social = c2 * r2 * (swarm_overall_pose-swarm_pos(:,:,particle));
        vel = inertia+cognitive+social;
                
        swarm_pos(:,:,particle) = swarm_pos(:,:,particle) + vel ;   % UPDATE PARTICLE POSE
        swarm_vel(:,:,particle) = vel;                              % UPDATE PARTICLE VEL
    end
    
end










%use this more useful !!

label_predict_hybrid=c(:,index)
y=(label_all);
x=transpose(label_predict_hybrid);
x1=transpose(label_predict_kmeans);

predict_entropy_hybrid = condEntropy(x,y) %1.2207
predict_entropy_kmeans = condEntropy(x1,y) %1.2207 1.2518

max_entropy=log2(3) %1.5850
result_all_predict_kmeans=result_all_predict_kmeans + predict_entropy_kmeans
result_all_predict_hybrid=result_all_predict_hybrid + predict_entropy_hybrid

 
end
  

result_all_predict/5  %    1.1026
%the code that use out owm data
%1.3691 use pretrained model 
label_predict = kmeans(feature_all,3,'MaxIter',100000);
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

 