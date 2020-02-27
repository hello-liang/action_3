result_all_predict_hybrid=0
result_all_predict_kmeans=0
%try a little one
 model_use=1
path_root='/media/liang/ssd2/action_3/extract_feature/';
%path_root='I:\action_3\extract_feature\';

path = strcat(path_root,'pretrain_result_',string(model_use));

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
label_predict_kmeans=idx ;


% GLOBAL PARAMETERS (the paper reports this values 0.72;1.49;1.49)
w  = 0.72; %INERTIA
c1 = 1.49; %COGNITIVE
c2 = 1.49; %SOCIAL




% SETTING UP PSO DATA STRUCTURES
swarm_vel = rand(centroids,dimensions,particles)*0.1;
swarm_pos = rand(centroids,dimensions,particles);
swarm_best = zeros(centroids,dimensions);
c = zeros(dataset_size(1),particles);%!!!!!!save class
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
        [value, index] = min(distances(:,:,particle),[],2);  %则 min(A,[],2) 是包�?��?一行的最�?值的列�?��?。 %loop to find 
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
       
        y=(label_all);
        c_number=size(tabulate(transpose(c(:,particle))));
        calcu=tabulate(transpose(c(:,particle)));
        if ((c_number(1)~=3) ||  (min(calcu(:,3))<15))
                    average_fitness(particle,1)=Inf;

        else
        average_fitness(particle,1) = condEntropy(transpose(c(:,particle)),y); %1.2207  
        end
        if (average_fitness(particle,1) < swarm_fitness(particle))  %average_fitness 1,1, the fitness       
            swarm_fitness(particle) = average_fitness(particle,1);
            swarm_best(:,:,particle) = swarm_pos(:,:,particle);     %LOCAL BEST FITNESS
        end
    end    
    [global_fitness, index] = min(swarm_fitness);       %GLOBAL BEST FITNESS
    swarm_overall_pose = swarm_pos(:,:,index);          %GLOBAL BEST POSITION
    
    
    % SOME INFO ON THE COMMAND WINDOW
 
    
    label_predict_hybrid=c(:,index);
    y=(label_all);
    x=transpose(label_predict_hybrid);
    x1=transpose(label_predict_kmeans);

    predict_entropy_hybrid = condEntropy(x,y) ; %1.2207
    predict_entropy_kmeans = condEntropy(x1,y) ;%1.2207 1.2518
    
    
   fprintf('%3d. global fitness is %5.4f.  entropy_hybrid :%5.4f.   entropy_kmeans %5.4f\n',iteration,global_fitness,predict_entropy_hybrid,predict_entropy_kmeans);   
       

    %uicontrol('Style','text','Position',[40 20 180 20],'String',sprintf('Actual fitness is: %5.4f', global_fitness),'BackgroundColor',get(gcf,'Color'));        
    pause(simtime);
    %calculate condition entropy
    
    
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


csvwrite('x.txt',x)
csvwrite('y.txt',y)


condEntropy(x,y)

tabulate(x)
tabulate(y)

max_entropy=log2(3) %1.5850

