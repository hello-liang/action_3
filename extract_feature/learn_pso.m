clear;
close all;
rng('default') % For reproducibility

% INIT PARTICLE SWARM
centroids = 3;          % == clusters here (aka centroids)
dimensions = 2;         % how many dimensions in each centroid
particles = 50;         % how many particles in the swarm, aka how many solutions
iterations = 50;        % iterations of the optimization alg.
simtime=0.01;           % simulation delay btw each iteration
dataset_subset = 2;     % for the IRIS dataset, change this value from 0 to 2
write_video = false;    % enable to grab the output picture and save a video
hybrid_pso = true;     % enable/disable hybrid_pso
manual_init = false;    % enable/disable manual initialization (only for dimensions={2,3})


% LOAD DEFAULT CLUSTER (IRIS DATASET); USE WITH CARE!
load fisheriris.mat
meas = meas(:,1+dataset_subset:dimensions+dataset_subset); %only choose 2 col%RESIZE THE DATASET WITH CURRENT DIMENSIONS; USE WITH CARE!
dataset_size = size (meas);

% EXECUTE K-MEANS
if hybrid_pso
    fprintf('Running Matlab K-Means Version\n');
    [idx,KMEANS_CENTROIDS] = kmeans(meas,centroids, 'dist','sqEuclidean', 'display','iter','start','uniform','onlinephase','off');
    fprintf('\n');
end

% GLOBAL PARAMETERS (the paper reports this values 0.72;1.49;1.49)
w  = 0.72; %INERTIA
c1 = 1.49; %COGNITIVE
c2 = 1.49; %SOCIAL




% SETTING UP PSO DATA STRUCTURES
swarm_vel = rand(centroids,dimensions,particles)*0.1;  % this is volocity
swarm_pos = rand(centroids,dimensions,particles);  %this  is last time particles
swarm_best = zeros(centroids,dimensions);   %% the best centroids at this iteration
c = zeros(dataset_size(1),particles);%!!!!!!save class！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
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
            for data_vector=1:dataset_size(1) %the first particle,first centroid ,the distance of each sample and centroid
                %meas(data_vector,:)
                distance(data_vector,1)=norm(swarm_pos(centroid,:,particle)-meas(data_vector,:)); %Euclidean length
            end
            distances(:,centroid,particle)=distance;
        end
    end
    %so this calculate the distance of each sample to each particle's
    %centroid ,.  for examlple ,4 particles have 3 centroids ,calculate the
    %distance of each sample  with all of them
    
    
    %ASSIGN MEASURES with CLUSTERS    
    for particle=1:particles
        [value, index] = min(distances(:,:,particle),[],2);  %则 min(A,[],2) 是包含每一行的最小值的列向量。 %loop to find 
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
            if any(c(:,particle) == centroid) %have this centroid 
                local_fitness=mean(distances(c(:,particle)==centroid,centroid,particle));  %for the fitness of
                %this particle's this centroid
                %
                average_fitness(particle,1) = average_fitness(particle,1) + local_fitness;
                
            end
        end
        average_fitness(particle,1) = average_fitness(particle,1) / centroids;
        %the fitness of this particle is average of local fitness for each 
        
        if (average_fitness(particle,1) < swarm_fitness(particle))  %average_fitness 1,1, the fitness
            %of this particle calculated is low than swam fitness  the
            %begin is very big inf
            swarm_fitness(particle) = average_fitness(particle,1);
            swarm_best(:,:,particle) = swarm_pos(:,:,particle);     %LOCAL BEST FITNESS  
            %because this iteration this particle have a better fitness so
            %the local best of this particle is the centroid.
        end
    end    
    [global_fitness, index] = min(swarm_fitness);       %GLOBAL BEST FITNESS
    swarm_overall_pose = swarm_pos(:,:,index);          %GLOBAL BEST POSITION
    %the centroid of best fitness for those particle is the global fitness!
          
    % SAMPLE r1 AND r2 FROM UNIFORM DISTRIBUTION [0..1]
    r1 = rand;
    r2 = rand;
    
    % UPDATE CLUSTER CENTROIDS
    for particle=1:particles        
        inertia = w * swarm_vel(:,:,particle);%this to is random matrix 3*2 same as centroids
        cognitive = c1 * r1 * (swarm_best(:,:,particle)-swarm_pos(:,:,particle)); %is random ini, but the first is kmeans
        social = c2 * r2 * (swarm_overall_pose-swarm_pos(:,:,particle));
        vel = inertia+cognitive+social;
        % use vel to updata the data swarm pos ,this is centroids
        swarm_pos(:,:,particle) = swarm_pos(:,:,particle) + vel ;   % UPDATE PARTICLE POSE
        swarm_vel(:,:,particle) = vel;                              % UPDATE PARTICLE VEL
    end
    
end
B = categorical(species);
label_all= double(B);


[idx,KMEANS_CENTROIDS] = kmeans(meas,centroids, 'dist','sqEuclidean', 'display','iter','start','uniform','onlinephase','off');
label_predict_kmeans=idx
label_predict_hybrid=c(:,index)
y=(label_all);
x=transpose(label_predict_hybrid);
x1=transpose(label_predict_kmeans);

predict_entropy_hybrid = condEntropy(x,y) %    0.2035

predict_entropy_kmeans = condEntropy(x1,y) %1.2207

max_entropy=log2(3) %1.5850


tabulate(x)
tabulate(y)

[Acc,rand_index,match]=AccMeasure(y,x)


csvwrite('x.txt',x)
csvwrite('y.txt',y)

