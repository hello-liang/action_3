
rng default % for reproducibility
Y = tsne(feature_all);




x=Y;
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));


fs = frames_all;
N = length(fs);

%% create an embedding image

S = 2000; % size of full embedding image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every single image

Ntake = N;
for i=1:Ntake
    
    if mod(i, 100)==0
        fprintf('%d/%d...\n', i, Ntake);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
    
    I = imread(fs{i});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end

imshow(G);


gscatter(Y(:,1),Y(:,2),label_all)
