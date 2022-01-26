
load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y = species(inds);
size_x=size(X)


SVMModel = fitcsvm(X,y,'Standardize',true,'KernelFunction','linear',...
    'KernelScale','auto');
[~,score] = predict(SVMModel,X); 
score(1:3,:)

N=X;
for i = 1: size_x(1)
    for j= 1:size_x(2)
   N(i,j)=(X(i,j)-SVMModel.Mu(j))/(SVMModel.Sigma(j)) ;
    end
end

calculate_by_me=(N/SVMModel.KernelParameters.Scale)*SVMModel.Beta+SVMModel.Bias;
calculate_by_me(1:3)  %%%%???????????????????????????????


A = [1 1 0 0];
B = [1; 2; 3; 4];