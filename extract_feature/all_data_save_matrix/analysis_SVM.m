result_data=data(1:100,:);
result_data=[result_data;data(64952:65052,:);data(104771:104871,:)];


label=table2array(result_data(:,1025))
used_table_1=result_data(:,1:1024);
used_table=table2array(result_data(:,1:1024));

trainedModel.ClassificationSVM.ClassNames
class=1
SVMModel=trainedModel.ClassificationSVM.BinaryLearners{class}



beta=SVMModel.Beta ;
Bias=SVMModel.Bias ;
Scale=SVMModel.KernelParameters.Scale ;


X=table2array(result_data(:,1:1024));

size_x=size(X);
for i = 1: size_x(1)
    for j= 1:1024
   new_num=(X(i,j)-SVMModel.Mu(j))/(SVMModel.Sigma(j)) ;
     X(i,j)=new_num;

    end
end


[labels_3,NegLoss,score_3] = predict(trainedModel.ClassificationSVM,used_table); 

trainedModel.ClassificationSVM.BinaryLearners{1}

score_3(1:3,:)
NegLoss(1:3,:)
[~,maxScore] = max(score_3,[],2);
[~,maxloss] = max(NegLoss,[],2);


unique(calculate_by_me==score_3(:,1))
[labels,score] = predict(SVMModel,used_table); 