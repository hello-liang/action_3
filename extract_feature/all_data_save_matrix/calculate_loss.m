
result_data=data(1:100,:);
result_data=[result_data;data(64952:65052,:);data(104771:104871,:)];


label=table2array(result_data(:,1025))
used_table_1=result_data(:,1:1024);
used_table=table2array(result_data(:,1:1024));



class=1
SVMModel=trainedModel_one_vs_all_no_stand.ClassificationSVM.BinaryLearners{class}

beta=SVMModel.Beta ;
Bias=SVMModel.Bias ;
Scale=SVMModel.KernelParameters.Scale ;
dlmwrite('beta1.txt',beta)
dlmwrite('Bias1.txt',Bias)
dlmwrite('Scale1.txt',Scale)



class=2
SVMModel=trainedModel_one_vs_all_no_stand.ClassificationSVM.BinaryLearners{class}

beta=SVMModel.Beta ;
Bias=SVMModel.Bias ;
Scale=SVMModel.KernelParameters.Scale ;
dlmwrite('beta2.txt',beta)
dlmwrite('Bias2.txt',Bias)
dlmwrite('Scale2.txt',Scale)

class=3
SVMModel=trainedModel_one_vs_all_no_stand.ClassificationSVM.BinaryLearners{class}

beta=SVMModel.Beta ;
Bias=SVMModel.Bias ;
Scale=SVMModel.KernelParameters.Scale ;
dlmwrite('beta3.txt',beta)
dlmwrite('Bias3.txt',Bias)
dlmwrite('Scale3.txt',Scale)





Mu=[]
Sigma=[]
for j= 1:1024
      Mu=[Mu;SVMModel.Mu(j)];
      Sigma=[Sigma;SVMModel.Sigma(j)];
end


X=table2array(result_data(:,1:1024));

size_x=size(X);
for i = 1: size_x(1)
    for j= 1:1024
   new_num=(X(i,j)-SVMModel.Mu(j))/(SVMModel.Sigma(j)) ;
     X(i,j)=new_num;

    end
end

dlmwrite('beta.txt',beta)
dlmwrite('Bias.txt',Bias)
dlmwrite('Scale.txt',Scale)
dlmwrite('Mu.txt',Mu)

dlmwrite('Sigma.txt',Sigma)

NegLoss_old=NegLoss

unique(NegLoss_old==NegLoss)

[labels_3,NegLoss,score_3] = predict(trainedModel.ClassificationSVM,used_table);
         0   -0.6144   -0.4207
         0   -0.6131   -0.4196
         0   -0.6128   -0.4150
         0   -0.6122   -0.4057
         0   -0.6172   -0.4036
         0   -0.6155   -0.4030
   -0.0001   -0.6117   -0.4022
         0   -0.6019   -0.4102
         0   -0.5893   -0.4138
   -0.0000   -0.5797   -0.4242
   
            0   -0.6000   -0.4357
         0   -0.5986   -0.4345
         0   -0.5986   -0.4297
         0   -0.6040   -0.4120
         0   -0.6231   -0.3983
         0   -0.6078   -0.4087
   -0.0013   -0.6045   -0.4072
   
   score
       1.1344    1.0763   -0.5520
    1.1245    1.0715   -0.5541
    1.1127    1.0542   -0.5643
    1.0954    1.0123   -0.5778
    1.1072    1.0178   -0.5961
    1.0982    1.0127   -0.5949
%[labels_3,NegLoss,score_3] = predict(trainedModel.ClassificationSVM,used_table);
name=trainedModel.ClassificationSVM.ClassNames
coding_matrix=trainedModel.ClassificationSVM.CodingMatrix
score_one_s=score_3(1,:)

name =

  3�1 cell array

    {'ArmFlapping'}
    {'HeadBanging'}
    {'Spinning'   }


coding_matrix =

     1     1     0
    -1     0     1
     0    -1    -1


score_one_s =

  1�3 single row vector

    1.1344    1.0763   -0.5536
    
    

score_one_s=score_3(1,:)
NegLoss_one=NegLoss(1,:)
         0   -0.6147   -0.4204

NegLoss(1,:)
k_num=[]
for k=1:3
    j_level_num=[]
    fen_mu=[]
for j=1:3
    if coding_matrix(k,j)==0
        loss=0.5
    else
    loss= (max(0,1-coding_matrix(k,j)*score_one_s(j)))/2
    end
k_num_loss=abs(coding_matrix(k,j))*loss
j_level_num=[j_level_num,k_num_loss]
fen_mu=[fen_mu,abs(coding_matrix(k,j))]
end

k_num=[k_num,sum(j_level_num)/sum(fen_mu)]
%         0    0.9216    0.6311

end

trainedModel.ClassificationSVM.BinaryLoss % hinge
trainedModel.ClassificationSVM.BinaryLearners{1}

score_3(1:3,:)
NegLoss(1:3,:)
[~,maxScore] = max(score_3,[],2);
[~,maxloss] = max(NegLoss,[],2);


[labels,score] = predict(SVMModel,used_table); 

labels=string(labels);
unique(label==labels);
score(1:3,:)
unique(score(:,2)==calculate_by_me)

calculate_by_me=(X/SVMModel.KernelParameters.Scale)*SVMModel.Beta+SVMModel.Bias;
calculate_by_me(1:3)  %%%%???????????????????????????????