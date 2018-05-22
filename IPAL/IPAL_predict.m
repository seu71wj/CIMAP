function [Accuarcy,predictLabel,prob]= IPAL_predict( model,testData,testTarget,k )

% IPAL deals with paritial label learning problem via an iterative label propagation procedure, and the then classifie
% the unseen instance based on mininum error reconstruction from its labels. 
% This function is the disambiguation phase of the algorithm. 
%
%    Syntax
%
% [Accuarcy,predictLabel]= IPAL_predict( model,testData,testTarget,k )
% 
%    Description
%
%       IPAL_predict takes,
%           model                       - the model which returned in the disambiguation phase
%           test_data                 - An MxN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target                 - A QxM array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
%           k                           - The number of nearest neighbors be considered (defalut 10)
% 
%      and returns,
%            Accuracy                   - Predictive accuracy on the test set
%            predictLabel                 - A QxM array, if the ith test instance is predicted to have the jth class label, then predictLabel(j,i) is 1, otherwise predictLabel(j,i) is 0
% 
%  [1] M.-L. Zhang,F. Yu Solving the Partial Label Learning Problem: An Instance-based Approach, In: Proceedings of the 24th International Joint Conference on Artificial Intelligence, 2015.

if nargin<4
    k=10;
end
if nargin<3
    error('Not enough input parameters, please check again.');
end

testData = normr(testData);
[testNeighbor,testDist] = knnsearch(model.kdtree,testData,'k',k);

accNum=0;
y = model.disambiguatedLabel';
data = model.kdtree.X;
[label_num,test_num] = size(testTarget);
predictLabel = zeros(label_num,test_num);
prob=zeros(size(testData,1),label_num);
totalscore=zeros(size(testData,1),1);
value=zeros(size(testData,1),label_num);
for i =1:test_num
    
    
    neighborIns = data(testNeighbor(i,:),:)';
    w = lsqnonneg(neighborIns,testData(i,:)');
    neighborLabel = y(testNeighbor(i,:),:);
%     w = w(1:k);
    minVal = inf;
    minIdx = -1;
    for label=1:label_num
        
        tmp=neighborLabel(:,label);
        restore = neighborIns*(tmp.*w);
        
        %                 restore = restore(1:featureNum,:);
        residual = norm(testData(i,:)'-restore,2);
        value(i,label)=1/residual;
        totalscore(i,1)=totalscore(i,1)+value(i,label);
        if residual<minVal
            minVal = residual;
            minIdx = label;
        end
    end
    if(totalscore(i,1)>0)
        prob(i,:)=value(i,:)/totalscore(i,1);
    else
        prob(i,:)=value(i,:)/(-totalscore(i,1));
    end
    predictLabel(minIdx,i) = 1;
    if testTarget(minIdx,i)==1
        accNum = accNum+1;
    end
end
%检验value有效性
for i=1:size(testData,1)
    num=length(find(value(i,:)<=0));
    if(num==label_num)
        disp('all the value is negative.')
    end
end
Accuarcy = accNum/test_num;
% labelMat = repmat(linspace(1,label_num,label_num),size(testData,1),1);
% pred = predictLabel'.*labelMat;
% pred = sum(pred,2);
cnt=0;

end

