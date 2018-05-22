function model = IPAL_train( trainData,trainTarget,k,alpha )

% IPAL deals with paritial label learning problem via an iterative label propagation procedure, and the then classifie
% the unseen instance based on mininum error reconstruction from its labels. 
% This function is the disambiguation phase of the algorithm. 
%
%    Syntax
%
%      model = IPAL_train( trainData,trainTarget,k,alpha )
%
%    Description
%
%       IPAL_train takes,
%           train_data                  - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target              - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           k                           - The number of nearest neighbors be considered     (defalut 10)
%           alpha                       - A balancing coefficient parameter which 0<alpha<1 (defalut 0.95)
% 
%      and returns,
%           model is a structure continues following elements
%           model.kdtree                -A kdtree which is used to find k-nearest neighbors in the test phase
%           model.finalConfidence       -A QxM array,finalConfidence(j,i) is the final confidence of j th label of i th training instance
%           model.disambiguatedLabel    -A QxM array,disambiguatedLabel(j,i) equals +1 if j th class label if i th instance is the single label after disambiguating, otherwise disambiguatedLabel(j,i) equals 0

%  [1] M.-L. Zhang,F. Yu Solving the Partial Label Learning Problem: An Instance-based Approach, In: Proceedings of the 24th International Joint Conference on Artificial Intelligence, 2015.

if nargin<4
    alpha = 0.95;
end
if nargin<3
    k=10;
end
if nargin<2
    error('Not enough input parameters, please check again.');
end

trainData = normr(trainData);
    kdtree = KDTreeSearcher(trainData);
    [label_num,ins_num] = size(trainTarget);
            kdtree = KDTreeSearcher(trainData);
     ins_num = size(trainData,1);
     [neighbor,dist] = knnsearch(kdtree,trainData,'k',k+1);
     neighbor = neighbor(:,2:k+1);
     dist = dist(:,2:k+1);
    y = trainTarget';
    y = y./repmat(sum(y,2),1,label_num);
    %     y = y.^2;
    trans = zeros(ins_num);
    for i=1:ins_num
                neighborIns = trainData(neighbor(i,:),:)';
                w = lsqnonneg(neighborIns,trainData(i,:)');
                trans(i,neighbor(i,:)) = w';
    end
    sumW = sum(trans,2);
    sumW(sumW==0)=1;
    trans = trans./repmat(sumW,1,ins_num);
    trans = sparse(trans);
    y0 = y;
    iterVal = zeros(1,100);
    for iter=1:100
        tmp= y;
        y = alpha*trans*y+(1-alpha)*y0;                          %propagate Y<-T
        y = y.*trainTarget';                                %row-normalize Y'
        y = y./repmat(sum(y,2),1,label_num);
        diff=norm(full(tmp)-full(y),2);
        iterVal(iter) = abs(diff);
        if abs(diff)<0.0001
            break
        end
    end

    
        labelSum = sum(y0);
    predSum = sum(y);
    poster = labelSum./predSum;
    y = y.*repmat(poster,ins_num,1);
        predLabel = zeros(ins_num,label_num);
    for i=1:ins_num
        [val,idx] = max(y(i,:));
        predLabel(i,idx)=1;
    end
    model.kdtree = kdtree;
    model.finalConfidence = y';
%     model.iterVal = iterVal;
    model.disambiguatedLabel = predLabel';
    
end

