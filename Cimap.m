function [ newdata, newpl_target ] = Cimap( train_data,  train_p_target, k, minsize, samplemethod )

% Cimap[1] deals with the class imbalance problem for partial label learning by candidate label set disambiguation 
% and training set replenishment.
%
% Syntax
%
%      [newdata, newpl_target] = Cimap(train_data,  train_p_target, k, minsize, samplemethod)
%
% Description
%
%       Cimap takes,
%           train_data          - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target      - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           k                   - The number of nearest neighbors be considered     (defalut 5)
%           minsize             - The minimum number of each class after disambiguation phrase    (defalut 5)
%           samplemethod        - The number that represents oversampling technique: 1 for ROS, 2 for SMOTE, 3 for POS
%      
%       and returns,
%           newdata             - An ?xN array, where 1st dimension is the number of new train data after coresponding oversampling technique
%           newpl_target        - A Qx? array, where 2nd dimension is the number of new train data after coresponding oversampling technique
%  
%  [1] J. Wang, M.-L. Zhang. Towards Mitigating the Class-Imbalance Problem
%  for Partial Label Learning, In: Proceedings of the 24th ACM SIGKDD
%  Conference on Knowledge Discovery and Data Mining (KDD'18), London,
%  England, 2018, in press.
if(samplemethod>3)
    samplemethod=1;
end
label_num=size(train_p_target,1);
ins_num=size(train_data,1);
f_partial_t=full(train_p_target);
% 1. candidate label set disambiguation
pred = zeros(ins_num,1);
gimma = zeros(ins_num,label_num);
for test=1:ins_num
    dis = zeros(ins_num,2);
    for i=1:ins_num
        disVec = train_data(test,:)-train_data(i,:);
        dis(i,1) = sum(disVec.^2);
        dis(i,2) = i;
    end
    dis = sortrows(dis,1);
    sumDis = sum(dis(2:k+1,1));    
    for near=2:k+1
        gimma(test,:) = gimma(test,:)+(1-dis(near,1)/sumDis)*train_p_target(:,dis(near,2))';       
    end
    [val,idx]=max(gimma(test,:));
    pred(test) = idx;    
end
dd=zeros(1,label_num);
for i=1:size(dd,2)
    dd(1,i)=length(find(pred(:)==i));
end
[~,labelIdx]=find(dd(1,:)<minsize);
newdd=zeros(1,label_num);
while(length(labelIdx)~=0)
    for i=1:length(labelIdx)
        [B,IX]=sort(gimma(:,labelIdx(1,i)),1,'descend');
        pred(IX(1:minsize))=labelIdx(1,i);
    end
    for i=1:size(dd,2)
        newdd(1,i)=length(find(pred(:)==i));
    end
    [~,labelIdx]=find(newdd(1,:)<minsize);    
end
% 2. training set replenishment
newdata=[];
newpl_target=[];
ClassD=zeros(1,label_num);
for i=1:label_num
    id=find(pred==i);
    ClassData{i}=train_data(id,:);
    ClassPTarget{i}=f_partial_t(:,id);
    ClassD(i)=length(id);
end
newClassD=repmat(max(ClassD),size(ClassD,1),size(ClassD,2));
sample_data=train_data;
partial_label=f_partial_t;
i=1;
while(i<label_num | i==label_num )
    if(newClassD(i)>ClassD(i)&ClassD(i)~=0)
        diff=newClassD(i)-ClassD(i);
        if(samplemethod==1)
            [s,pl]=ROS(ClassData{i},ClassPTarget{i},diff);
        elseif(samplemethod==2)
            [s,pl]=SMOTE(ClassData{i},ClassPTarget{i},diff,k);
        elseif(samplemethod==3)
            [s,pl]=POS(ClassData{i},ClassPTarget{i},diff,k);
        end
        sample_data=[sample_data; s];
        partial_label=[partial_label pl];
    end
    i=i+1;
end
newdata = sample_data;
newpl_target = partial_label;
    
end

