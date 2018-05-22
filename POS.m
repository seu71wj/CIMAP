function [sample,wplTarget]=POS(classData,pl_target,Num,k)
% POS implements the perturbation oversampling technique for partial label learning
%
% Syntax
%
%      [sample,plTarget]=POS(classData,pl_target,Num,k)
%
% Description
%
%       POS takes,
%           classData           - An ?xN array, '?' is the number of instances in classData, the ith instance is stored in classData(i,:)
%           pl_target           - A Qx? array, '?' is the number of instances in classData, if the jth class label is one of the partial labels for the ith training instance, then pl_target(j,i) equals +1, otherwise pl_target(j,i) equals 0
%           Num                 - The number of new samples to generate
%           k                   - The number of nearest neighbors be considered     (defalut 5)
%      
%       and returns,
%           sample              - An (?+Num)xN array, the ith instance is stored in sample(i,:)
%           wplTarget           - A Qx(?+Num) array, if the jth class label is one of the partial labels for sample(i,:), then wplTarget(j,i) equals +1, otherwise wplTarget(j,i) equals 0
%  
%
classSize=size(classData, 1);
if(classSize==0)
    error('check classData.')
elseif(classSize==1)
    sample=repmat(classData,Num,1);
    wplTarget=repmat(pl_target,1,Num);
else
    if(k>classSize-1)
        k=classSize-1;
    end
    attNum=size(classData,2);
    n=floor(Num/classSize);
    remainder=Num-classSize*n;
    id=randperm(classSize);
    No=ones(1,classSize)*n;
    No(id(1:remainder))=No(id(1:remainder))+1;   
    sample=[];
    wplTarget=[];
    for i=find(No~=0)
        d=dist(classData(i,:), classData');
        d(i)=Inf;
        if(k<log(classSize))
            min_id=[];
            for j=1:k
                [tmp,id]=min(d);
                d(id)=Inf;
                min_id=[min_id id];
            end
        else
            [tmp,id]=sort(d);
            min_id=id(1:k);
        end      
        rn=floor(rand(1,No(i))*k)+1; 
        id=min_id(rn);
        weight=rand(attNum,No(i));
        D=repmat(classData(i,:),No(i),1);
        wplT=repmat(pl_target(:,i),1,No(i));      
        aid=[1:attNum]';
        D(:,aid)=D(:,aid)+weight(aid,:)'.*(classData(id,aid)-D(:,aid));
        wplT=wplT+(pl_target(:,id)-wplT)*(weight(aid,:)'*(classData(id,aid)'-D(:,aid)'));
        for jj=1:No(i)
            for kk=1:size(pl_target,1)
                if(wplT(kk,jj)>0)
                    wplT(kk,jj)=1;
                else
                    wplT(kk,jj)=0;
                end
            end
            if(sum(wplT(:,jj))==0)
                wplT(:,jj)=pl_target(:,i);
            end
        end       
        sample=[sample; D];
        wplTarget=[wplTarget wplT ];
    end
end

