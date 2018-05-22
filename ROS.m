function    [sample,plTarget]=ROS(classData,pl_target,Num)
% ROS implements the perturbation oversampling technique for partial label learning
%
% Syntax
%
%      [sample,plTarget]=ROS(classData,pl_target,Num,k)
%
% Description
%
%       ROS takes,
%           classData           - An ?xN array, '?' is the number of instances in classData, the ith instance is stored in classData(i,:)
%           pl_target           - A Qx? array, '?' is the number of instances in classData, if the jth class label is one of the partial labels for the ith training instance, then pl_target(j,i) equals +1, otherwise pl_target(j,i) equals 0
%           Num                 - The number of new samples to generate
%      
%       and returns,
%           sample              - An (?+Num)xN array, the ith instance is stored in sample(i,:)
%           plTarget            - A Qx(?+Num) array, if the jth class label is one of the partial labels for sample(i,:), then plTarget(j,i) equals +1, otherwise plTarget(j,i) equals 0
%  
%
classSize=size(classData, 1);
if(classSize==0)
    error('check classData.')
elseif(classSize==1)%duplicate
    sample=repmat(classData,Num,1);
    plTarget=repmat(pl_target,1,Num);
else
    ids=round(rand(1,Num)*(size(classData,1)-1)+1);
    sample = classData(ids,:);
    plTarget =pl_target(:,ids);     
end

end

