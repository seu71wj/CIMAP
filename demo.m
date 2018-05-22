% DEMO
% This is an examplar file on how the Cimap program can be used
% 
% [Conributor]
%	Jing Wang (jing_w@seu.edu.cn),
%	Min-Ling Zhang (zhangml@seu.edu.cn)
%	School of Computer Science and Engineering, Southeast University, Nanjing 210096, China
%

% Loading the file containing the necessary inputs for calling the Cimap function
load 'sample data'
k=5;
minsize=5;
samplemethod=1;

[newdata, newpartial_target] = Cimap( train_data, train_p_target, k, minsize, samplemethod);

train_data = zscore(newdata);
train_labels = newpartial_target;
test_data = zscore(test_data);

% Use IPAL as the training model
model = IPAL_train(train_data,train_labels,10,0.95);
[accuracy,predLabel,prob] = IPAL_predict(model,test_data,test_target,10);

MAUC =calMAUC( test_label,predLabel, prob)