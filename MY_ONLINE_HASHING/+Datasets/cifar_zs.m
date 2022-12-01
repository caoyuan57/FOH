function DS = cifar_zs(opts, normalizeX)
% Load and prepare CNN features. The data paths must be changed. For all datasets,
% X represents the data matrix. Rows correspond to data instances and columns
% correspond to variables/features.
% Y represents the label matrix where each row corresponds to a label vector of 
% an item, i.e., for multiclass datasets this vector has a single dimension and 
% for multilabel datasets the number of columns of Y equal the number of labels
% in the dataset. Y can be empty for unsupervised datasets.
% 
%
% INPUTS
%	opts   - (struct)  Parameter structure.
% 		
% OUTPUTS: struct DS
% 	Xtrain - (nxd) 	   Training data matrix, each row corresponds to a data
%			   instance.
%	Ytrain - (nxl)     Training data label matrix. l=1 for multiclass datasets.
%			   For unsupervised dataset Ytrain=[], see LabelMe in 
%			   load_gist.m
%	Xtest  - (nxd)     Test data matrix, each row corresponds to a data instance.
%	Ytest  - (nxl)	   Test data label matrix, l=1 for multiclass datasets. 
%			   For unsupervised dataset Ytrain=[], see LabelMe in 
%			   load_gist.m
% 
if nargin < 2, normalizeX = 1; end
if ~normalizeX, logInfo('will NOT pre-normalize data'); end
    
tic;
load(fullfile(opts.dirs.data, 'CIFAR10_VGG16_fc7.mat'), ...
    'trainCNN', 'testCNN', 'trainLabels', 'testLabels');
X = [trainCNN; testCNN];
Y = [trainLabels; testLabels] + 1;
ind = randperm(length(Y));
X = X(ind, :);
Y = Y(ind);

% normalize features
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
end

% ����seen class��unseen class
num_class = 10;
ratio = 0.25;
classes = randperm(num_class);
unseen_num = round(ratio * num_class);
unseen_class = classes(1:unseen_num)
seen_class = classes(unseen_num+1:end)

% ���ɰ���75%��seen class����
ind_seen = logical(sum(Y==seen_class, 2));
X_seen = X(ind_seen, :);
Y_seen = Y(ind_seen);

% ���ɰ���25%��unseen class����
ind_unseen = logical(sum(Y==unseen_class, 2));
X_unseen = X(ind_unseen, :);
Y_unseen = Y(ind_unseen);

clear ind train_ind test_ind;

% T = round(ratio * length(Y_unseen) / length(unseen_class));
T = 100;

% split
[iretrieval, itest] = Datasets.split_dataset(X_unseen, Y_unseen, T);

DS = [];
DS.Xtrain = X_seen;
DS.Ytrain = Y_seen;
DS.Xtest  = X_unseen(itest, :);
DS.Ytest  = Y_unseen(itest);
DS.Xretrieval  = X_unseen(iretrieval, :);
DS.Yretrieval  = Y_unseen(iretrieval);
DS.thr_dist = -Inf;

logInfo('[CIFAR10_CNN_Zero_Shot] loaded in %.2f secs', toc);
end
