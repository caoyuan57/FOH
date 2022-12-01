function DS = cifar(opts, normalizeX)
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
clear ind
T = 100;

% normalize features
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
end

% split
[itrain, itest] = Datasets.split_dataset(X, Y, T);

DS = [];
DS.Xtrain = X(itrain, :);
DS.Ytrain = Y(itrain);
DS.Xtest  = X(itest, :);
DS.Ytest  = Y(itest);
DS.thr_dist = -Inf;

logInfo('[CIFAR10_CNN] loaded in %.2f secs', toc);
end
