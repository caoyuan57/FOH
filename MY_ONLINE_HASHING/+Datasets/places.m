function DS = places(opts, normalizeX)
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
%   normalizeX - (int)     Choices are {0, 1}. If normalizeX = 1, the data is 
% 			   mean centered and unit-normalized. 
% 		
% OUTPUTS
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
load(fullfile(opts.dirs.data, 'Places205_AlexNet_fc7_PCA128.mat'), ...
    'pca_feats', 'labels');
X = pca_feats;
Y = labels + 1;
T = 20;

% normalize features
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
end

[itrain, itest] = Datasets.split_dataset(X, Y, T);

DS = [];
DS.Xtrain = X(itrain, :);
DS.Ytrain = Y(itrain);
DS.Xtest  = X(itest, :);
DS.Ytest  = Y(itest);
DS.thr_dist = -Inf;

logInfo('[Places205_CNN] loaded in %.2f secs', toc);
end
