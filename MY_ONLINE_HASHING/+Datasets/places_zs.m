function DS = places_zs(opts, normalizeX)
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

% normalize features
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
end

% 生成seen class和unseen class
num_class = 205;
ratio = 0.25;
classes = randperm(num_class);
unseen_num = round(ratio * num_class);
unseen_class = classes(1:unseen_num)
seen_class = classes(unseen_num+1:end)

% 生成包含75%的seen class数据
ind_seen = logical(sum(Y==seen_class, 2));
X_seen = X(ind_seen, :);
Y_seen = Y(ind_seen);

% 生成包含25%的unseen class数据
ind_unseen = logical(sum(Y==unseen_class, 2));
X_unseen = X(ind_unseen, :);
Y_unseen = Y(ind_unseen);

clear ind_seen ind_unseen;

% T = round(ratio * length(Y_unseen) / length(unseen_class));
T = 20;

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

logInfo('[Places205_CNN_Zero_Shot] loaded in %.2f secs', toc);
end
