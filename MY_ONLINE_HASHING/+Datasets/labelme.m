function DS = load_gist(opts, normalizeX)
% Load and prepare GIST features. The data paths must be changed. For all datasets,
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
%	thr_dist - (int)   For unlabelled datasets, corresponds to the distance 
%			   value to be used in determining whether two data instance
% 			   are neighbors. If their distance is smaller, then they are
% 			   considered neighbors.
%			   Given the standard setup, this threshold value
%			   is hard-wired to be compute from the 5th percentile 
% 			   distance value obtain through 2,000 training instance. 
% 			   See labelme case below.
%
if nargin < 2, normalizeX = 1; end
if ~normalizeX, logInfo('will NOT pre-normalize data'); end

tic;
load(fullfile(opts.dirs.data, 'LabelMe_GIST.mat'), 'gist');

% normalize features
if normalizeX
    X = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
end

ind = randperm(size(X, 1));
no_tst = 1000;

DS = [];
DS.Xtest  = X(ind(1:no_tst),:);
DS.Xtrain = X(ind(no_tst+1:end),:);
DS.Ytrain = [];
DS.Ytest  = [];
DS.thr_dist = prctile(pdist(Xtrain(1:2000,:), 'Euclidean'), 5); 

logInfo('[LabelMe_GIST] loaded in %.2f secs', toc);
end
