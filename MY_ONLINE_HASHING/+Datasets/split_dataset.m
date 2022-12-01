function [itrain, itest] = split_dataset(X, Y, T)
% X: original features
% Y: original labels
% T: # test points per class

[N, D] = size(X);
labels = unique(Y);
ntest  = numel(labels) * T;
ntrain = N - ntest;
Ytrain = zeros(ntrain, 1);  
Ytest  = zeros(ntest, 1);
itrain = [];
itest  = [];

% construct test and training set
cnt = 0;
for i = 1:length(labels)
    % find examples in this class, randomize ordering
    ind = find(Y == labels(i));
    n_i = numel(ind);
    ind = ind(randperm(n_i));

    % assign test
    Ytest((i-1)*T+1:i*T) = labels(i);
    itest = [itest; ind(1:T)];

    % assign train
    itrain = [itrain; ind(T+1:end)];
    Ytrain(cnt+1:cnt+n_i-T) = labels(i);
    cnt = cnt+n_i-T;
end

% randomize again
itrain = itrain(randperm(ntrain));
itest  = itest(randperm(ntest));
end
