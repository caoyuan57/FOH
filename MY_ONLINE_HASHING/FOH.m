
clear;
opts.dirs.data = '../data';
opts.unsupervised = 0;
opts.nbits = 16;
normalizeX = 1;
K = 500;
numq = 500;

DS = Datasets.cifar(opts, normalizeX);

trainCNN = DS.Xtrain;  % n x  d
testCNN = DS.Xtest;
trainLabels = DS.Ytrain;  %  n x d
testLabels = DS.Ytest;


% mapped into a sphere space
test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  

testLabel = testLabels;  % n x 1
testLabelvec = full(ind2vec(testLabels')); % c x n
train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));   

trainLabel = trainLabels; % n x 1
trainLabelvec = full(ind2vec(trainLabels')); %c x n

clear testCNN trainCNN testLabels trainLabels

[Ntrain, Dtrain] = size(train);
[Ntest, Dtest] = size(test);
[Nvtrain, Dvtrain] = size(trainLabelvec);
[Nvtest, Dvtest] = size(testLabelvec);

test = test';
train = train';
%q = q';

W_t = randn(Dtest, opts.nbits);
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1);

P_t = randn(opts.nbits, Nvtrain);   %k x c
P_t = P_t ./ repmat(diag(sqrt(P_t' * P_t))', opts.nbits, 1);

%%%%%%%%%%%% seven parameters depicted in the paper %%%%%%%%%%%%%%%%
lambda = 0.6;   
sigma = 0.8;    
etad = 0.2;    
etas = 1.6;     
theta = 1.2;
mu = 0.5;
tau = 0.6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_t = 2000;     % training size at each stage       % 2K for CIFAR-10 
training_size = 20000;   % total training instances % 20K for CIFAR-10 


Xs_t = [];
Bs_t = [];
ls_t = [];
vs_t = [];

Be_t = [];
Xe_t = [];
le_t = [];
ve_t = [];

S_t = [];

now_L = [];

tic
for t = n_t:n_t:training_size
    if t == n_t       % first stage
        Xe_t = train(:, 1 : n_t);
        tmp = W_t' * Xe_t;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        
        Be_t = tmp;
        
        le_t = trainLabel(1 : n_t);
        ve_t = trainLabelvec(: , 1:n_t);
        
        [qpool, qind] = datasample(Xe_t',numq);
        q = qpool';
        qlabel = le_t(qind);
        Hq = single(W_t' * q >= 0);
        now_X = Xe_t;
        now_B = single(W_t' * Xe_t >=0);
        dex = single(knnsearch(now_B',Hq','K',K,'Distance','hamming'));
        ud = unique(dex(:));   
        seq = (1:size(now_B,2));
        now_B(:,setdiff(seq,ud)) = nan;
        now_X(:,setdiff(seq,ud)) = nan;
        now_L = le_t; 
        now_L(setdiff(seq,ud)) = nan;
        tmp_W = W_t;
        continue;
    end
    if t + n_t > Ntrain
        break
    end
    Xe_t = [Xe_t, Xs_t];
    Be_t = [Be_t, Bs_t];
    le_t = [le_t; ls_t];
    ve_t = [ve_t, vs_t];
    Xs_t = train(:, t - n_t + 1 : t);
    now_X = [now_X,Xs_t];
    
    tmp = W_t' * Xs_t;
    tmp(tmp >= 0) = 1;
    tmp(tmp < 0) = -1;
    
    Bs_t = tmp;
    
    ls_t = trainLabel(t - n_t + 1 : t);
    vs_t = trainLabelvec(: ,t - n_t + 1 : t);
    now_L = [now_L; ls_t];
    
    
    S_t = single(ls_t == le_t');    % single lable case
    for i = 1:n_t
        if sum(S_t(i,:)) ~= 0
            ind = find(S_t(i,:) ~=0);
            Bs_t(:, i) = Be_t(:, ind(1));
        end
    end
    
    S_t(S_t == 0) = -etad;
    S_t(S_t == 1) = etas;
    
    %S_t = S_t * opts.nbits;
    
    
    tag = 1;
    
    
    
    % update Bs
    G = opts.nbits * Be_t * S_t' + sigma * W_t' * Xs_t + theta * P_t * vs_t;
    for r = 1:opts.nbits
        be = Be_t(r, :);
        Be_r = [Be_t(1:(r-1),:); Be_t((r+1):end, :)];
        
        bs = Bs_t(r, :);
        Bs_r = [Bs_t(1:(r-1),:); Bs_t((r+1):end, :)];
        
        g = G(r, :);
        G_r = [G(1:(r-1), :); G((r+1):end, :)];
        
        tmp = g - be * Be_r' * Bs_r;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        
        Bs_t(r, :) = tmp;
    end
    
    
    Z = opts.nbits * Bs_t * S_t - mu * P_t * ve_t;
    Be_t = 2 * Z - Bs_t * Bs_t' * Be_t;
    Be_t(Be_t >= 0) = 1;
    Be_t(Be_t < 0) =-1;
    
    %update P_t
    I_c = eye(Nvtrain);
    P_t = (mu * Be_t * ve_t' + theta * Bs_t * vs_t')/(theta * (vs_t * vs_t')+ mu * (ve_t * ve_t') + tau * I_c);
    
    % update W
    I = eye(Dtrain);
    W_t = sigma * inv(sigma * (Xs_t * Xs_t') + lambda * I) * Xs_t * Bs_t';
    
   [q,qlabel] = update_q(q, 200, Xs_t, ls_t, qlabel);
    now_B = single(W_t' * now_X >= 0);
    Hq = single(W_t' * q >= 0);
    dex = knnsearch(now_B',Hq','K',K,'Distance','hamming');
    ud = unique(dex(:));
    seq = (1:size(now_B,2));
    now_B(:,setdiff(seq,ud)) = nan;
    now_X(:,setdiff(seq,ud)) = nan;
    now_L(setdiff(seq,ud)) = nan;
    
    tmp_W = tmp_W + W_t;
end
toc
W_t = tmp_W ./ 10;
tic
Htest = single(W_t' * test >= 0);
Hq = single(W_t' * q >= 0);
tic
edex = knnsearch(Hq',Htest','K',10,'Distance','hamming');
eud = unique(edex(:));
prenei = dex(eud,:);
up = unique(prenei(:));
now_B = now_B(:,up);
now_L = now_L(up);
toc
Aff = affinity([], [], now_L, testLabel, opts);

opts.metric = 'mAP';
res = evaluate(now_B', Htest', opts, Aff);

toc

%clear;
