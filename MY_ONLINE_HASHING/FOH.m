clear;
% 设置选项参数
opts.dirs.data = '../data'; % 数据目录
opts.unsupervised = 0;      % 使用监督学习
opts.nbits = 16;            % 哈希码位数
normalizeX = 1;             % 归一化数据
K = 500;                    % 近邻数
numq = 500;                 % 查询样本数量

% 加载CIFAR数据集
DS = Datasets.cifar(opts, normalizeX);

% 获取训练和测试数据
trainCNN = DS.Xtrain;        % 训练数据（n x d）
testCNN = DS.Xtest;          % 测试数据
trainLabels = DS.Ytrain;     % 训练标签（n x 1）
testLabels = DS.Ytest;       % 测试标签

% 将数据归一化到单位球面空间（按行归一化）
test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  
testLabel = testLabels;      % 测试标签向量
testLabelvec = full(ind2vec(testLabels')); % 转换为one-hot编码（c x n）

train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2)); % 归一化训练数据
trainLabel = trainLabels;    % 训练标签
trainLabelvec = full(ind2vec(trainLabel')); % 转换为one-hot编码

% 清理不需要的变量
clear testCNN trainCNN testLabels trainLabels

% 获取数据维度
[Ntrain, Dtrain] = size(train);
[Ntest, Dtest] = size(test);
[Nvtrain, Dvtrain] = size(trainLabelvec);
[Nvtest, Dvtest] = size(testLabelvec);

% 转置数据矩阵以便后续处理（d x n）
test = test';
train = train';

% 初始化投影矩阵W_t和P_t
W_t = randn(Dtest, opts.nbits);              % 随机初始化W_t（d x k）
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1); % 列归一化

P_t = randn(opts.nbits, Nvtrain);            % 随机初始化P_t（k x c）
P_t = P_t ./ repmat(diag(sqrt(P_t' * P_t))', opts.nbits, 1); % 行归一化

% 设置超参数（来自论文）
lambda = 0.6;   % 正则化参数
sigma = 0.8;    % 量化损失权重
etad = 0.2;     % 不相似对的惩罚
etas = 1.6;     % 相似对的奖励
theta = 1.2;    % 标签投影的权重
mu = 0.5;       % 标签一致性的权重
tau = 0.6;      % P_t的正则化参数

% 训练参数
n_t = 2000;             % 每个阶段的样本数
training_size = 20000;  % 总训练样本数

% 初始化存储变量
Xs_t = [];      % 当前阶段的数据
Bs_t = [];      % 当前阶段的哈希码
ls_t = [];      % 当前阶段的标签
vs_t = [];      % 当前阶段的one-hot标签

Be_t = [];      % 累积的哈希码
Xe_t = [];      % 累积的数据
le_t = [];      % 累积的标签
ve_t = [];      % 累积的one-hot标签

S_t = [];       % 相似性矩阵
now_L = [];     % 当前有效的标签

tic % 开始计时

% 分阶段训练
for t = n_t:n_t:training_size
    if t == n_t % 第一阶段初始化
        Xe_t = train(:, 1:n_t); % 初始训练数据
        tmp = W_t' * Xe_t; % 计算哈希码
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        Be_t = tmp; % 二值化哈希码
        
        le_t = trainLabel(1:n_t); % 初始标签
        ve_t = trainLabelvec(:, 1:n_t); % one-hot标签
        
        % 随机选择查询样本
        [qpool, qind] = datasample(Xe_t', numq);
        q = qpool';
        qlabel = le_t(qind);
        
        % 计算查询样本的哈希码
        Hq = single(W_t' * q >= 0);
        
        % 在当前数据中寻找近邻
        now_X = Xe_t;
        now_B = single(W_t' * Xe_t >= 0);
        dex = single(knnsearch(now_B', Hq', 'K', K, 'Distance', 'hamming'));
        ud = unique(dex(:));   
        seq = (1:size(now_B,2));
        % 剔除不在近邻中的样本
        now_B(:, setdiff(seq,ud)) = nan;
        now_X(:, setdiff(seq,ud)) = nan;
        now_L = le_t; 
        now_L(setdiff(seq,ud)) = nan;
        tmp_W = W_t; % 临时保存W_t
        continue;
    end
    
    % 如果下一阶段超出数据范围，终止
    if t + n_t > Ntrain
        break
    end
    
    % 合并新旧数据
    Xe_t = [Xe_t, Xs_t];
    Be_t = [Be_t, Bs_t];
    le_t = [le_t; ls_t];
    ve_t = [ve_t, vs_t];
    
    % 获取新阶段的数据
    Xs_t = train(:, t - n_t + 1 : t);
    now_X = [now_X, Xs_t];
    
    % 计算新数据的哈希码
    tmp = W_t' * Xs_t;
    tmp(tmp >= 0) = 1;
    tmp(tmp < 0) = -1;
    Bs_t = tmp;
    
    % 新数据的标签
    ls_t = trainLabel(t - n_t + 1 : t);
    vs_t = trainLabelvec(:, t - n_t + 1 : t);
    now_L = [now_L; ls_t];
    
    % 构建相似性矩阵（单标签情况）
    S_t = single(ls_t == le_t'); % 相似对为1，否则为0
    % 处理相似对，继承哈希码
    for i = 1:n_t
        if sum(S_t(i,:)) ~= 0
            ind = find(S_t(i,:) ~=0);
            Bs_t(:, i) = Be_t(:, ind(1));
        end
    end
    % 设置相似性权重
    S_t(S_t == 0) = -etad;
    S_t(S_t == 1) = etas;
    
    % 更新Bs_t（当前阶段哈希码）
    G = opts.nbits * Be_t * S_t' + sigma * W_t' * Xs_t + theta * P_t * vs_t;
    for r = 1:opts.nbits
        be = Be_t(r, :);
        Be_r = [Be_t(1:(r-1),:); Be_t((r+1):end, :)];
        
        bs = Bs_t(r, :);
        Bs_r = [Bs_t(1:(r-1),:); Bs_t((r+1):end, :)];
        
        g = G(r, :);
        G_r = [G(1:(r-1), :); G((r+1):end, :)];
        
        % 更新当前比特位
        tmp = g - be * Be_r' * Bs_r;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        Bs_t(r, :) = tmp;
    end
    
    % 更新Be_t（累积哈希码）
    Z = opts.nbits * Bs_t * S_t - mu * P_t * ve_t;
    Be_t = 2 * Z - Bs_t * Bs_t' * Be_t;
    Be_t(Be_t >= 0) = 1;
    Be_t(Be_t < 0) = -1;
    
    % 更新P_t（标签投影矩阵）
    I_c = eye(Nvtrain);
    P_t = (mu * Be_t * ve_t' + theta * Bs_t * vs_t') / (theta * (vs_t * vs_t') + mu * (ve_t * ve_t') + tau * I_c);
    
    % 更新W_t（投影矩阵）
    I = eye(Dtrain);
    W_t = sigma * inv(sigma * (Xs_t * Xs_t') + lambda * I) * Xs_t * Bs_t';
    
    % 更新查询样本
    [q, qlabel] = update_q(q, 200, Xs_t, ls_t, qlabel);
    
    % 更新当前有效数据
    now_B = single(W_t' * now_X >= 0);
    Hq = single(W_t' * q >= 0);
    dex = knnsearch(now_B', Hq', 'K', K, 'Distance', 'hamming');
    ud = unique(dex(:));
    seq = (1:size(now_B,2));
    now_B(:, setdiff(seq,ud)) = nan;
    now_X(:, setdiff(seq,ud)) = nan;
    now_L(setdiff(seq,ud)) = nan;
    
    tmp_W = tmp_W + W_t; % 累积W_t用于最终平均
end
toc % 阶段训练结束

% 平均W_t
W_t = tmp_W ./ 10;

% 计算测试集哈希码
Htest = single(W_t' * test >= 0);
Hq = single(W_t' * q >= 0);

% 最终近邻搜索
tic
edex = knnsearch(Hq', Htest', 'K', 10, 'Distance', 'hamming');
eud = unique(edex(:));
prenei = dex(eud,:);
up = unique(prenei(:));
now_B = now_B(:,up);
now_L = now_L(up);
toc

% 计算相似性关系
Aff = affinity([], [], now_L, testLabel, opts);

% 评估结果（mAP）
opts.metric = 'mAP';
res = evaluate(now_B', Htest', opts, Aff);

toc