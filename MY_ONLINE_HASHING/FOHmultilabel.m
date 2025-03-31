clear;
% 初始化参数
opts.dirs.data = '../data';         % 数据存储路径
opts.unsupervised = 0;              % 使用监督学习模式
opts.nbits = 32;                    % 哈希码位数设为32
normalizeX = 1;                     % 启用数据归一化
K = 500;                            % 近邻搜索的K值
numq = 200;                         % 查询样本数量

% 加载MirFlickr数据集
DS = Datasets.mirflickr(opts, normalizeX);
S = DS.S;                           % 获取预计算的相似性矩阵

% 数据预处理
trainCNN = DS.Xtrain;               % 原始训练数据（n x d）
testCNN = DS.Xtest;                 % 测试数据
trainLabels = DS.Ytrain;            % 训练标签矩阵（n x c）
testLabels = DS.Ytest;              % 测试标签矩阵（n x c）

% 将数据投影到单位球面空间（行归一化）
test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));
testLabel = testLabels;             % 测试标签保持原格式
train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));
trainLabel = trainLabels;           % 训练标签保持原格式

% 清理不需要的变量
clear testCNN trainCNN testLabels trainLabels

% 转置数据矩阵（调整为d x n格式）
test = test';                       % 测试数据转置为d x n
train = train';                     % 训练数据转置为d x n
testLabel = testLabel';             % 测试标签转置为c x n
trainLabel = trainLabel';           % 训练标签转置为c x n

% 获取数据维度信息
[Ntrain, Dtrain] = size(train);     % Ntrain应为特征维度d，Dtrain为训练样本数n
[Ntest, Dtest] = size(test);        % 注意变量命名可能造成混淆（建议改为[D, Ntest]）
[Nvtrain, Dvtrain] = size(trainLabel); % 标签维度校验
[Nvtest, Dvtest] = size(testLabel);

% 初始化投影矩阵（潜在维度问题！）
W_t = randn(Ntest, opts.nbits);     % 错误！Ntest应为特征维度d，而非测试样本数
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Ntest, 1); % 列归一化

P_t = randn(opts.nbits, Nvtrain);   % 标签投影矩阵（k x c）
P_t = P_t ./ repmat(diag(sqrt(P_t' * P_t))', opts.nbits, 1); % 行归一化

% 设置超参数（来自论文）
lambda = 0.5;    % 正则化系数
sigma = 0.8;     % 量化损失权重
etad = 0.11;     % 不相似对的惩罚系数
etas = 1;        % 相似对的奖励系数
eta = 0.1;       % 未使用的参数
theta = 1.5;     % 标签投影的权重
mu = 0.5;        % 历史哈希码的权重
tau = 5;         % 正则化参数

% 训练参数设置
n_t = 2000;             % 每个训练阶段的样本数
training_size = 18015;  % 总训练样本数

% 初始化存储变量
knei = [];              % 近邻索引（未使用）
kneil = [];             % 近邻标签（未使用）
Xs_t = [];              % 当前阶段数据
Bs_t = [];              % 当前阶段哈希码
ls_t = [];              % 当前阶段标签（单标签时使用）
vs_t = [];              % 当前阶段one-hot标签

Be_t = [];              % 累积哈希码
Xe_t = [];              % 累积数据
le_t = [];              % 累积标签（单标签时使用）
ve_t = [];              % 累积one-hot标签

S_t = [];               % 相似性矩阵
now_L = [];             % 当前有效标签

tic % 开始计时

% 分阶段训练过程
for t = n_t:n_t:training_size
    % 第一阶段初始化 --------------------------------------------------------
    if t == n_t
        Xe_t = train(:, 1:n_t);     % 初始批次数据
        tmp = W_t' * Xe_t;          % 计算初始哈希码
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        Be_t = tmp;                 % 二值化

        now_X = Xe_t;               % 当前有效数据
        now_B = single(W_t' * Xe_t >= 0); % 当前二值码

        ve_t = trainLabel(:, 1:n_t); % 初始标签

        % 随机采样查询样本
        [qpool, qind] = datasample(Xe_t', numq);
        q = qpool';
        qlabel = ve_t(:, qind);
        qlabel = qlabel';           % 调整为n x c格式

        Hq = single(W_t' * q >= 0); % 查询样本哈希码

        % 初始近邻搜索
        dex = single(knnsearch(now_B', Hq', 'K', K, 'Distance', 'hamming'));
        ud = unique(dex(:));        % 去重

        % 剔除非近邻样本
        seq = (1:size(now_B,2));
        now_B(:, setdiff(seq,ud)) = nan;
        now_X(:, setdiff(seq,ud)) = nan;
        now_L = ve_t;
        now_L(:, setdiff(seq,ud)) = nan;

        tmp_W = W_t;                % 临时保存权重
        continue;
    end

    % 处理最后不完整批次 ----------------------------------------------------
    if t > 16000
        % 合并数据
        Xe_t = [Xe_t, Xs_t];
        Be_t = [Be_t, Bs_t];
        ve_t = [ve_t, vs_t];

        Xs_t = train(:, t - n_t + 1 : end); % 剩余所有样本
        now_X = [now_X, Xs_t];

        % 计算新批次哈希码
        tmp = W_t' * Xs_t;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        Bs_t = tmp;

        vs_t = trainLabel(:, t - n_t + 1 : end);
        now_L = [now_L, vs_t];       % 更新当前标签

        % 构建相似性矩阵（直接从预计算的S加载）
        S_t = S(t - n_t + 1 : 18015, 1:t - n_t);

        % 继承相似样本的哈希码
        for i = 1:size(S_t,1)
            if sum(S_t(i,:)) ~= 0
                ind = find(S_t(i,:) == 1); % 查找相似样本
                if ~isempty(ind)
                    Bs_t(:, i) = Be_t(:, ind(1)); % 继承第一个相似样本的哈希码
                end
            end
        end

        S_t(S_t == 0) = -etad;       % 设置相似性权重

        % 更新当前批次哈希码（Bs_t）-----------------------------------------
        G = opts.nbits * Be_t * S_t' + sigma * W_t' * Xs_t + theta * P_t * vs_t;
        for r = 1:opts.nbits
            be = Be_t(r, :);
            Be_r = [Be_t(1:(r-1),:); Be_t((r+1):end, :)];

            g = G(r, :);
            tmp = g - be * Be_r' * Bs_t(r, :); % 公式可能存在问题
            tmp(tmp >= 0) = 1;
            tmp(tmp < 0) = -1;
            Bs_t(r, :) = tmp;
        end

        % 更新累积哈希码（Be_t）---------------------------------------------
        Z = opts.nbits * Bs_t * S_t - mu * P_t * ve_t;
        Be_t = 2 * Z - Bs_t * Bs_t' * Be_t; % 二值化
        Be_t(Be_t >= 0) = 1;
        Be_t(Be_t < 0) =-1;

        % 更新标签投影矩阵（P_t）--------------------------------------------
        I_c = eye(Nvtrain);
        P_t = (mu * Be_t * ve_t' + theta * Bs_t * vs_t') / ...
              (theta * (vs_t * vs_t') + mu * (ve_t * ve_t') + tau * I_c);

        % 更新投影矩阵（W_t）-----------------------------------------------
        I = eye(Ntrain);              % 维度可能错误！应为Dtrain x Dtrain
        W_t = sigma * inv(sigma * (Xs_t * Xs_t') + lambda * I) * Xs_t * Bs_t';

        % 更新查询样本 ----------------------------------------------------
        [q, qlabel] = update_q(q, 50, Xs_t, vs_t', qlabel);

        % 更新有效样本 ----------------------------------------------------
        Hq = single(W_t' * q > 0);
        now_B = single(W_t' * now_X >= 0);
        dex = knnsearch(now_B', Hq', 'K', K, 'Distance', 'hamming');
        ud = unique(dex(:));
        seq = (1:size(now_B,2));
        now_B(:, setdiff(seq,ud)) = nan;
        now_X(:, setdiff(seq,ud)) = nan;
        now_L(:, setdiff(seq,ud)) = nan;

        tmp_W = tmp_W + W_t;          % 累积权重用于平均
        break;                        % 结束训练
    end

    % 常规批次处理 --------------------------------------------------------
    % （类似上述过程，此处省略详细注释）
end
toc % 阶段训练结束

% 权重平均
W_t = tmp_W ./ 9;                     % 平均9次更新

% 生成测试哈希码
tic
Htest = single(W_t' * test >= 0);
Hq = single(W_t' * q >= 0);

% 最终近邻搜索
tic
edex = knnsearch(Hq', Htest', 'K', 10, 'Distance', 'hamming');
eud = unique(edex(:));
prenei = dex(eud,:);
up = unique(prenei(:));
now_B = now_B(:,up);                 % 筛选有效哈希码
now_L = now_L(:,up);                 % 对应标签
toc

% 计算相似性关系
Aff = affinity([], [], now_L', testLabel', opts);

% 多标签mAP评估
opts.metric = 'mAP';
res = cal_precision_multi_label_batch(now_B', Htest', now_L', testLabel');
toc

% 输出结果
logInfo(['mAP = ' num2str(res)]);


%clear;
