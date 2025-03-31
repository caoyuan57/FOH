function res = evaluate(Htrain, Htest, opts, Aff)
% 该函数用于评估训练数据（Htrain）和测试数据（Htest）的哈希性能。
%
% 输入：
%   Htrain - (logical) 训练数据的哈希码矩阵。每列对应一个哈希码。
%   Htest  - (logical) 测试数据的哈希码矩阵。每列对应一个哈希码。
%   opts   - (struct) 参数结构体，包含评估指标（如mAP, Precision等）。
%   Aff    - (logical) 邻居指示矩阵，大小为训练集大小 x 测试集大小。
%
% 输出：
%   res    - (float) 根据 opts.metric 指定的评估指标计算的性能值。

% 获取训练集大小和测试集大小
[trainsize, testsize] = size(Aff);

if strcmp(opts.metric, 'mAP')
    % 计算平均精度（mAP）
    AP  = zeros(1, testsize);  % 初始化平均精度数组
    sim = compare_hash_tables(Htrain, Htest);  % 计算哈希表相似度

    for j = 1:testsize
        labels = 2*Aff(:, j) - 1;  % 将邻居矩阵转换为 -1 和 1
        [~, ~, info] = vl_pr(labels, double(sim(:, j)));  % 计算精度-召回曲线
        AP(j) = info.ap;  % 保存每个测试样本的平均精度（AP）
    end
    res = mean(AP(~isnan(AP)));  % 计算所有测试样本的 mAP
    logInfo(['mAP = ' num2str(res)]);

elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % 计算基于前 N 个检索结果的 mAP
    assert(isfield(opts, 'mAP') & opts.mAP > 0);  % 确保 mAP 参数有效
    assert(opts.mAP < trainsize);  % 确保 N 小于训练集大小
    N = opts.mAP;  % 获取前 N 个结果的数量
    AP = zeros(1, testsize);  % 初始化平均精度数组
    sim = compare_hash_tables(Htrain, Htest);  % 计算哈希表相似度

    for j = 1:testsize
        sim_j = double(sim(:, j));  % 获取第 j 个测试样本的相似度
        idx = [];
        for th = opts.nbits:-1:-opts.nbits
            idx = [idx; find(sim_j == th)];  % 找到与当前测试样本相似度为 th 的训练样本
            if length(idx) >= N, break; end
        end
        labels = 2*Aff(idx(1:N), j) - 1;  % 获取前 N 个样本的标签
        [~, ~, info] = vl_pr(labels, sim_j(idx(1:N)));  % 计算精度-召回曲线
        AP(j) = info.ap;  % 保存平均精度（AP）
    end
    res = mean(AP(~isnan(AP)));  % 计算所有测试样本的 mAP
    logInfo('mAP@(N=%d) = %g', N, res);

elseif ~isempty(strfind(opts.metric, 'prec_k'))
    % 计算前 K 个最近邻的精度
    K = opts.prec_k;  % 获取最近邻数量 K
    prec_k = zeros(1, testsize);  % 初始化精度数组
    sim = compare_hash_tables(Htrain, Htest);  % 计算哈希表相似度

    for i = 1:testsize
        labels = Aff(:, i);  % 获取第 i 个测试样本的标签
        sim_i = sim(:, i);   % 获取第 i 个测试样本的相似度
        [~, I] = sort(sim_i, 'descend');  % 对相似度进行降序排序
        I = I(1:K);  % 获取前 K 个索引
        prec_k(i) = mean(labels(I));  % 计算前 K 个最近邻的精度
    end
    res = mean(prec_k);  % 计算所有测试样本的精度
    logInfo('Prec@(neighs=%d) = %g', K, res);

elseif ~isempty(strfind(opts.metric, 'prec_n'))
    % 计算基于 Hamming 距离为 N 的邻域内的精度
    N = opts.prec_n;  % 获取 Hamming 距离为 N 的邻域半径
    R = opts.nbits;   % 获取哈希码长度
    prec_n = zeros(1, testsize);  % 初始化精度数组
    sim = compare_hash_tables(Htrain, Htest);  % 计算哈希表相似度

    for j = 1:testsize
        labels = Aff(:, j);  % 获取第 j 个测试样本的标签
        ind = find(R - sim(:, j) <= 2 * N);  % 找到距离小于或等于 2N 的样本
        if ~isempty(ind)
            prec_n(j) = mean(labels(ind));  % 计算在该范围内的精度
        end
    end
    res = mean(prec_n);  % 计算所有测试样本的精度
    logInfo('Prec@(radius=%d) = %g', N, res);

else
    error(['Evaluation metric ' opts.metric ' not implemented']);  % 如果评估指标未实现，抛出错误
end
end

% ----------------------------------------------------------
function sim = compare_hash_tables(Htrain, Htest)
% 比较训练集和测试集的哈希表，计算相似度
trainsize = size(Htrain, 1);  % 训练集大小
testsize  = size(Htest, 1);   % 测试集大小

if trainsize < 100e3
    % 如果训练集较小，直接计算哈希表之间的相似度
    sim = (2 * single(Htrain) - 1) * (2 * single(Htest) - 1)';  % 计算哈希表相似度
    sim = int8(sim);  % 将相似度转换为 int8 类型
else
    % 如果训练集较大，采用分块处理方式
    Ltest = 2 * single(Htest) - 1;  % 将测试哈希码转换为 -1 和 1
    sim = zeros(trainsize, testsize, 'int8');  % 初始化相似度矩阵
    chunkSize = ceil(trainsize / 10);  % 设置分块大小

    for i = 1:ceil(trainsize / chunkSize)
        I = (i - 1) * chunkSize + 1 : min(i * chunkSize, trainsize);  % 当前块的训练样本索引
        tmp = (2 * single(Htrain(:, I)) - 1) * Ltest';  % 计算当前块与测试集的相似度
        sim(I, :) = int8(tmp);  % 存储计算结果
    end
    clear Ltest tmp  % 清除中间变量
end
end
