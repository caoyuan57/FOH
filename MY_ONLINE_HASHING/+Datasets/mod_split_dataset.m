function [itrain, Ytrain, itest, Ytest] = mod_split_dataset(X, Y, T)
% mod_split_dataset 用于将数据集按照类别划分为训练集和测试集
% 输入：
%   X: 原始特征矩阵，大小为 N x D，其中 N 是样本数量，D 是特征维度
%   Y: 原始标签向量，大小为 N x 1，每个元素是样本的类别标签
%   T: 每个类别选择的测试样本数量
%
% 输出：
%   itrain: 训练集样本索引
%   Ytrain: 训练集标签
%   itest: 测试集样本索引
%   Ytest: 测试集标签

% 获取样本数量 N 和特征维度 D
[N, D] = size(X);

% 获取所有标签的唯一值（类别）
labels = unique(Y);

% 计算总的测试集大小
ntest  = numel(labels) * T;
% 计算训练集的样本数量
ntrain = N - ntest;

% 初始化训练集和测试集标签
Ytrain = zeros(ntrain, 1);
Ytest  = zeros(ntest, 1);

% 初始化训练集和测试集的索引
itrain = [];
itest  = [];

% 为每个类别构建测试集和训练集
cnt = 0;  % 用于更新训练标签的索引
for i = 1:length(labels)
    % 找到当前类别的样本索引，并随机化顺序
    ind = find(Y == labels(i));
    n_i = numel(ind);  % 当前类别的样本数
    ind = ind(randperm(n_i));  % 随机打乱样本顺序

    % 分配测试集样本
    Ytest((i-1)*T+1:i*T) = labels(i);  % 为每个类别的测试集样本赋标签
    itest = [itest; ind(1:T)];  % 将前 T 个样本作为测试集

    % 分配训练集样本
    itrain = [itrain; ind(T+1:end)];  % 剩余的样本作为训练集
    Ytrain(cnt+1:cnt+n_i-T) = labels(i);  % 为训练集样本赋标签
    cnt = cnt + n_i - T;  % 更新训练集样本数
end

% 再次随机化训练集和测试集的顺序
ind = randperm(ntrain);  % 对训练集样本进行随机打乱
itrain = itrain(ind);    % 更新训练集样本的索引
Ytrain = Ytrain(ind);    % 更新训练集标签的顺序

ind = randperm(ntest);   % 对测试集样本进行随机打乱
itest  = itest(ind);     % 更新测试集样本的索引
Ytest = Ytest(ind);      % 更新测试集标签的顺序
end
