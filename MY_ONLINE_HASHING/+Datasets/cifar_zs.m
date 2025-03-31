function DS = cifar_zs(opts, normalizeX)
% cifar_zs 用于加载和准备 CIFAR-10 数据集的 CNN 特征，进行零样本学习任务
% 输入：
%   opts   - (struct) 参数结构体，包含数据路径等设置。
%   normalizeX - (int) 选择是否归一化数据。{0, 1}。如果 normalizeX = 1，数据将进行均值中心化和单位长度归一化。
%
% 输出：
%   DS - 包含训练集、测试集和检索集的结构体，含有以下字段：
%       Xtrain - (nxd) 训练数据矩阵，每行对应一个数据实例。
%       Ytrain - (nxl) 训练数据标签矩阵，l=1 对于单类数据集。
%       Xtest  - (nxd) 测试数据矩阵。
%       Ytest  - (nxl) 测试数据标签矩阵。
%       Xretrieval - (nxd) 检索数据矩阵。
%       Yretrieval - (nxl) 检索数据标签矩阵。
%       thr_dist - 距离阈值，初始设置为 -Inf。

if nargin < 2, normalizeX = 1; end  % 如果没有指定归一化参数，默认归一化
if ~normalizeX, logInfo('will NOT pre-normalize data'); end  % 如果不进行归一化，输出提示信息

tic;  % 开始计时

% 加载 CIFAR-10 数据集的 CNN 特征（VGG16 提取的 fc7 特征）和标签
load(fullfile(opts.dirs.data, 'CIFAR10_VGG16_fc7.mat'), 'trainCNN', 'testCNN', 'trainLabels', 'testLabels');

% 合并训练集和测试集特征
X = [trainCNN; testCNN];
% 合并训练集和测试集标签，标签 +1，确保标签从 1 开始
Y = [trainLabels; testLabels] + 1;
% 随机打乱样本顺序
ind = randperm(length(Y));
X = X(ind, :);  % 按照打乱的顺序重新排列特征
Y = Y(ind);  % 按照打乱的顺序重新排列标签
clear ind  % 清理中间变量

% 特征归一化
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));  % 对特征进行零均值处理（每列减去均值）
    X = normalize(double(X));  % 将特征矩阵缩放为单位长度
end

% 生成 seen 类别和 unseen 类别
num_class = 10;  % CIFAR-10 数据集有 10 类
ratio = 0.25;  % 设置 25% 的类别为 unseen 类别
classes = randperm(num_class);  % 随机排列类别
unseen_num = round(ratio * num_class);  % 计算 unseen 类别的数量
unseen_class = classes(1:unseen_num);  % 获取 unseen 类别
seen_class = classes(unseen_num + 1:end);  % 获取 seen 类别

% 生成包含 75% seen 类别数据
ind_seen = logical(sum(Y == seen_class, 2));  % 找到属于 seen 类别的样本
X_seen = X(ind_seen, :);  % 获取对应的特征
Y_seen = Y(ind_seen);  % 获取对应的标签

% 生成包含 25% unseen 类别数据
ind_unseen = logical(sum(Y == unseen_class, 2));  % 找到属于 unseen 类别的样本
X_unseen = X(ind_unseen, :);  % 获取对应的特征
Y_unseen = Y(ind_unseen);  % 获取对应的标签

clear ind_seen ind_unseen;  % 清理中间变量

% T 设置为 100，表示从每个 unseen 类别中选择 100 个样本进行测试
T = 100;

% 划分 unseen 类别的训练集和测试集
[iretrieval, itest] = Datasets.split_dataset(X_unseen, Y_unseen, T);  % 使用 split_dataset 函数划分数据集

% 构建 DS 结构体，包含训练集、测试集和检索集
DS = [];
DS.Xtrain = X_seen;  % 训练集特征
DS.Ytrain = Y_seen;  % 训练集标签
DS.Xtest  = X_unseen(itest, :);  % 测试集特征
DS.Ytest  = Y_unseen(itest);  % 测试集标签
DS.Xretrieval  = X_unseen(iretrieval, :);  % 检索集特征
DS.Yretrieval  = Y_unseen(iretrieval);  % 检索集标签
DS.thr_dist = -Inf;  % 初始化距离阈值

logInfo('[CIFAR10_CNN_Zero_Shot] loaded in %.2f secs', toc);  % 输出加载时间
end
