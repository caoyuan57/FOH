function DS = places(opts, normalizeX)
% places 用于加载和准备 CNN 特征，进行零样本学习任务
% 输入：
%   opts   - (struct) 参数结构体，包含数据路径等设置。
%   normalizeX - (int) 选择是否归一化数据。{0, 1}。如果 normalizeX = 1，数据将进行均值中心化和单位长度归一化。
%
% 输出：
%   DS - 包含训练集和测试集的结构体，含有以下字段：
%       Xtrain - (nxd) 训练数据矩阵，每行对应一个数据实例。
%       Ytrain - (nxl) 训练数据标签矩阵，l=1 对于单类数据集。
%       Xtest  - (nxd) 测试数据矩阵。
%       Ytest  - (nxl) 测试数据标签矩阵。
%       thr_dist - 距离阈值，初始设置为 -Inf。

if nargin < 2, normalizeX = 1; end  % 如果没有指定归一化参数，默认归一化
if ~normalizeX, logInfo('will NOT pre-normalize data'); end  % 如果不进行归一化，输出提示信息

tic;  % 开始计时

% 加载 Places205 数据集（PCA 降维后的 AlexNet fc7 特征和标签）
load(fullfile(opts.dirs.data, 'Places205_AlexNet_fc7_PCA128.mat'), 'pca_feats', 'labels');

X = pca_feats;  % 特征矩阵（PCA 降维后的数据）
Y = labels + 1;  % 标签矩阵，+1 是为了将标签从 0 开始转换为从 1 开始

% 特征归一化
if normalizeX
    X = bsxfun(@minus, X, mean(X, 1));  % 对特征进行零均值处理（每列减去均值）
    X = normalize(double(X));  % 然后将数据缩放到单位长度
end

% 划分数据集（训练集和测试集）
T = 20;  % 设置每个类别用于测试的数据点数量

% 使用 Datasets.split_dataset 函数划分训练集和测试集
[itrain, itest] = Datasets.split_dataset(X, Y, T);

% 构建 DS 结构体，包含训练集和测试集
DS = [];
DS.Xtrain = X(itrain, :);  % 训练集特征
DS.Ytrain = Y(itrain);  % 训练集标签
DS.Xtest  = X(itest, :);  % 测试集特征
DS.Ytest  = Y(itest);  % 测试集标签
DS.thr_dist = -Inf;  % 初始化距离阈值

logInfo('[Places205_CNN] loaded in %.2f secs', toc);  % 输出加载时间
end
