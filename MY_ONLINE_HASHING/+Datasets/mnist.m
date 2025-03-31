function DS = mnist(opts, normalizeX)
% mnist 用于加载和准备 MNIST 数据集的 CNN 特征
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

% 加载 MNIST 数据集（训练集、测试集特征和标签）
load(fullfile(opts.dirs.data, 'mnist.mat'), 'trainMNIST', 'testMNIST', 'trainLabel', 'testLabel');

% 合并训练集和测试集特征
X = [trainMNIST; testMNIST];
% 合并训练集和测试集标签，标签 +1，确保标签从 1 开始
Y = [trainLabel; testLabel] + 1;
% 随机打乱样本顺序
ind = randperm(length(Y));
X = X(ind, :);  % 按照打乱的顺序重新排列特征
Y = Y(ind);  % 按照打乱的顺序重新排列标签
clear ind  % 清理中间变量

% 特征归一化
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));  % 对特征进行零均值处理（每列减去均值）
    X = normalize(double(X));  % 然后将数据缩放到单位长度
end

% 划分训练集和测试集
T = 100;  % 设置每个类别用于测试的数据点数量

% 使用 Datasets.split_dataset 函数划分训练集和测试集
[itrain, itest] = Datasets.split_dataset(X, Y, T);

% 构建 DS 结构体，包含训练集和测试集
DS = [];
DS.Xtrain = X(itrain, :);  % 训练集特征
DS.Ytrain = Y(itrain);  % 训练集标签
DS.Xtest  = X(itest, :);  % 测试集特征
DS.Ytest  = Y(itest);  % 测试集标签
DS.thr_dist = -Inf;  % 初始化距离阈值

logInfo('[MNIST_PIXELS] loaded in %.2f secs', toc);  % 输出加载时间
end
