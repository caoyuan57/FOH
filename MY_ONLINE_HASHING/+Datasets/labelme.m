function DS = load_gist(opts, normalizeX)
% load_gist 用于加载 GIST 特征，并准备数据集。此函数适用于有标签和无标签数据集。
% 输入：
%   opts   - (struct) 参数结构体，包含数据路径等设置。
%   normalizeX - (int) 选择是否归一化数据。{0, 1}。如果 normalizeX = 1，数据将进行均值中心化和单位长度归一化。
%
% 输出：
%   DS - 包含训练集、测试集的结构体，含有以下字段：
%       Xtrain - (nxd) 训练数据矩阵，每行对应一个数据实例。
%       Ytrain - (nxl) 训练数据标签矩阵，l=1 对于单类数据集。
%       Xtest  - (nxd) 测试数据矩阵。
%       Ytest  - (nxl) 测试数据标签矩阵。
%       thr_dist - 距离阈值，用于无标签数据集，决定数据实例是否为邻居。

if nargin < 2, normalizeX = 1; end  % 如果没有指定归一化参数，默认归一化
if ~normalizeX, logInfo('will NOT pre-normalize data'); end  % 如果不进行归一化，输出提示信息

tic;  % 开始计时

% 加载 GIST 特征数据（来自 LabelMe 数据集）
load(fullfile(opts.dirs.data, 'LabelMe_GIST.mat'), 'gist');  % 加载 GIST 特征矩阵

% 特征归一化
if normalizeX
    X = bsxfun(@minus, gist, mean(gist, 1));  % 对特征进行零均值处理（每列减去均值）
    X = normalize(double(X));  % 将特征矩阵缩放为单位长度
end

% 随机打乱样本顺序
ind = randperm(size(X, 1));  % 随机排列索引
no_tst = 1000;  % 设置测试集样本数为 1000

% 构建 DS 结构体，包含训练集和测试集
DS = [];
DS.Xtest  = X(ind(1:no_tst), :);  % 选择前 1000 个样本作为测试集
DS.Xtrain = X(ind(no_tst+1:end), :);  % 剩余的样本作为训练集
DS.Ytrain = [];  % 无标签数据集，标签为空
DS.Ytest  = [];  % 无标签数据集，标签为空

% 计算距离阈值（用于无标签数据集的邻居判定）
DS.thr_dist = prctile(pdist(DS.Xtrain(1:2000, :), 'Euclidean'), 5);  % 计算训练集前 2000 个样本的 5th percentile 距离

logInfo('[LabelMe_GIST] loaded in %.2f secs', toc);  % 输出加载时间
end
