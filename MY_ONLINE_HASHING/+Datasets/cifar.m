function DS = cifar(opts, normalizeX)
% 该函数用于加载并预处理 CIFAR-10 的 CNN 特征（来自 VGG16 的 fc7 层），包括打乱、归一化、划分训练/测试集等
% 输入：
%   opts - 参数结构体，需包含数据路径 opts.dirs.data
%   normalizeX - 是否归一化输入特征（默认为1，归一化）
% 输出：
%   DS - 包含训练和测试集数据与标签的结构体

if nargin < 2, normalizeX = 1; end                       % 如果没有提供 normalizeX 参数，默认归一化
if ~normalizeX, logInfo('will NOT pre-normalize data'); end  % 输出提示信息：是否进行预归一化

tic;    % 启动计时器

% 加载特征和标签，这里使用的是 VGG16 提取的 CIFAR10 的 fc7 层特征
load(fullfile(opts.dirs.data, 'CIFAR10_VGG16_fc7.mat'), ...
    'trainCNN', 'testCNN', 'trainLabels', 'testLabels');

X = [trainCNN; testCNN];              % 将训练和测试特征拼接成总数据矩阵 X（n x d）
Y = [trainLabels; testLabels] + 1;    % 标签也拼接，+1 是为了转为从 1 开始的类别标签（MATLAB 从1开始）

ind = randperm(length(Y));            % 打乱索引顺序
X = X(ind, :);                        % 按照打乱顺序重排特征
Y = Y(ind);                           % 同样重排标签
clear ind                             % 清除中间变量

T = 100;                              % 划分训练测试集时的样本数阈值（具体划分细节由 split_dataset 决定）

% 如果需要归一化特征
if normalizeX
    X = bsxfun(@minus, X, mean(X,1));         % 首先进行均值中心化（每列减去均值）
    X = normalize(double(X));                 % 然后将每一行特征归一化为单位向量（L2范数为1）
end

% 根据预设函数 split_dataset 划分训练/测试集索引
[itrain, itest] = Datasets.split_dataset(X, Y, T);

DS = [];                         % 初始化输出结构体
DS.Xtrain = X(itrain, :);        % 设置训练特征
DS.Ytrain = Y(itrain);           % 设置训练标签
DS.Xtest  = X(itest, :);         % 设置测试特征
DS.Ytest  = Y(itest);            % 设置测试标签
DS.thr_dist = -Inf;              % 设置距离阈值（一般不使用）

logInfo('[CIFAR10_CNN] loaded in %.2f secs', toc);   % 输出加载时间日志
end
