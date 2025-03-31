function DS = nuswide(opts, normalizeX)
% nuswide 用于加载和准备 NUS-WIDE 数据集的 CNN 特征
% 输入：
%   opts   - (struct)  参数结构体，包含数据路径等设置。
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

% 加载 NUS-WIDE 数据集的训练集和测试集特征以及标签
load(fullfile(opts.dirs.data, 'NUS-WIDE-split.mat'), 'I_tr', 'I_te', 'L_tr', 'L_te');

% 初始化 DS 结构体
DS = [];
DS.Xtrain = I_tr;  % 训练集特征
DS.Ytrain = L_tr;  % 训练集标签
DS.Xtest  = I_te;  % 测试集特征
DS.Ytest  = L_te;  % 测试集标签
DS.thr_dist = -Inf;  % 初始化距离阈值

logInfo('[NUS-WIDE] loaded in %.2f secs', toc);  % 输出加载时间
end
