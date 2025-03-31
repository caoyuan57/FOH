function A = affinity(X1, X2, Y1, Y2, opts)
% 计算二进制亲和矩阵
%
% 输入：
%   X1 - (n1 x d) 第一个数据集的特征矩阵，n1 是样本数，d 是特征维度
%   X2 - (n2 x d) 第二个数据集的特征矩阵，n2 是样本数，d 是特征维度
%   Y1 - (n1 x l) 第一个数据集的标签矩阵
%   Y2 - (n2 x l) 第二个数据集的标签矩阵
%   opts - (struct) 参数结构体，包含阈值和其他设置
%
% 输出：
%   A - (n1 x n2) 亲和矩阵，表示两个数据集之间的相似性

% 如果是无监督学习，或者标签为空，则基于距离计算亲和矩阵
if opts.unsupervised || isempty(Y1) || isempty(Y2)
    assert(~isempty(X1));  % 确保 X1 不为空
    assert(~isempty(X2));  % 确保 X2 不为空
    % 计算 X1 和 X2 之间的欧氏距离，并将小于阈值（thr_dist）的视为相似
    A = pdist2(X1, X2, 'Euclidean') <= opts.thr_dist;

% 如果 Y1 和 Y2 只有一列，表示是单类标签（即分类任务）
elseif size(Y1, 2) == 1
    assert(size(Y2, 2) == 1);  % 确保 Y2 也只有一列
    % 根据标签是否相等来计算亲和矩阵，标签相同则视为相似
    A = bsxfun(@eq, Y1, Y2');  % bsxfun 用于广播，比较 Y1 和 Y2 的标签是否相等

% 如果 Y1 和 Y2 都是多类标签（即多标签任务）
else
    assert(size(Y2, 2) == size(Y1, 2));  % 确保 Y1 和 Y2 的标签列数相同
    % 计算标签矩阵之间的亲和度，如果内积大于 0，则视为相似
    A = Y1 * Y2' > 0;
end

end
