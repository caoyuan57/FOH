function [q, qlabel] = update_q(q, k, Xs_t, ls_t, qlabel)
% 更新查询池
%
% 输入：
%   q      - (d x k) 查询池矩阵，d 为特征维度，k 为池大小。
%   k      - 查询池的大小。
%   Xs_t   - (d x n_t) 当前批次的特征矩阵，d 为特征维度，n_t 为样本数量。
%   ls_t   - (n_t x l) 当前批次的标签矩阵，n_t 为样本数量，l 为标签数量。
%   qlabel - (k x l) 查询池的标签矩阵，l 为标签数量。
%
% 输出：
%   q      - 更新后的查询池矩阵。
%   qlabel - 更新后的查询池标签矩阵。

n_t = size(Xs_t, 2);  % 获取当前批次样本的数量

% 遍历当前批次的所有样本
for i = 1:n_t
    d = randi(i + k);  % 随机选择一个小于等于 i + k 的整数
    if d < k  % 如果 d 小于 k，更新查询池
        q(:, d) = Xs_t(:, i);  % 将当前样本的特征加入查询池
        qlabel(d, :) = ls_t(i, :);  % 将当前样本的标签加入查询池
    end
end
end
