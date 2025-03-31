function [mAP] = cal_precision_multi_label_batch(B1, B2, data_label, query_label)
% 计算多标签检索的平均精度（mAP）
%
% 输入：
%   B1 - 训练集的哈希码矩阵（每行是一个哈希码）
%   B2 - 测试集的哈希码矩阵（每行是一个哈希码）
%   data_label - 训练集的标签矩阵
%   query_label - 测试集的标签矩阵
%
% 输出：
%   mAP - 多标签检索的平均精度

mAP = 0;  % 初始化 mAP 为 0

% 遍历每个测试样本
for i = 1 : size(B2, 1)
    % 计算当前查询样本与训练样本的 Hamming 距离
    F = bsxfun(@minus, B2(i,:), B1);
    F = abs(F);  % 计算差值的绝对值
    F = sum(F, 2);  % 对每个训练样本与当前查询样本的距离求和

    % 对距离进行排序，获取排序后的索引
    [~, ind] = sort(F);

    n = size(B1, 1);  % 训练集样本数
    % 将排好序的训练样本的标签与当前查询样本的标签相加
    d = bsxfun(@plus, data_label(ind(1:n),:), query_label(i,:));
    % 获取相加结果中最大值的索引
    [l] = max(d, [], 2);
    % 找到所有标签为 2 的位置，表示该样本与查询样本有匹配的标签
    l = find(l == 2);

    % 如果有匹配的标签
    if length(l) ~= 0
        truth = zeros(size(B1,1), 1);  % 创建一个与训练集大小相同的真值向量
        truth(l) = 1;  % 对于匹配的标签位置，设置为 1
        truth_s = cumsum(truth);  % 计算累积和，表示到目前为止正确的匹配数量
        % 计算该查询样本的平均精度，并加到 mAP 上
        mAP = mAP + mean(truth_s(l) ./ l);
    end
end

% 计算所有查询样本的平均精度（mAP）
mAP = mAP / size(B2, 1);
end
