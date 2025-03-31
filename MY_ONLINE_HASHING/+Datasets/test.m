load ../data/CIFAR10_VGG16_fc7.mat  % 加载 CIFAR10 数据集（VGG16 提取的 fc7 特征）

trainCNN = trainCNN;                % 训练数据的 CNN 特征
trainLabels = trainLabels;          % 训练数据的标签
testCNN = testCNN;                  % 测试数据的 CNN 特征
testLabels = testLabels;            % 测试数据的标签

% 加载 Places205 数据集（注释掉的代码）
% load ../data/Places205_AlexNet_fc7_PCA128.mat
% [trainCNN, trainLabels, testCNN, testLabels] = mod_split_dataset(pca_feats, labels, 20);

% 获取训练集和测试集的样本数及特征维度
[Ntrain, Dtrain] = size(trainCNN);  % 训练集样本数和特征维度
[Ntest, Dtest] = size(testCNN);     % 测试集样本数和特征维度
Nclasses = length(unique(trainLabels));  % 获取类别数（标签种类数）

w = zeros(1, Dtrain);               % 初始化权重向量 w
b = zeros(Nclasses, 1);             % 初始化偏置 b
b(end) = inf;                       % 最后一个类别的偏置设置为 inf

% 初始化标签矩阵
label_Tr = zeros(Ntrain, 10);       % 训练集标签矩阵（假设最多 10 类）
label_Te = zeros(Ntest, 10);        % 测试集标签矩阵（假设最多 10 类）

% 根据训练集标签填充 label_Tr
for i = 1:Ntrain
    label_Tr(i, trainLabels(i)+1) = 1;  % 标签从 0 开始，需要 +1
end

% 根据测试集标签填充 label_Te
for i = 1:Ntest
    label_Te(i, testLabels(i)+1) = 1;  % 标签从 0 开始，需要 +1
end

% PRank 算法（训练过程）
tic;  % 计时开始
for t = 1:Ntrain
    x_t = trainCNN(t,:);             % 获取第 t 个训练样本
    y_t = trainLabels(t) + 1;        % 获取该样本的标签（+1，保证标签从 1 开始）
    yy = w * x_t' - b;               % 计算 w * x_t' - b
    ind = find(yy < 0);              % 找到得分小于 0 的索引
    yhat_t = ind(1);                  % 预测标签

    if yhat_t ~= y_t
        % 如果预测标签与真实标签不一致，进行更新
        yr_t = double([1:Nclasses-1]' < y_t);  % 计算真实标签的标准（1 或 -1）
        yr_t(yr_t == 0) = -1;                  % 将非正标签设置为 -1

        itar_t = double(yy(1:end-1) .* yr_t <= 0);  % 更新准则
        itar_t(itar_t == 1) = yr_t(itar_t == 1);    % 更新训练集标签

        % 更新权重和偏置
        w = w + sum(itar_t) * x_t;
        b(1:end-1) = b(1:end-1) - itar_t;   % 偏置更新
    end
end
toc;  % 计时结束

% LSH（局部敏感哈希）映射
lshW = randn(10, 8);  % 初始化类别哈希映射矩阵
train_class_hash = single(label_Tr * lshW > 0);  % 训练数据的类别哈希
train_class_hash(train_class_hash <= 0) = -1;    % 将 <= 0 的值设置为 -1

lshD = randn(Dtrain, 8);  % 初始化数据哈希映射矩阵
train_hash = single(trainCNN * lshD > 0);  % 训练数据的哈希映射
train_hash(train_hash <= 0) = -1;   % 将 <= 0 的值设置为 -1

test_hash = single(testCNN * lshD);  % 测试数据的哈希映射
test_hash(test_hash <= 0) = -1;     % 将 <= 0 的值设置为 -1

% 测试集预测过程
tic;
pred = zeros(Ntest, 10);  % 初始化测试集的预测矩阵
for t = 1:Ntest
    x = testCNN(t,:);  % 获取第 t 个测试样本
    yy = w * x' - b;   % 计算预测得分
    ind = find(yy < 0);  % 找到得分小于 0 的索引
    pred(t, ind(1)) = 1;  % 更新预测矩阵
end
toc;

% 测试集的类别哈希
test_class_hash = single(pred * lshW > 0);  % 测试数据的类别哈希
test_class_hash(test_class_hash <= 0) = -1;  % 将 <= 0 的值设置为 -1

% 合并训练集和测试集的哈希特征
TRAIN = [train_class_hash, train_hash];  % 训练集哈希特征
TEST = [test_class_hash, test_hash];     % 测试集哈希特征

% 计算训练集和测试集标签之间的相似度矩阵
Aff = trainLabels * testLabels';  % 计算标签之间的相似度
