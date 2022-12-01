load ../data/CIFAR10_VGG16_fc7.mat

trainCNN = trainCNN;
trainLabels = trainLabels;
testCNN = testCNN;
testLabels = testLabels;

% load ../data/Places205_AlexNet_fc7_PCA128.mat
% [trainCNN, trainLabels, testCNN, testLabels] = mod_split_dataset(pca_feats, labels, 20);


[Ntrain, Dtrain] = size(trainCNN);
[Ntest, Dtest] = size(testCNN);
Nclasses = length(unique(trainLabels));
w = zeros(1, Dtrain);
b = zeros(Nclasses, 1);
b(end) = inf;

label_Tr = zeros(Ntrain, 10);
label_Te = zeros(Ntest, 10);

for i = 1:Ntrain
    label_Tr(i, trainLabels(i)+1) = 1;
end

for i = 1:Ntest
    label_Te(i, testLabels(i)+1) = 1;
end




% PRank algorithm
tic;
for t = 1:Ntrain
    x_t = trainCNN(t,:);
    y_t = trainLabels(t) + 1;
    yy = w * x_t' -b;
    ind = find(yy < 0);
    yhat_t = ind(1);
    
    if yhat_t ~= y_t
        
        yr_t = double([1:Nclasses-1]' < y_t);
        yr_t(yr_t==0) = -1;
        
 
    
        itar_t = double(yy(1:end-1) .* yr_t <= 0);
        itar_t(itar_t == 1) = yr_t(itar_t == 1);
        
        
        w = w + sum(itar_t) * x_t;
        b(1:end-1) = b(1:end-1) - itar_t;
    end
    
    
end
toc


lshW = randn(10, 8);
train_class_hash = single(label_Tr * lshW > 0);
train_class_hash(train_class_hash <= 0) = -1;

lshD = randn(Dtrain, 8);
train_hash = single(trainCNN * lshD > 0);
train_hash(train_hash <= 0) = -1;
test_hash = single(testCNN * lshD);
test_hash(test_hash <= 0) = -1;



tic;
pred = zeros(Ntest, 10);
for t = 1:Ntest
    x = testCNN(t,:);
    yy = w * x' - b;
    ind = find(yy<0);
    pred(t, ind(1)) = 1;
end
toc

test_class_hash = single(pred * lshW > 0);
test_class_hash(test_class_hash <= 0) = -1;


TRAIN = [train_class_hash, train_hash];
TEST = [test_class_hash, test_hash];

Aff = trainLabels * testLabels';

