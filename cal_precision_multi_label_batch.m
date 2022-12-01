function [mAP] = cal_precision_multi_label_batch(B1, B2, data_label, query_label)
    mAP = 0;
    for i = 1 : size(B2, 1)
        F = bsxfun(@minus, B2(i,:), B1);
        F = abs(F);
        F = sum(F, 2);
        [~, ind] = sort(F);
        
        n = size(B1, 1);
        d = bsxfun(@plus, data_label(ind(1:n),:), query_label(i,:));
        [l] = max(d, [], 2);
        l = find(l == 2);
        if length(l) ~= 0
            truth = zeros(size(B1,1), 1);
            truth(l) = 1;
            truth_s = cumsum(truth);
            mAP = mAP + mean(truth_s(l) ./ l);
        end
        
    end
    mAP = mAP / size(B2, 1);
end


