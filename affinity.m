function A = affinity(X1, X2, Y1, Y2, opts)
% compute binary affinity matrix
%
if opts.unsupervised || isempty(Y1) || isempty(Y2)
    assert(~isempty(X1));
    assert(~isempty(X2));
    A = pdist2(X1, X2, 'Euclidean') <= opts.thr_dist;

elseif size(Y1, 2) == 1
    assert(size(Y2, 2) == 1);
    A = bsxfun(@eq, Y1, Y2');

else
    assert(size(Y2, 2) == size(Y1, 2));
    A = Y1 * Y2' > 0;
end

end
