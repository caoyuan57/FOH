function res = evaluate(Htrain, Htest, opts, Aff)
% Given the hash codes of the training data (Htrain) and the hash codes of the test
% data (Htest) evaluates the performance.
% 
% INPUTS
% 	Htrain - (logical) Matrix containing the hash codes of the training
% 			   data. Each column corresponds to a hash code. 
%  	Htest  - (logical) Matrix containing the hash codes of the test data. 
%			   Each column corresponds to a hash code.
%	opts   - (struct)  Parameter structure.
%       Aff    - (logical) Neighbor indicator matrix. trainingsize x testsize. 
%
% OUTPUTS
%  	res    - (float) performance value as determined by opts.metric

[trainsize, testsize] = size(Aff);

if strcmp(opts.metric, 'mAP')
    % eval mAP
    AP  = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j = 1:testsize
        labels = 2*Aff(:, j)-1;
        [~, ~, info] = vl_pr(labels, double(sim(:, j)));
        AP(j) = info.ap;
    end
    res = mean(AP(~isnan(AP)));
    logInfo(['mAP = ' num2str(res)]);

elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eval mAP on top N retrieved results
    assert(isfield(opts, 'mAP') & opts.mAP > 0);
    assert(opts.mAP < trainsize);
    N = opts.mAP; 
    AP = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j = 1:testsize
        sim_j = double(sim(:, j));
        idx = [];
        for th = opts.nbits:-1:-opts.nbits
            idx = [idx; find(sim_j == th)];
            if length(idx) >= N, break; end
        end
        labels = 2*Aff(idx(1:N), j)-1;
        [~, ~, info] = vl_pr(labels, sim_j(idx(1:N)));
        AP(j) = info.ap;
    end
    res = mean(AP(~isnan(AP)));
    logInfo('mAP@(N=%d) = %g', N, res);

elseif ~isempty(strfind(opts.metric, 'prec_k'))
    % eval precision @ k (nearest neighbors)
    K = opts.prec_k; 
    prec_k = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for i = 1:testsize
        labels = Aff(:, i);
        sim_i = sim(:, i);
        [~, I] = sort(sim_i, 'descend');
        I = I(1:K);
        prec_k(i) = mean(labels(I));
    end
    res = mean(prec_k);
    logInfo('Prec@(neighs=%d) = %g', K, res);

elseif ~isempty(strfind(opts.metric, 'prec_n'))
    % eval precision @ N (Hamming ball radius)
    N = opts.prec_n; 
    R = opts.nbits;
    prec_n = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j=1:testsize
        labels = Aff(:, j);
        ind = find(R-sim(:,j) <= 2*N);
        if ~isempty(ind)
            prec_n(j) = mean(labels(ind));
        end
    end
    res = mean(prec_n);
    logInfo('Prec@(radius=%d) = %g', N, res);

else
    error(['Evaluation metric ' opts.metric ' not implemented']);
end
end

% ----------------------------------------------------------
function sim = compare_hash_tables(Htrain, Htest)
trainsize = size(Htrain, 1);
testsize  = size(Htest, 1);
if trainsize < 100e3
    sim = (2*single(Htrain)-1)*(2*single(Htest)-1)';
    sim = int8(sim);
else
    % for large scale data: process in chunks
    Ltest = 2*single(Htest)-1;
    sim = zeros(trainsize, testsize, 'int8');
    chunkSize = ceil(trainsize/10);
    for i = 1:ceil(trainsize/chunkSize)
        I = (i-1)*chunkSize+1 : min(i*chunkSize, trainsize);
        tmp = (2*single(Htrain(:,I))-1) * Ltest';
        sim(I, :) = int8(tmp);
    end
    clear Ltest tmp
end
end
