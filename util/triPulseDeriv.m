function y = triPulseDeriv(D, mid, delta);
% vectorized version
% mid: scalar bin center
%   D: can be a matrix
ind1 = (D > mid-delta) & (D <= mid);
ind2 = (D > mid) & (D <= mid+delta);
y = (ind1 - ind2) / delta;
end
