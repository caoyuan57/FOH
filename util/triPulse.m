function y = triPulse(D, mid, delta)
% triPulse: triangular pulse
%
%     D: input matrix of distance values
%   mid: scalar, the center of some histogram bin
% delta: scalar, histogram bin width
%
% For histogram bin mid, compute the contribution y ("pulse") 
% from every element in D.  
% Interpolation using the triangular kernel
ind = (mid-delta < D) & (D <= mid+delta);
y   = 1 - abs(D - mid) / delta;
y   = y .* ind;
end
