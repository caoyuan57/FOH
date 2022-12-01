function s = sigmoid(x, p)
s = p .* x;
s = min(42, max(-42, s));
s = 1 ./ (1 + exp(-s));
end
