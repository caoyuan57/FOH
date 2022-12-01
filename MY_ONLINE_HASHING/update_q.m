function [q,qlabel] = update_q(q,k,Xs_t,ls_t,qlabel)
%Update the query pool%
n_t = size(Xs_t,2);
for i = 1:n_t
    d = randi(i+k);
    if d<k
        q(:,d) = Xs_t(:,i);
        qlabel(d,:) = ls_t(i,:);
    end
end
