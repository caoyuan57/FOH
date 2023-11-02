%The similarity construction algorithm proposed in this paper%
%For multi lable dataset%
S = zeros(20015,20015);
load('..\data\MIRFLICKR.mat','LAll');
for i = 1 : 20015
    c =  LAll(i,:);
    for j = 1 : 20015
        d =  LAll(j,:);
        e = length(intersect(find(c==1),find(d==1)));
        m = length(find(d==1));
        n = length(find(c==1));
        S(i,j) = (e/m + e/n)/2;
    end
end
save('S20015.mat','S');
    
