function A = myConstructA_NP(TrainData, Anchor, k, newIdx, flag)
%d*n
n = size(TrainData, 1);
p = size(Anchor, 1);
m = size(newIdx, 2);
Dis = zeros(n, m);
for i = 1:p
    Dis(flag==i, :) = EuDist2(TrainData(flag==i, :), Anchor(newIdx(i, :), :));
end

% Dis = Dis + eps*ones(size(Dis,1),size(Dis,2));
[~,idx] = sort(Dis,2);%3.673212 seconds.
idx1 = idx(:,1:k+1);
clear idx;
[anchor_num, ~] = size(Anchor);
[num, ~] = size(TrainData);
A = zeros(num,anchor_num);
for i = 1:num
    id = idx1(i,1:k+1);
    di = Dis(i,id);
    dii = newIdx(flag(i), :);
    A(i, dii(:, id)) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end