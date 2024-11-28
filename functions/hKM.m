% hKM(X',[1:500],2,1)，
%X: n*d
%idx0: [1:n]
%k: 2^k cluster number
%count: 1
function [ys, centers] = hKM(X, idx0, k, count)


n = size(X,2);

X0 = X(:,idx0);

% centers为簇中心，F为指示矩阵，y是类别归属
[centers, F, y] = BalancedKM_with_one_XX(X0,0.5);

ys = 2*count+1-y;
% 使标签与簇中心对应
ys = sum(unique(ys))-ys;
if k > 1
    id1 = find(y==1);
    idx1 = idx0(id1);
    [ys1, centers1] = hKM(X,idx1,k-1,2*count-1);

    id2 = find(y==2);
    idx2 = idx0(id2);
    [ys2, centers2] = hKM(X,idx2,k-1,2*count);

    ys(id1) = ys1;
    ys(id2) = ys2;
    centers = [centers1, centers2];
end