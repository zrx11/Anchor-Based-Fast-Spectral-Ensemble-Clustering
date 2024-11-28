% X:num*dim
%alpha:regulization num
%numanchor:the real number of anchors is 2^numAnchor
function [Z,M] = myULGE(X,numAnchor,numNearestAnchor)

[num,~] = size(X);
% flag为样本点所属类别
[flag, locAnchor] = hKM(X',[1 :num],numAnchor,1); % here we use BKHK algorithm proposed in [2] to generate anchors.
M=locAnchor';

% 计算簇中心之间的距离并排序
Dis = EuDist2(M, M);
[~, idx] = sort(Dis, 2);
% 保留K'个最近邻
newIdx = idx(:, 1:10*numNearestAnchor);

%K'-Nearest
Z = myConstructA_NP(X,M,numNearestAnchor,newIdx,flag);
end