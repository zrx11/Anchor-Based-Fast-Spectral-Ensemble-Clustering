function outcome = FSEC(fileName, numNearestAnchor, numBase, numAnchor)

%load data
data = load(fileName);
X = data.X;

fileName = fileName(1:length(fileName)-4);

%normalization
attrib = mapminmax(X', 0, 1);
X = attrib';
Y = data.Y;
[n, ~] = size(X);
K = length(unique(Y));

if nargin < 4
    numAnchor = floor(log2(sqrt(n*K)));
end

%construct similarity and anchor matrix
[B, ~] = myULGE(X, numAnchor, numNearestAnchor);

%SVD decomposition
B = B./(sqrt(sum(B, 1))+1e-10);

if sum(sum(isnan(B)))>0
    outcome = zeros(1, 3);
else
    [U, ~, ~] = svd(B, 0);
    
    %generate multiple base clusterings
    numKs = randi([K, min(50, round(sqrt(n)))], numBase, 1);
    baseCls = zeros(n, numBase);
    for j=1:numBase
        nowK = numKs(j, 1);
        nowU = U(:, 1:nowK);
        %normalization
        nowU = nowU./vecnorm(nowU, 2, 2); 
        %k-means
        baseCls(:, j) = kmeans(nowU, nowK);
    end

    preLabel = consensus_function(baseCls, K);
    outcome = zeros(1, 3);
    result = ClusteringMeasure(Y, preLabel);
    outcome(1) = result(2);
    outcome(2) = cluster_acc(Y, preLabel);
    outcome(3) = RandIndex(Y, preLabel);
end
end