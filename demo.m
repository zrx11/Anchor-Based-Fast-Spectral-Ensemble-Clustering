clear;
clc;

addpath('functions');
addpath('Measures');
addpath('datas');
fileList = dir('datas\*.mat');
fileNum = length(fileList);

numAnchors = 10;
numNearestAnchor = 5;
numBase = 20;
cntTimes = 10;

for i=1:fileNum
    file = fileList(i).name;
    fileName = file(1:length(file)-4);
    outcomes = zeros(cntTimes, 3);   
    for k=1:cntTimes
        outcome = FSEC(file, numNearestAnchor, numBase, numAnchors);
        outcomes(k, :) = outcome;
    end
    out = [mean(outcomes); std(outcomes)];
    save([fileName, '_out'], 'out'); 
end