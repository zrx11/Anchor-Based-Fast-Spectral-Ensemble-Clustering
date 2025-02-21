function acc=cluster_acc(label,pred)
%Find the clustering accuracy of prediction, given the true labels
%Output
% acc = Accuracy of clustering results

%Input
% label = a vector of true labels
% pred = a vector of the predicted labels

% Written by Dong Dong (dongdongwork@gmail.com).

ytrue=int64(label);
ypred=int64(pred);

n=length(ytrue); % number of samples
m=length(ypred); % number of samples in the prediction
if n~=m 
    error('The dimensions of two vectors do not match');
end
s=min([ytrue(:);ypred(:)])-1; 
if s<0 %make sure all labels are positive
    ytrue=ytrue-s;
    ypred=ypred-s;
end
D=max([ytrue(:);ypred(:)]); % get the largest label
w=zeros(D);
for i=1:n  %get the confusion matrix
    w(ypred(i),ytrue(i))=w(ypred(i),ytrue(i))+1;
end
M=matchpairs(w, -1, 'max'); %solve the linear assignment problem
acc=sum(w(sub2ind(size(w), M(:,1), M(:,2))))/n;
