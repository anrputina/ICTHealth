close all
clear all
% Lab 5 - Prepare Data

load('a.mat');

keylist={'normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd','?',''};     
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];

%% Prepare Data

for kr = 1:size(chronickidneydisease,1)
    for kc = 1:size(chronickidneydisease,2)
        c = strtrim(chronickidneydisease(kr,kc));
        check=strcmp(c,keylist);% check(i)=1 if c==keylist(i)

        if sum(check)==0
            b(kr,kc)=str2num(chronickidneydisease{kr,kc});% from text to numeric
        else
            ii=find(check==1);
            b(kr,kc)=keymap(ii);% use the lists
        end;
    end
end

b=b(:,1:end-1);
[rows,columns]=size(b);
class_id = b(:,end);

%% Hierarchical Clustering

%Distance between each measurement 
%Transform into matrix the result
D = pdist(b(:,1:end-1));    
D_mat = squareform(D);

%Number of clusters
%Linkage and representation
K=2;
tree = linkage(D);
class_id_clustering = cluster(tree, 'maxclust', K);

p=0;
figure
dendrogram(tree,p)

%Compare clustering results with real data
perc_true = length(find(class_id==class_id_clustering))/rows;

%Square error
%Performance clustering algorithm

w1 = b(find(class_id==1),:);
w2 = b(find(class_id==2),:);

m_k(1,:) = mean(w1,1);
m_k(2,:) = mean(w2,1);

SSE=0;

for i=1:size(w1,1)
    error_1(i) = norm(w1(i,:)-m_k(1,:)).^2;
    SSE = SSE + error_1(i);
end

for i=1:size(w2,1)
    error_2(i) = norm(w2(i,:)-m_k(2,:)).^2;
    SSE = SSE + error_2(i);
end


%% Classification Tree

tc = fitctree(b(:,1:end-1),class_id);

view(tc);
view(tc,'Mode','graph');

for i=1:rows
    if b(i,15)<13.05
        if b(i,16)<44.5
            ct_class(i)=2;
        else
            ct_class(i)=1;
        end
    else
        if b(i,3)<1.0175
            ct_class(i)=2;
        else
            if b(i,4)<0.5
                ct_class(i)=1;
            else
                ct_class(i)=2;
            end
        end
    end
end

ct_class = ct_class';
perc_true = length(find(class_id==ct_class))/rows;