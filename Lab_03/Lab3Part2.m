close all
clear all
% Lab 3 - Execution

load('arrhythmiaCleaned.mat');

class_id = arrhythmiaCleaned(:,end);
y = arrhythmiaCleaned(:,1:end-1);

class1 = class_id(:) == 1;
class2 = class_id(:) == 2;

y1 = y(class1,:);
y2 = y(class2,:);

x1 = mean(y1);
x2 = mean(y2);

%COMPUTATION

xmeans = [x1;x2];
eny = diag(y*transpose(y));
enx = diag(xmeans*transpose(xmeans));
dotprod = y * transpose(xmeans);
%each y(n) and each x
[U,V]=meshgrid(enx,eny);
dist2=U+V-2*dotprod;


[dummy, position] = min(dist2.');
position = position';

predizione = position == class_id;














