close all
clear all
%Lab 1 - ICT HEALTH - Perform Regresson

load('finalMatrix.mat');

Npacients = 36;
rowPacients_train = finalMatrix(:,1) <= Npacients;
rowPacients_test =  finalMatrix(:,1) > Npacients;

data_train = finalMatrix(rowPacients_train,:);
data_test = finalMatrix(rowPacients_test,:);

m_data_train = mean(data_train,1);
v_data_train = var(data_train,1);
std_data_train = std(data_train,1);

data_train_norm = data_train - m_data_train;
data_train_norm = data_train_norm ./ sqrt(v_data_train) ;
var(data_train_norm)