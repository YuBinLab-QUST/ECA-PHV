clear all
clc
load('abc.mat');
input=importdata('virus-test1.txt');
data=input(2:2:end,:);
[m,n]=size(data);
output=[];
for i=1:m;
 output= [output,MMI(data{i},abc)];
 out=output';
end
save MMI-virus-test.mat out
