
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import torch.nn.functional as F
import pandas as pd
import dgl
import dgl.nn as dglnn
import random
from sklearn.model_selection import KFold
import math
from scipy import interp
from sklearn.metrics import roc_curve,auc
import os
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def load_data():
    data1 = pd.read_csv('',index_col = 0).values
    data2 = pd.read_csv('',index_col = 0).values
    data3 = pd.read_csv('',index_col = 0).values
    data4 = pd.read_csv('',index_col = 0).values
    data5 = pd.read_csv('',index_col = 0).values
    features1 = torch.FloatTensor(data1)
    features2 = torch.FloatTensor(data2)
    features3 = torch.FloatTensor(data3)
    features4 = torch.FloatTensor(data4)
    features5 = torch.FloatTensor(data5)
    features = pd.read_csv('',index_col=0).values
    
    features = torch.FloatTensor(features)
    labels = pd.read_csv('',index_col=0)
    labels = labels.values
    labels = labels.T
    labels = torch.LongTensor(labels)
    labels = torch.squeeze(labels)
    g = dgl.knn_graph(features, int(features.size()[0]/10))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, features1, features2, features3, features4, features5, labels



g, features1, features2, features3, features4, features5, labels = load_data()

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
#        h = F.relu(h) 
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


def fit_fun(X):  # 适应函数
    features = torch.cat((X[0]*features1,X[1]*features2,X[2]*features3,X[3]*features4,X[4]*features5),1)
    def classfier():

# main loop
        kf = KFold(n_splits=5,shuffle=False)
        index = kf.split(X=features ,y=labels)
        dur = []
        acc1 = []
        precision1 = []
        npv1 = []
        sensitivity1 = []
        specificity1 = []
        mcc1 = []
        f11 = []
        tprs=[]
        aucs=[]
        mean_fpr=np.linspace(0,1,100)

        for train_index,test_index in index:
            net = SAGE(in_feats=features.size()[1], hid_feats=int(features.size()[1]/2), out_feats=2)           
# create optimizer
#optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

            for epoch in range(100):
                t0 = time.time()
                logits = net(g,features)
                m = nn.LogSoftmax(dim=1)
                criteria = nn.NLLLoss()
                loss = criteria(m(logits[train_index,:]), labels[train_index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                dur.append(time.time() - t0)
                print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), np.mean(dur)))    
            a=logits
#print(a[:, [0]]) ## 第0列
            list=[]
            for i in test_index: 
#    print(a[[i],:]) ## 第1行
                b=a[[i],:]
                if b[:, [0]] > b[:, [1]]:
                    x=0
#        print (x)
                else:
                    x=1
#        print (x)
                list.append(x)
#print(list)
            pred_y=np.array(list)
            labels1=labels[test_index]
            tp = 0
            fp = 0
            tn = 0
            fn = 0
#    for index in range(test_num):
            for index in range(len(test_index)): 
                if labels1[index] ==1:
                    if labels1[index] == pred_y[index]:
                        tp = tp +1
                    else:
                        fn = fn + 1
                else:
                    if labels1[index] == pred_y[index]:
                        tn = tn +1
                    else:
                        fp = fp + 1               
            
            acc = float(tp + tn)/len(test_index)
            precision = float(tp)/(tp+ fp + 1e-06)
            npv = float(tn)/(tn + fn + 1e-06)
            sensitivity = float(tp)/ (tp + fn + 1e-06)
            specificity = float(tn)/(tn + fp + 1e-06)
            mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
            f1=float(tp*2)/(tp*2+fp+fn+1e-06)

    
    
            acc1.append(acc)
            precision1.append(precision)
            npv1.append(npv)
            sensitivity1.append(sensitivity)
            specificity1.append(specificity)
            mcc1.append(mcc)
            f11.append(f1)
#    print(acc,precision,npv,sensitivity,specificity,mcc,f1)
            fpr,tpr,thresholds=roc_curve(labels1,pred_y)
    #interp:插值 把结果添加到tprs列表中 
            tprs.append(interp(mean_fpr,fpr,tpr))
            tprs[-1][0]=0.0
    #计算auc
            roc_auc=auc(fpr,tpr)
            aucs.append(roc_auc)
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
#            plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
#            i +=1
  
#画对角线
#        plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
        mean_tpr=np.mean(tprs,axis=0)
        mean_tpr[-1]=1.0
        mean_auc=auc(mean_fpr,mean_tpr)
        print(mean_auc)
        return mean_auc
    return classfier()


#print(func3([-2,-3]))
##################初始化##############
NP=100                               #个体数目
D=5                                 #变量的维数
G=100                                #最大进化代数
f=0.5                                #变异算子
CR=0.5                               #交叉算子
Xs=2                                 #上限
Xx=-2                                #下限
ob = np.zeros(NP)  # 存放个体目标函数值(父辈）
ob1 = np.zeros(NP)  # 存放个体目标函数值(子代）
##################赋初值##############
X = np.zeros((NP, D))  # 初始种群 (个体数目,维数）
v = np.zeros((NP, D))  # 变异种群  (个体数目,维数）
u = np.zeros((NP, D));  # 选择种群 (个体数目,维数）
#X = np.random.randint(Xx, Xs, (NP, D))  # 赋初值  (xx-xs之间的随机整数 ，(个体数目,维数）
X = np.random.uniform(Xx, Xs, (NP, D))  # 赋初值  (xx-xs之间的随机数 ，(个体数目,维数）


trace = []  # 记录每次迭代的最小适应值
###################计算当前群体个体目标函数值#############
for i in range(NP):  # 遍历每一个个体
    ob[i] = fit_fun(X[i, :])
trace.append(np.max(ob))

###################差分进化循环#############
for gen in range(G):  # 遍历每一代
    ############变异操作###########
    ############r1,r2,r3和m互不相同########
    for m in range(NP):  # 遍历每一个个体
        r1 = np.random.randint(0, NP, 1)
        while r1 == m:  # r1不能取m
            r1 = np.random.randint(0, NP, 1)

        r2 = np.random.randint(0, NP, 1)
        while (r2 == m) or (r2 == r1):  # r2不能取m 和r1
            r2 = np.random.randint(0, NP, 1)
        r3 = np.random.randint(0, NP, 1)
        while (r3 == m) or (r3 == r2) or (r3 == r1):  # r3不能取m，r2,r1
            r3 = np.random.randint(0, NP, 1)
#        v[m, :] = np.floor(X[r1, :] + f * (X[r2, :] - X[r3, :]))  # v.shape =(20, 2)  存放的是变异后的种群  np.floor 向下取整
        v[m, :] = X[r1, :] + f * (X[r2, :] - X[r3, :])  # v.shape =(20, 2)  存放的是变异后的种群


     ######交叉操作######
    r = np.random.randint(0, D, 1)  # 随机选择维度 （即选择x的一维）
    for n in range(D):  # 遍历每一个维度
        cr = np.random.random()  # 生成一个0-1之间的随机数
        if (cr < CR) or (n == r):  # 如果随机数小于交叉算子 或者 当前维数 等于r
            u[:, n] = v[:, n]  # 则选择群体个体维数 为变异后的维数
        else:
            u[:, n] = X[:, n]  # 为原始维度

     #####边界条件处理####
    for m in range(NP):  # 遍历每一个个体
        for n in range(D):  # 遍历每一个维度
            if (u[m, n] < Xx) or (u[m, n] > Xs):  # 如果当前元素不处于最大值和最小值之间
                u[m, n] = np.random.randint(Xx, Xs)  # 则重新初始化该元素
    #####选择操作####
    for m in range(NP):  # 遍历每一个个体
        ob1[m] = fit_fun(u[m, :])  # 计算子代个体适应度值
    for m in range(NP):  # 遍历每一个个体
        if ob1[m] > ob[m]:  # 如果子代个体适应度值大于父代个体适应度值
            X[m, :] = u[m, :]  # 则替换个体
    for m in range(NP):  # 遍历每一个个体
        ob[m] = fit_fun(X[m, :])  # 修改父代适应度值

    trace.append(max(ob))  # 记录当代最优适应度值

index = np.argmax(ob)  # 取出最大值所在位置索引
print('最优值解\n', X[index, :])
print('最优值\n', fit_fun(X[index, :]))
print('最优值解\n', X[index, :])
plt.plot(trace)
plt.title('迭代曲线')
plt.show()
