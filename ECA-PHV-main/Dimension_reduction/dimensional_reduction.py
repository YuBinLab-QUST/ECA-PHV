# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.preprocessing import scale,StandardScaler
from sklearn.preprocessing import normalize,Normalizer
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from sklearn.decomposition import KernelPCA, SparsePCA,PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.manifold import SpectralEmbedding 
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

def zscore_scaler(data):
    data=scale(data)
    return data
def normalizer(data):
    data = Normalizer().fit_transform(data)
    return data
def minmaxscaler(data):
    data=MinMaxScaler().fit_transform(data)
    return data
def maxabsscaler(data):
    data=MaxAbsScaler().fit_transform(data)
    return data
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)#axis represents the obtained mean value by column
    stdVal=np.std(dataMat,axis=0)
    newData=dataMat-meanVal
    new_data=newData/stdVal
    return new_data,meanVal
def covArray(dataMat):
    #obtain the  covariance matrix
    covMat=np.cov(dataMat,rowvar=0)
    return covMat
def featureMatrix(dataMat):
    eigVals,eigVects=np.linalg.eig(np.mat(dataMat))
	#datermine the eigenvalue and eigenvector
    return eigVals,eigVects
def percentage2n(eigVals,percentage=0.99):  
    #percentage represents the rate of contribution
    sortArray=np.sort(eigVals)   #ascending sort 
    sortArray=sortArray[-1::-1]  #descending order
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num #num is the number of remaining principal component

def lassodimension(data,label,alpha=np.array([0.01,0.05,0.1])):#The alpha value range is 0.001, 0.002, 0.005, 0.01, 0.015, 0.02
    lassocv=LassoCV(cv=5, alphas=alpha).fit(data, label)
    x_lasso = lassocv.fit(data,label)#Substituting alpha for dimensionality reduction
    mask = x_lasso.coef_ != 0 
    new_data = data[:,mask] 
    return new_data,mask 
# using factor analysis to reduce the dimension
def factorAnalysis(data,percentage = 0.9):
    dataMat = np.array(data) 
    newData,meanVal=zeroMean(data)  #equalization
    covMat=covArray(newData)  #covariance matrix
    eigVals,eigVects=featureMatrix(covMat)
    n_components = percentage2n(eigVals,percentage)
    clf = FactorAnalysis(n_components=n_components)
    new_data = clf.fit_transform(dataMat)
    return new_data  # return the new data
# using principal component to reduce dimension
def pca(data,percentage = 0.9):  
    dataMat = np.array(data) 
    newData,meanVal=zeroMean(data)  #equalization
    covMat=covArray(newData)  #covariance matrix
    eigVals,eigVects=featureMatrix(covMat)
    n_components = percentage2n(eigVals,percentage)
    clf=PCA(n_components=n_components)  
    new_x = clf.fit_transform(dataMat)
    return new_x
#using kernel principal component to reduce dimension
def KPCA(data,percentage=0.9):
    dataMat = np.array(data)  
    newData,meanVal=zeroMean(data)  
    covMat=covArray(newData)  
    eigVals,eigVects=featureMatrix(covMat)
    n_components = percentage2n(eigVals,percentage)
    #n_component=n_components
    clf=KernelPCA(n_components=n_components,kernel='rbf',gamma=1/1318)#rbf linear poly
    new_x=clf.fit_transform(dataMat)
    return new_x
def SPCA(data,percentage=0.9):
    dataMat = np.array(data)  
    newData,meanVal=zeroMean(data)  
    covMat=covArray(newData)  
    eigVals,eigVects=featureMatrix(covMat)
    n_components = percentage2n(eigVals,percentage)
    #n_component=n_components
    clf=SparsePCA(n_components=n_components)#rbf linear poly
    new_x=clf.fit_transform(dataMat)
    return new_x
def SE(data,n_components=300):
    embedding = SpectralEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed
def LLE(data,n_components=300):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed
#Lasso中的正交匹配追踪
def omp_omp(data,label,n_nonzero_coefs=300):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(data, label)
    coef = omp.coef_
    idx_r, = coef.nonzero()
    #omp_cv = OrthogonalMatchingPursuitCV()
    #omp_cv.fit(X, y_noisy)
    #coef = omp_cv.coef_
    #idx_r, = coef.nonzero()
    new_data=data[:,idx_r]
    return new_data,idx_r
#卡方进行特征选择
def chi2_chi2(data,label,k=300):
    model_chi2= SelectKBest(chi2, k=k)
    new_data=model_chi2.fit_transform(data,label)
    return new_data
#SVC降维
def selectFromLinearSVC(data,label,lamda):
    lsvc = LinearSVC(C=lamda, penalty="l1", dual=False).fit(data,label)
    model = SelectFromModel(lsvc,prefit=True)
    new_data= model.transform(data)
    return new_data
def TSVD(data,n_components=500):
    svd = TruncatedSVD(n_components=n_components)
    new_data=svd.fit_transform(data)  
    return new_data