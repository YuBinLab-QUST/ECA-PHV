# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:54:19 2019

@author: LX
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import scipy.io as sio
from dimensional_reduction import TSVD
  

data_train=pd.read_csv('')
data_=np.array(data_train)
data=data_[:,1:]
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
svd = TruncatedSVD(n_components=500, n_iter=10, random_state=42)
hist=svd.fit(X)  
new_data=svd.transform(X)
data_csv = pd.DataFrame(data=new_data)
data_csv.to_csv('')