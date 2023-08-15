import math
from keras.layers import *
from keras.layers import Activation
import keras.backend as K
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU, Bidirectional, Reshape, GlobalAveragePooling1D
from keras.models import Model
from sklearn.metrics import roc_curve, auc 
from sklearn.preprocessing import scale
import utils.tools as utils
import keras.layers
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Input, Lambda
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.regularizers import l2
from tensorflow.keras import datasets,optimizers
from keras import layers




def to_class(p):
    return np.argmax(p, axis=1)


def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


# Origanize data
def get_shuffle(dataset, label):
    # shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label



data_=pd.read_csv(r'')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
#label1=np.ones((int(m1/2),1))#Value can be changed
#label2=np.zeros((int(m1/2),1))
label1=np.ones(((11341),1)) #Value can be changed
label2=np.zeros(((11341),1))
label=np.append(label1,label2)
X_=scale(data)
y_= label
X,y=get_shuffle(X_,y_)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5





def get_CNN_model(input_dim):
    gru = Sequential()
    gru.add(Bidirectional(GRU(int(input_dim/2), return_sequences=True),input_shape=(1,input_dim)))
    gru.add(Dropout(0.5))
    gru.add(Bidirectional(GRU(int(input_dim/4), return_sequences=True)))
    gru.add(Dropout(0.5))
    gru.add(Conv1D(filters = 32, kernel_size = 1, padding = 'same', activation= 'relu'))
    gru.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    return gru



def eca_block(input_dim):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = int(input_dim.shape[channel_axis])
    kernel_size = int(abs((math.log(channel, 2) + 1) / 2))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
 
    avg_pool = GlobalAveragePooling1D()(input_dim)
 
    x = Reshape((channel, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_" , use_bias=False, )(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1,1, channel))(x)

    output_eca = keras.layers.multiply([input_dim, x])
    return output_eca




input_shape =X.shape[1:]
def model(input_dim, out_dim,sample_num):
    input_layer = keras.Input(shape=(None,input_dim))
    gru_out = get_CNN_model(input_dim)(input_layer)

    eca = eca_block(gru_out)
    eca = layers.add([gru_out,eca])
    eca = Lambda(lambda x: K.squeeze(x, axis=1))(eca)
    print(eca.shape,"*"*100)
            
    output1 = Dense(units = input_dim//4, activation='relu')(eca)
    output2 = Dense(units = input_dim//8, activation='relu')(output1) 
    

    output = Dense(units=out_dim, activation='softmax', name="Dense_2")(output2)
    model = Model(input_layer,output)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


[sample_num, input_dim] = np.shape(X)
out_dim = 2
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
probas_rnn = []
tprs_rnn = []
sepscore_rnn = []
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(X, y):
    clf_rnn = model(input_dim, out_dim,sample_num)
    X_train_rnn = np.reshape(X[train], (-1, 1, input_dim))
    X_test_rnn = np.reshape(X[test], (-1, 1, input_dim))
    clf_list = clf_rnn.fit(X_train_rnn, to_categorical(y[train]).reshape((-1, 1, 2)), epochs=57) # nb_epoch鏀逛负epochs
    y_rnn_probas = clf_rnn.predict(X_test_rnn)
    
    
    y_rnn_probas = y_rnn_probas.reshape((-1, 2))
    probas_rnn.append(y_rnn_probas)
    y_class = utils.categorical_probas_to_classes(y_rnn_probas)

    y_test = utils.to_categorical(y[test])  # generate the test
    ytest = np.vstack((ytest, y_test))
    y_test_tmp = y[test]
    yscore = np.vstack((yscore, y_rnn_probas))

    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class, y[test])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y[test], y_rnn_probas[:, 1])
    tprs_rnn.append(interp(mean_fpr, fpr, tpr))
    tprs_rnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_rnn.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])

row = ytest.shape[0]
ytest = ytest[np.array(range(1, row)), :]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('')

yscore_ = yscore[np.array(range(1, row)), :]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('')

scores = np.array(sepscore_rnn)
result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscore_rnn.append(H1)
result = sepscore_rnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('')
