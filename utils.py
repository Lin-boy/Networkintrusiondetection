# ----------------------------------------
# Written by Xiao Feng
# ----------------------------------------
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from config import cfg

def data_load_process(path):
    traindata = pd.read_csv(path,encoding="gb18030",header=None,low_memory=False)
    #traindata = pd.read_csv(path,header=None,low_memory=False)
    traindata.dropna(axis=1,thresh=10,inplace=True)
    traindata.fillna("0",inplace=True)
    onehot = np.zeros((traindata.shape[0]-1,1))
    enc = preprocessing.LabelEncoder()
    for i in range(0,traindata.shape[1]):
      tmp = enc.fit_transform(traindata.iloc[1:traindata.shape[0],i])
      onehot = np.hstack((onehot,tmp.reshape(-1,1)))
    X = np.hstack((onehot[:,1:2],onehot[:,4:]))
    label = onehot[:,2]
    assert X.shape[1] == cfg.IN_DIM
    print(int(max(label)))
    assert int(max(label))+1 == cfg.MODEL_NUM_CLASSES


    X_train, X_test, label_train, label_test = train_test_split(X, label, test_size=0.33, random_state=0)

    truelabel = label_test
    min_max_scaler = preprocessing.MinMaxScaler()
    trainX = min_max_scaler.fit_transform(X_train)
    testX = min_max_scaler.transform(X_test)
    label_train= to_categorical(label_train,num_classes=cfg.MODEL_NUM_CLASSES)
    label_test= to_categorical(label_test,num_classes=cfg.MODEL_NUM_CLASSES)

    if cfg.MODEL_NAME == 'LSTMModel':
        X_train = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
        X_test = testX.reshape(testX.shape[0], 1, testX.shape[1])

    elif cfg.MODEL_NAME =='CNNModel':
        X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
        X_test = np.reshape(testX, (testX.shape[0],testX.shape[1],1))
    else:
        X_train = np.array(trainX)
        X_test = np.array(testX)
    return X_train,label_train,X_test, label_test,truelabel


