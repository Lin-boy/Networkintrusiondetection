# ----------------------------------------
# Written by Xiao Feng
# ----------------------------------------
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

traindata = pd.read_csv("/content/sample_data/event.csv",encoding="gb18030",header=None,low_memory=False)
traindata.dropna(axis=1,thresh=10,inplace=True)
traindata.fillna("0",inplace=True)
onehot = np.zeros((traindata.shape[0]-1,1));

enc = preprocessing.LabelEncoder()
for i in range(0,traindata.shape[1]):
  tmp = enc.fit_transform(traindata.iloc[1:traindata.shape[0],i])
  onehot = np.hstack((onehot,tmp.reshape(-1,1)))
X = np.hstack((onehot[:,1:2],onehot[:,4:]))
label = onehot[:,2]

X_train, X_test, label_train, label_test = train_test_split(X, label, test_size=0.33, random_state=0)

truelabel = label_test

##归一化
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)


#Naive Bayes model
model = GaussianNB()
model.fit(X_train, label_train)

expected = label_test
predicted = model.predict(X_test)

np.savetxt("NBM.txt",np.hstack((expected.reshape(-1, 1), predicted.reshape(-1, 1))), fmt='%01d')


# KNN
model = KNeighborsClassifier()
model.fit(X_train, label_train)

expected = label_test
predicted = model.predict(X_test)

np.savetxt("KNN.txt",np.hstack((expected.reshape(-1, 1), predicted.reshape(-1, 1))), fmt='%01d')
