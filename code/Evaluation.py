# ----------------------------------------
# Written by Xiao Feng
# ----------------------------------------
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier



def LogALLINFO(FILENAME,SAVENAME):
    data = np.loadtxt(FILENAME)
    expected = data[:,0]
    predicted = data[:,1]
    acc = accuracy_score(expected, predicted)
    # precision = precision_score(expected, predicted,average='micro')
    # recall = recall_score(expected, predicted,average='micro')
    # f1 = f1_score(expected, predicted,average='micro')
    precision = precision_score(expected, predicted,average='macro')
    recall = recall_score(expected, predicted,average='macro')
    f1 = f1_score(expected, predicted,average='weighted')
    info = ['Accuracy',str(acc),'precision',str(precision),'recall',str(recall),'f1-score',str(f1)]
    with open(SAVENAME,"w") as f:
      for line in info:
        f.write(line)
        f.write('\n')

if __name__ == '__main__':
    data_name = ["LSTMModel","CNNModel","FCNModel","KNN","NBM"]
    for data in data_name:
        data_full_path = "/content/"+data+".txt"
        save_full = "/content/"+data+"_eval.txt"
        LogALLINFO(data_full_path,save_full)
