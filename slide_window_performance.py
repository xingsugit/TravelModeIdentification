import numpy as np
import time
import timeit
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import pandas as pd
import prepare_data
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

### test on slide window mechanis
### vary on lag, where lag == segLen means no slide window.
### return the predict accuracy
def slide_window_performance_test(segLen,merge_minor = True,ten_users = True):
    acc_list = []
    var_list = []
    timestamp = '2017-02-28'
    for lag in range(1,segLen+1):
        print('now we are at lag:',lag)
        df_android_winter, df_android_summer, df_iphone = prepare_data.prepareTravelData(segLen,lag,timestamp)
        df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
        #merge_minor: uid 4,6,7 only have jogging data. so we combine them into other user's data: 4->1, 6->2, 7->3.
        if merge_minor:
            df_all.loc[df_all['UID'] == 4,['UID']] = 1
            df_all.loc[df_all['UID'] == 6,['UID']] = 2
            df_all.loc[df_all['UID'] == 7,['UID']] = 3
        if ten_users:
            df_all = df_all.loc[df_all['UID'].isin([1,2,3,5,7,8,9,10,21,22,31])]
        print(len(df_all))
        #pivot_table = pd.pivot_table(df_all, values = ['count'], columns=['label'], aggfunc=np.sum)
        #print(pivot_table)
        df_all = df_all.loc[df_all['label'].isin(['car','bus','subway'])]

        x = df_all.as_matrix()[:,:-2]
        y = df_all.as_matrix()[:,-1]
        #y = ['active' if e in ['walk','jog','bike'] else 'still' for e in y]
        random_state = np.random.RandomState(0)
        clf = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
        scores = cross_val_score(clf, x, y, cv=10)
        accuracy = scores.mean()
        variance = scores.std() * 2
        acc_list.append(accuracy)
        var_list.append(variance)
    return acc_list,var_list

def segmentation_size_performance_test(merge_minor = True,ten_users = True):
    acc_list = []
    var_list = []
    timestamp = '2017-02-27'
    for segLen in [2,4,8,10,12,16,32,48,64,72,96,128,256]:
        lag = segLen
        print('Current segment size is:',segLen)
        df_android_winter, df_android_summer, df_iphone = prepare_data.prepareTravelData(segLen,lag,timestamp)
        df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
        #merge_minor: uid 4,6,7 only have jogging data. so we combine them into other user's data: 4->1, 6->2, 7->3.
        if merge_minor:
            df_all.loc[df_all['UID'] == 4,['UID']] = 1
            df_all.loc[df_all['UID'] == 6,['UID']] = 2
            df_all.loc[df_all['UID'] == 7,['UID']] = 3
        if ten_users:
            df_all = df_all[df_all['UID'].isin([1,2,3,5,7,8,9,10,21,22,31])]
        x = df_all.as_matrix()[:,:-2]
        y = df_all.as_matrix()[:,-1]
        random_state = np.random.RandomState(0)
        clf = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
        scores = cross_val_score(clf, x, y, cv=10,n_jobs = -1)
        accuracy = scores.mean()
        variance = scores.std() * 2
        acc_list.append(accuracy)
        var_list.append(variance)
    return acc_list,var_list

if __name__ == '__main__':
    segLen = 10
    acc_list,var_list = slide_window_performance_test(segLen)
    #acc_list,var_list = segmentation_size_performance_test()
    print(acc_list)
    print(var_list)
    plt.plot( range(1,len(acc_list)+1), acc_list, '.r-',label = 'Segment Length: 16',linewidth = 3)
    plt.show()
