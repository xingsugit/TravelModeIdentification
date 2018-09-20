import adaptive_learning as al
from collections import Counter

import time
import timeit
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#from pegasospak import pegasos
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import pandas as pd



print('hello world!')
acc_headers = ['XMAX', 'XMIN', 'XSTND', 'XAVG', 'XOFFSET', 'XFRQ', 'XENERGYSTND','X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9','XQUANTILE20', 'XQUANTILE40', 'XQUANTILE60', 'XQUANTILE80', 'YMAX','YMIN', 'YSTND', 'YAVG', 'YOFFSET', 'YFRQ', 'YENERGYSTND', 'Y0','Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'YQUANTILE20','YQUANTILE40', 'YQUANTILE60', 'YQUANTILE80', 'ZMAX', 'ZMIN','ZSTND', 'ZAVG', 'ZOFFSET', 'ZFRQ', 'ZENERGYSTND', 'Z0', 'Z1', 'Z2','Z3', 'Z4', 'Z5', 'Z6',
            'Z7', 'Z8', 'Z9', 'ZQUANTILE20','ZQUANTILE40', 'ZQUANTILE60', 'ZQUANTILE80', 'ACC_MAGNITUDE_MAX','ACC_MAGNITUDE_MIN', 'ACC_MAGNITUDE_STND', 'ACC_MAGNITUDE_AVG','ACC_MAGNITUDE_OFFSET', 'ACC_MAGNITUDE_FRQ','ACC_MAGNITUDE_ENERGYSTND', 'ACC_MAGNITUDE_0', 'ACC_MAGNITUDE_1','ACC_MAGNITUDE_2', 'ACC_MAGNITUDE_3', 'ACC_MAGNITUDE_4','ACC_MAGNITUDE_5', 'ACC_MAGNITUDE_6', 'ACC_MAGNITUDE_7','ACC_MAGNITUDE_8', 'ACC_MAGNITUDE_9', 'ACC_MAGNITUDE_QUANTILE20','ACC_MAGNITUDE_QUANTILE40',
            'ACC_MAGNITUDE_QUANTILE60','ACC_MAGNITUDE_QUANTILE80']
rotation_headers = ['X_Rotation_MAX', 'X_Rotation_MIN', 'X_Rotation_STND', 'X_Rotation_AVG', 'X_Rotation_OFFSET','X_Rotation_FRQ', 'X_Rotation_ENERGYSTND', 'X_Rotation_0',  'X_Rotation_1', 'X_Rotation_2', 'X_Rotation_3', 'X_Rotation_4','X_Rotation_5', 'X_Rotation_6', 'X_Rotation_7', 'X_Rotation_8','X_Rotation_9', 'X_Rotation_QUANTILE20', 'X_Rotation_QUANTILE40', 'X_Rotation_QUANTILE60', 'X_Rotation_QUANTILE80', 'Y_RotationMAX','Y_RotationMIN', 'Y_RotationSTND',
                'Y_RotationAVG','Y_RotationOFFSET', 'Y_RotationFRQ', 'Y_RotationENERGYSTND', 'Y_Rotation0', 'Y_Rotation1', 'Y_Rotation2', 'Y_Rotation3','Y_Rotation4', 'Y_Rotation5', 'Y_Rotation6', 'Y_Rotation7','Y_Rotation8', 'Y_Rotation9', 'Y_RotationQUANTILE20','Y_RotationQUANTILE40', 'Y_RotationQUANTILE60','Y_RotationQUANTILE80', 'Z_RotationMAX', 'Z_RotationMIN','Z_RotationSTND', 'Z_RotationAVG', 'Z_RotationOFFSET','Z_RotationFRQ', 'Z_RotationENERGYSTND', 'Z_Rotation0',
                'Z_Rotation1', 'Z_Rotation2', 'Z_Rotation3', 'Z_Rotation4','Z_Rotation5', 'Z_Rotation6', 'Z_Rotation7', 'Z_Rotation8','Z_Rotation9', 'Z_RotationQUANTILE20', 'Z_RotationQUANTILE40','Z_RotationQUANTILE60', 'Z_RotationQUANTILE80']
light_headers = ['Screen_brightness_MAX', 'Screen_brightness_MIN','Screen_brightness_STND', 'Screen_brightness_AVG']
pressure_headers = ['pressure_MAX','pressure_MIN', 'pressure_STND', 'pressure_AVG', 'magEnergy_MAX']
mag_headers = ['magEnergy_MIN', 'magEnergy_STND', 'magEnergy_AVG','magEnergy_FREQ', 'magEnergy_OFFSET']

df_header = acc_headers + rotation_headers + light_headers + pressure_headers + mag_headers + ['UID','class']

def prepareTravelData():
    android_winter_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_winter_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
    android_summer_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
    iphone_data_file = '/Users/Xing/Dropbox/TravelData/data/iPhone_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-02.csv'
    df_iphone = pd.read_csv(iphone_data_file,header = None)
    df_android_summer = pd.read_csv(android_summer_data_file,header = None)
    df_android_winter = pd.read_csv(android_winter_data_file,header = None)
    df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
    df_all.columns = df_header
    df_android_winter.columns = df_header
    df_android_summer.columns = df_header
    df_iphone.columns = df_header
    return df_all,df_android_winter,df_android_summer,df_iphone



def similar_data_weight_prepare(uid,topn,C,df,xi,ratio,ns_similarity_matrix,cz = False):
    print('current uid is:',uid)
    scaler = preprocessing.StandardScaler().fit(df.as_matrix()[:,:-2].astype(float))
    ## for test data
    print('prepare T')
    test_df = df.loc[df['UID'] == int(uid)]
    test = test_df.as_matrix()
    test_x = test[:,:-2]
    test_y = test[:,-1]
    print('Test data summary:')
    print(Counter(test_y))
    test_x_scaled = scaler.transform(test_x.astype(float))
    test_x_scaled, train_x_addin, test_y,train_y_addin = cv.train_test_split(test_x_scaled,test_y,test_size = ratio,random_state = 42)
    #get data of similarity, as well as non_similar_data.
    top_n_learning_data, non_similar_data = al.get_similar_data_from_mode_uid_pair_rank(uid,topn,df,xi,train_x_addin,ns_similarity_matrix,size_control = cz)
    cm = pd.unique(df.loc[df['UID'] == int(uid),'class'])

    ## for training data:
    train = top_n_learning_data.as_matrix()
    train_x = train[:,:-2] #because #2 is the uid
    train_y = train[:,-1]
    print('Top n similar learning data summary:')
    print(Counter(train_y))


    ### normalize (is there a better way to normalize?):
    print('************************************************************************')
    train_x_scaled = scaler.transform(train_x.astype(float))
    ##### Here we split test data into two different sets: one part uses as labeled data, for sample reweigting. The other part is for testing.
    ### the test_size in train_test_split is correspond to train_x_addin (which is the ratio in T we use for transfer learning)
    #weight = al.get_theoretical_weights(train_x_scaled,train_y,train_x_addin,train_y_addin,C)
    train_y = np.append(train_y, train_y_addin,0)
    train_x_scaled = np.append(train_x_scaled,train_x_addin,0)
    #max_w_s = max(weight) + 1
    #weight += [max_w_s for i in range(len(train_y_addin))]
    return train_x_scaled, train_y, test_x_scaled, test_y#,weight, max_w_s


### for every batch, first verify the model on it, and then update the model with this batch , and test on next batch. and so forth.
def online_offline_learning(uid,topn,C,df,xi,ratio,ns_similarity_matrix,cz = False):
    train_x_scaled, train_y,test_x_scaled, test_y = similar_data_weight_prepare(uid,topn,C,df,xi,ratio,ns_similarity_matrix,cz)
    online_acc_list = []
    offline_acc_list = []
    ##### SGD data from top 3 similar uids:
    random_state = np.random.RandomState(0)
    clf_online = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state)#,class_weight = 'balanced')
    clf_offline =  SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state)#,class_weight = 'balanced')
    clf_online.partial_fit(train_x_scaled, train_y,classes = np.unique(train_y))#,sample_weight = weight)
    clf_offline.fit(train_x_scaled,train_y)#,sample_weight=weight)
    batch_size = 50
    batch_start_index = 0
    print("total length of the available data:",len(test_x_scaled))
    while batch_start_index +  batch_size < len(test_x_scaled):
        print("Current batch starts at:",batch_start_index)
        batch_end_index = batch_start_index + batch_size
        online_predict = clf_online.predict(test_x_scaled[batch_start_index:batch_end_index,:])
        online_acc = sum(online_predict == test_y[batch_start_index:batch_end_index])*1.0/len(test_y[batch_start_index:batch_end_index])
        print("\033[1;31;40m top similar data accuracy: %f\033[0;32;40m" % online_acc)

        offline_predict = clf_offline.predict(test_x_scaled[batch_start_index:batch_end_index])
        offline_acc = sum(offline_predict == test_y[batch_start_index:batch_end_index])*1.0/len(test_y[batch_start_index:batch_end_index])
        print("\033[1;31;40m top similar data accuracy: %f\033[0;32;40m" % offline_acc)
        online_acc_list.append(online_acc)
        offline_acc_list.append(offline_acc)
        #### online updating with new data
        clf_online.partial_fit(test_x_scaled[batch_start_index:batch_end_index,:], test_y[batch_start_index:batch_end_index])#,sample_weight = max_w_s)
        #### prepare new data for offline training
        train_y = np.append(train_y, test_y[batch_start_index:batch_end_index],0)
        train_x_scaled = np.append(train_x_scaled,test_x_scaled[batch_start_index:batch_end_index,:],0)
        #weight += [max_w_s for i in range(batch_size)]
        #### offline retrain with updated data
        clf_offline.fit(train_x_scaled,train_y)#,sample_weight = weight)
        batch_start_index = batch_end_index
    return online_acc_list, offline_acc_list

### separate the train_x_scaled into inital_training and add_up training.
def online_offline_learning2(uid,topn,C,df,xi,ratio,ns_similarity_matrix,cz = False):
    train_x_scaled, train_y,test_x_scaled, test_y = similar_data_weight_prepare(uid,topn,C,df,xi,ratio,ns_similarity_matrix,cz)
    train_x_scaled_initial,train_x_addup,train_y_initial,train_y_addup = cv.train_test_split(train_x_scaled,train_y,test_size = 0.5, random_state = 1)
    online_acc_list = []
    offline_acc_list = []
    ##### SGD data from top 3 similar uids:
    random_state = np.random.RandomState(0)
    clf_online = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state)#,class_weight = 'balanced')
    clf_offline =  SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state)#,class_weight = 'balanced')
    clf_online.partial_fit(train_x_scaled_initial, train_y_initial,classes = np.unique(train_y))#,sample_weight = weight)
    clf_offline.fit(train_x_scaled_initial,train_y_initial)#,sample_weight=weight)
    batch_size = 20
    batch_start_index = 0
    print("total length of the available data:",len(test_x_scaled))
    while batch_start_index +  batch_size < len(test_x_addup):
        print("Current batch starts at:",batch_start_index)
        batch_end_index = batch_start_index + batch_size
        online_predict = clf_online.predict(test_x_scaled)
        online_acc = sum(online_predict == test_y)*1.0/len(test_y)
        print("\033[1;31;40m top similar data accuracy: %f\033[0;32;40m" % online_acc)

        offline_predict = clf_offline.predict(test_x_scaled)
        offline_acc = sum(offline_predict == test_y)*1.0/len(test_y)
        print("\033[1;31;40m top similar data accuracy: %f\033[0;32;40m" % offline_acc)
        online_acc_list.append(online_acc)
        offline_acc_list.append(offline_acc)
        #### online updating with new data
        clf_online.partial_fit(train_x_addup[batch_start_index:batch_end_index,:], train_y_addup[batch_start_index:batch_end_index])#,sample_weight = max_w_s)
        #### prepare new data for offline training
        train_y = np.append(train_y_initial, train_y_addup[batch_start_index:batch_end_index],0)
        train_x_scaled_initial = np.append(train_x_scaled_initial,train_x_addup[batch_start_index:batch_end_index,:],0)
        #weight += [max_w_s for i in range(batch_size)]
        #### offline retrain with updated data
        clf_offline.fit(train_x_scaled,train_y)#,sample_weight = weight)
        batch_start_index = batch_end_index
    return online_acc_list, offline_acc_list

def test_online_offline_similarity(topn,C,xi,ratio,merge_minor = True,hierarchy = False):
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    acc = []
    if merge_minor:
        df_all.loc[df_all['UID'] == 4,['UID']] = 1
        df_all.loc[df_all['UID'] == 6,['UID']] = 2
        df_all.loc[df_all['UID'] == 7,['UID']] = 3
    uids = sorted(pd.unique(df_all['UID']))
    uu = [1,2,3,5,8,9,10,21,22,23,31,32,33]
    if hierarchy:
        for u in uids:
            test_usr_mode_similarity_hierarchical(u,topn,C,df_all,cz = False)
    else:
        for u in uu:
            acc.append(test_usr_mode_similarity(u,topn,C,df_all,xi,ratio,ns_similarity_matrix,cz=True))
        tops_acc = [e[0] for e in acc]
        random_acc = [e[1] for e in acc]
        weighted_tops_acc = [e[2] for e in acc]
        rest_uid_acc = [e[3] for e in acc]
        rest_uid_weighted_acc = [e[4] for e in acc]
        ratio_compare_acc = [e[5] for e in acc]
        xi_text = str(xi*10)
        ratio_text = str(int(ratio*100))
        file_name = '../data/T_add_weight1_usr_mode_ratio_' + ratio_text+'_top_'+ str(topn) +'_performance_weighted_vs_random_C0dot003_xi'+xi_text+'.csv'
        result = pd.DataFrame({'uid':uu,'tops_acc':tops_acc,'random_acc':random_acc,'weighted_tops_acc':weighted_tops_acc,'rest_uid_acc':rest_uid_acc,'weighted_rest_uid_acc':rest_uid_weighted_acc,'ratio_compare_acc':ratio_compare_acc})
        result.to_csv(file_name, sep=',')
        print('finish generating file %s' % file_name)


def usr_online_offline_learning_comparision():
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    mergeMinor = True
    #uid = 8
    #paramter_performance_tour(uid)
    ratio = 0.02
    ratio_text = str(int(ratio*100))
    xi = 0.2
    xi_text = str(xi*10)
    C = 0.003
    topn = 3
    ns_similarity_matrix = al.similarity_calc.calculate_ns_similarity(xi)
    for u in [1]:
        online_acc_list,offline_acc_list = online_offline_learning(u,topn,C,df_all,xi,ratio,ns_similarity_matrix,cz = False)
        file_name = '../data/usr_' + str(u) + '_weighted_ratio_' + ratio_text+'_top_'+ str(topn) +'_online_offline_performance_C0dot003_xi'+xi_text+'.csv'
        result = pd.DataFrame({'online_acc':online_acc_list,'offline_acc':offline_acc_list})
        result.to_csv(file_name, sep=',')
        print('finish generating file %s' % file_name)


if __name__ == '__main__':
    usr_online_offline_learning_comparision()


