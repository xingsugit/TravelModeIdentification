import time
import numpy as np
import timeit
from sklearn.svm import SVC
from pegasospak import pegasos
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import pandas as pd

from sklearn.naive_bayes import GaussianNB


def define_global_variables():
    global acc_headers, rotation_headers, light_headers, pressure_headers,mag_headers
    acc_headers = ['XMAX', 'XMIN', 'XSTND', 'XAVG', 'XOFFSET', 'XFRQ', 'XENERGYSTND','X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9','XQUANTILE20', 'XQUANTILE40', 'XQUANTILE60', 'XQUANTILE80', 'YMAX','YMIN', 'YSTND', 'YAVG', 'YOFFSET', 'YFRQ', 'YENERGYSTND', 'Y0','Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'YQUANTILE20','YQUANTILE40', 'YQUANTILE60', 'YQUANTILE80', 'ZMAX', 'ZMIN','ZSTND', 'ZAVG', 'ZOFFSET', 'ZFRQ', 'ZENERGYSTND', 'Z0', 'Z1', 'Z2','Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'ZQUANTILE20','ZQUANTILE40', 'ZQUANTILE60', 'ZQUANTILE80', 'ACC_MAGNITUDE_MAX','ACC_MAGNITUDE_MIN', 'ACC_MAGNITUDE_STND', 'ACC_MAGNITUDE_AVG','ACC_MAGNITUDE_OFFSET', 'ACC_MAGNITUDE_FRQ','ACC_MAGNITUDE_ENERGYSTND', 'ACC_MAGNITUDE_0', 'ACC_MAGNITUDE_1','ACC_MAGNITUDE_2', 'ACC_MAGNITUDE_3', 'ACC_MAGNITUDE_4','ACC_MAGNITUDE_5', 'ACC_MAGNITUDE_6', 'ACC_MAGNITUDE_7','ACC_MAGNITUDE_8', 'ACC_MAGNITUDE_9', 'ACC_MAGNITUDE_QUANTILE20','ACC_MAGNITUDE_QUANTILE40', 'ACC_MAGNITUDE_QUANTILE60','ACC_MAGNITUDE_QUANTILE80']
    rotation_headers = ['X_Rotation_MAX', 'X_Rotation_MIN', 'X_Rotation_STND', 'X_Rotation_AVG', 'X_Rotation_OFFSET','X_Rotation_FRQ', 'X_Rotation_ENERGYSTND', 'X_Rotation_0',  'X_Rotation_1', 'X_Rotation_2', 'X_Rotation_3', 'X_Rotation_4','X_Rotation_5', 'X_Rotation_6', 'X_Rotation_7', 'X_Rotation_8','X_Rotation_9', 'X_Rotation_QUANTILE20', 'X_Rotation_QUANTILE40', 'X_Rotation_QUANTILE60', 'X_Rotation_QUANTILE80', 'Y_RotationMAX','Y_RotationMIN', 'Y_RotationSTND', 'Y_RotationAVG','Y_RotationOFFSET', 'Y_RotationFRQ', 'Y_RotationENERGYSTND', 'Y_Rotation0', 'Y_Rotation1', 'Y_Rotation2', 'Y_Rotation3','Y_Rotation4', 'Y_Rotation5', 'Y_Rotation6', 'Y_Rotation7','Y_Rotation8', 'Y_Rotation9', 'Y_RotationQUANTILE20','Y_RotationQUANTILE40', 'Y_RotationQUANTILE60','Y_RotationQUANTILE80', 'Z_RotationMAX', 'Z_RotationMIN','Z_RotationSTND', 'Z_RotationAVG', 'Z_RotationOFFSET','Z_RotationFRQ', 'Z_RotationENERGYSTND', 'Z_Rotation0', 'Z_Rotation1', 'Z_Rotation2', 'Z_Rotation3', 'Z_Rotation4','Z_Rotation5', 'Z_Rotation6', 'Z_Rotation7', 'Z_Rotation8','Z_Rotation9', 'Z_RotationQUANTILE20', 'Z_RotationQUANTILE40','Z_RotationQUANTILE60', 'Z_RotationQUANTILE80']
    light_headers = ['Screen_brightness_MAX', 'Screen_brightness_MIN','Screen_brightness_STND', 'Screen_brightness_AVG']
    pressure_headers = ['pressure_MAX','pressure_MIN', 'pressure_STND', 'pressure_AVG', 'magEnergy_MAX']
    mag_headers = ['magEnergy_MIN', 'magEnergy_STND', 'magEnergy_AVG','magEnergy_FREQ', 'magEnergy_OFFSET']
    ## group: acc: 1:84, rotation:85:147, Screen_brightness: 148:151; Pressure: 152:155; Mag: 156:161
    global acc_index_begin, acc_index_end, rotation_index_begin,rotation_index_end,light_index_begin,light_index_end,pressure_index_begin,pressure_index_end,mag_index_begin,mag_index_end
    acc_index_begin = 0
    acc_index_end = 83
    rotation_index_begin = 84
    rotation_index_end = 146
    light_index_begin = 147
    light_index_end = 150
    pressure_index_begin = 151
    pressure_index_end = 156
    mag_index_begin = 157
    mag_index_end = 160
    global df_header
    df_header = acc_headers + rotation_headers + light_headers + pressure_headers + mag_headers + ['UID','class']
    global uid_header,label_headeri, uid_index, label_index
    uid_header = 'UID'
    label_header = 'class'
    uid_index = 162
    label_index = 163
    global indoor_mode,wheeled_mode, unwheeled_mode
    indoor_mode  = ['car','bus','subway']
    wheeled_mode = ['car','bus','subway','bike']
    unwheeled_mode = ['jog','walk']
    global travel_mode
    travel_mode = ['car','bus','subway','bike','walk','jog']
    global confusion_matrix1,confusion_matrix2,confusion_matrix3,confusion_matrix4,confusion_matrix5
    confusion_matrix1 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix2 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix3 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix4 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix5 = [[0 for i in range(6)] for j in range(6)]




def prepareTravelData():
    android_winter_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_winter_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
    android_summer_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
    iphone_data_file = '/Users/Xing/Dropbox/TravelData/data/iPhone_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-02.csv'

    df_android_winter = pd.read_csv(android_winter_data_file,header = None)
    df_android_summer = pd.read_csv(android_summer_data_file,header = None)
    df_iphone = pd.read_csv(iphone_data_file,header = None)


    df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
    print(df_all.describe())
    df_all.columns = df_header
    df_android_winter.columns = df_header
    df_android_summer.columns = df_header
    df_iphone.columns = df_header


    ### select some uids in df_iphone, to see whether the diversity is the reason for online learning no converge
    #df_iphone = df_iphone.loc[df_iphone['UID'].isin([5])]

    uids = sorted(pd.unique(df_all['UID']))
    return df_all,df_android_winter,df_android_summer,df_iphone

def prepare_train_test_data_from_orig_df(df,test_ratio):
    sd = df.as_matrix()
    dataset = sd[:,:-2] #because #2 is the uid
    labels = sd[:,-1]
    x_train,x_test,y_train,y_test = cv.train_test_split(dataset,labels,test_size = test_ratio,random_state = 42)
    ## process the data scale
    x_train_scaled = preprocessing.scale(x_train.astype(float))
    scaler = preprocessing.StandardScaler().fit(x_train.astype(float))
    x_test_scaled = scaler.transform(x_test.astype(float))
    return x_train_scaled,y_train,x_test_scaled,y_test
    unwheeled_mode = ['jog','walk']
    global travel_mode
    travel_mode = ['car','bus','subway','bike','walk','jog']
    global confusion_matrix1,confusion_matrix2,confusion_matrix3,confusion_matrix4,confusion_matrix5
    confusion_matrix1 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix2 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix3 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix4 = [[0 for i in range(6)] for j in range(6)]
    confusion_matrix5 = [[0 for i in range(6)] for j in range(6)]



def calc_acc(confusion_matrix):
    return 1.0 * sum(confusion_matrix[i][i] for i in range(6))/sum(sum(e[i] for i in range(6)) for e in confusion_matrix)


def processf1Label(y):
    y_new = ['wheeled' if y[i] in ['car','bus','bike','subway'] else 'unwheeled' for i in range(len(y))]
    return y_new


def processf21Label(y):
    y_new = ['indoor' if y[i] in ['car','bus','subway'] else 'bike' for i in range(len(y))]
    return y_new

### f21: 'car','bus','subway','bike'
def extractf21data(x,y):
    x_new = [x[i] for i in range(len(x)) if y[i] in wheeled_mode]
    y_new = [e for e in y if e in wheeled_mode]
    y_new = processf21Label(y_new)
    return x_new, y_new

def extractf22data(x,y):
    x_new = [x[i] for i in range(len(x)) if y[i] in unwheeled_mode]
    y_new = [e for e in y if e in unwheeled_mode]
    return x_new, y_new

def extractf3data(x,y):
    x_new = [x[i] for i in range(len(x)) if y[i] in indoor_mode]
    y_new = [e for e in y if e in indoor_mode]
    return x_new, y_new



## f1: acc + rotation (here f1 is the same as f22)
def prepare_f1_fixed_sensors_col(isHeaders):
    if isHeaders:
        f1_cols = acc_headers #+ rotation_headers
    else:
        f1_cols = range(acc_index_begin,acc_index_end+1)# + range(rotation_index_begin,rotation_index_end+1)
    return f1_cols

def prepare_f22_fixed_sensors_col(isHeaders):
    if isHeaders:
        f22_cols = acc_headers + rotation_headers
    else:
        f22_cols= range(acc_index_begin,acc_index_end+1)# + range(rotation_index_begin,rotation_index_end+1)
    return f22_cols


### f21: acc, rotation, mag
def prepare_f21_fixed_sensors_col(isHeaders):
    if isHeaders:
        f21_cols = acc_headers  + mag_headers + rotation_headers
    else:
        f21_cols = range(acc_index_begin,acc_index_end+1)+ range(pressure_index_begin,pressure_index_end +1) + range(mag_index_begin,mag_index_end+1)
    return f21_cols


def prepare_f3_fixed_sensors_col(isHeaders):
    if isHeaders:
        f3_cols = acc_headers + mag_headers + rotation_headers + mag_headers + pressure_headers
    else:
        f3_cols = range(acc_index_begin,acc_index_end+1) + range(rotation_index_begin, rotation_index_end+1) + range(pressure_index_begin,pressure_index_end +1) + range(mag_index_begin,mag_index_end+1)
    return f3_cols



def calc_nolayer_acc_from_cm(f,x_test,y_test,confusion_matrix):
    for i in range(len(x_test)):
        y_predict = f.predict(x_test[i].reshape(1,-1))
        row_ind = travel_mode.index(y_test[i])
        col_ind = travel_mode.index(y_predict)
        confusion_matrix[row_ind][col_ind] += 1
    acc = calc_acc(confusion_matrix)
    return acc

def calc_layered_procedure_acc_from_cm(f1,f21,f22,f3,x_test,y_test,sensor_selection,confusion_matrix):
    f1_sensor_col_index = prepare_f1_fixed_sensors_col(isHeaders=False)
    f21_sensor_col_index = prepare_f21_fixed_sensors_col(isHeaders = False)
    f22_sensor_col_index = prepare_f22_fixed_sensors_col(isHeaders = False)
    f3_sensor_col_index = prepare_f3_fixed_sensors_col(isHeaders = False)
    for i in range(len(x_test)):
        x = x_test[i]
        #x = x.reshape(1,-1)
        y = y_test[i]
        row_ind = travel_mode.index(y)
        ## begin procedure
        ## process x_f1
        if sensor_selection:
            x_f1 = x[f1_sensor_col_index]
        else:
            x_f1 = x
        y_f1 = f1.predict(x_f1.reshape(1,-1))
        #print y_f1
        if y_f1 == ['wheeled']:
            ## come into f21
            if sensor_selection:
                x_f21 = x[f21_sensor_col_index]
            else:
                x_f21 = x
            y_21 = f21.predict(x_f21.reshape(1,-1))
            #print y_21
            if y_21 == ['indoor']:
                ##come to f3
                if sensor_selection:
                    x_f3 = x[f3_sensor_col_index]
                else:
                    x_f3 = x
                y_predict = f3.predict(x_f3.reshape(1,-1))
            else:
                ##bike
                y_predict = y_21
        else:
            if sensor_selection:
                x_f22 = x[f22_sensor_col_index]
            else:
                x_f22 = x
            y_predict = f22.predict(x_f22.reshape(1,-1))
        col_ind = travel_mode.index(y_predict)
        confusion_matrix[row_ind][col_ind] += 1
    acc = calc_acc(confusion_matrix)
    return acc


def learn_f3(source):
    define_global_variables()
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    if source == 'ip':
        df = df_iphone
    elif source == 'aw':
        df = df_android_winter
    elif source == 'as':
        df = df_android_summer
    elif source =='iaw':
        df = pd.concat([df_iphone, df_android_winter])
    elif source == 'a':
        df = pd.concat([df_android_summer, df_android_winter])
    else:
        df = df_all

    sd = df.as_matrix()
    dataset = sd[:,:-2] #because #2 is the uid
    labels = sd[:,-1]
    ## standardization
    scaler = preprocessing.StandardScaler().fit(dataset.astype(float))
    dataset = scaler.transform(dataset.astype(float))

    new_x, new_y = extractf3data(dataset,labels)

    f3_linearSGD = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    f3_svr_rbf = SVC(kernel='rbf', C=1e3, gamma=0.1)
    f3_svr_lin = SVC(kernel='linear', C=1e3)
    f3_svr_poly = SVC(kernel='poly', C=1e3, degree=2)


    f3_linearSGD_scores = cv.cross_val_score(f3_linearSGD, new_x, new_y, cv=10)
    f3_svr_rbf_scores = cv.cross_val_score(f3_svr_rbf, new_x, new_y, cv=10)
    f3_svr_lin_score = cv.cross_val_score(f3_svr_lin, new_x, new_y, cv=10)
    f3_svr_poly_score = cv.cross_val_score(f3_svr_poly, new_x, new_y, cv=10)

    print ("f3_linearSGD_scores: %4f, f3_svr_rbf_scores:%4f,f3_svr_lin_score:%4f,f3_svr_poly_score:%4f" % f3_linearSGD_scores,f3_svr_rbf_scores,f3_svr_lin_score,f3_svr_poly_score)

def prepare_layered_data(x,y,sensor_selection):
    f1_train = x
    f1_label = processf1Label(y)
    f21_train, f21_label = extractf21data(x,y)
    f22_train, f22_label = extractf22data(x,y)
    f3_train, f3_label = extractf3data(x,y)

    f1_sensor_col_index = prepare_f1_fixed_sensors_col(isHeaders=False)
    f21_sensor_col_index = prepare_f21_fixed_sensors_col(isHeaders = False)
    f22_sensor_col_index = prepare_f22_fixed_sensors_col(isHeaders = False)
    f3_sensor_col_index = prepare_f3_fixed_sensors_col(isHeaders = False)

    if sensor_selection:
        f1_train = [[e[i] for i in f1_sensor_col_index] for e in f1_train]
        f21_train = [[e[i] for i in f21_sensor_col_index] for e in f21_train]
        f22_train = [[e[i] for i in f22_sensor_col_index] for e in f22_train]
        f3_train = [[e[i] for i in f3_sensor_col_index] for e in f3_train]

    return f1_train,f1_label,f21_train, f21_label,f22_train, f22_label,f3_train, f3_label



def four_setting_sync_timeline(source,initial_data_ratio,new_data_size,smallSeg = False):
    define_global_variables()
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    if source == 'ip':
        df = df_iphone
    elif source == 'aw':
        df = df_android_winter
    elif source == 'as':
        df = df_android_summer
    elif source =='iaw':
        df = pd.concat([df_iphone, df_android_winter])
    elif source == 'a':
        df = pd.concat([df_android_summer, df_android_winter])
    else:
        df = df_all

    sd = df.as_matrix()
    dataset = sd[:,:-2] #because #2 is the uid
    labels = sd[:,-1]
    ## standardization
    scaler = preprocessing.StandardScaler().fit(dataset.astype(float))
    dataset = scaler.transform(dataset.astype(float))
    ## split data into initial training set, and the rest
    x_init,x_stock,y_init,y_stock = cv.train_test_split(dataset,labels,test_size = 1-initial_data_ratio,random_state = 42)

    ### procedure:
    # 1. learn a classifier for current segment (4 settings)
    ## s1: use all sensors, without hierarchy
    clf_1 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    ## s2: use all sensors, with hierarchy
    clf_2_1 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_2_21 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_2_22 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_2_3 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    ## s3: sensor selection, prelearnt f
    clf_3_1 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_3_21 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_3_22 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_3_3 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    ## s4: sensor selection, incremental f
    clf_4_1 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_4_21 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_4_22 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_4_3  = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)

    ## s5: sensor selection, incremental f, with all data
    clf_5_1 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_5_21 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_5_22 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    clf_5_3 = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100)
    # prepare data:
    no_layer_x = x_init
    no_layer_y = y_init

    f1_train_ns,f1_label,f21_train_ns, f21_label,f22_train_ns, f22_label,f3_train_ns, f3_label = prepare_layered_data(x_init,y_init,sensor_selection = False)
    f1_train_ss,f1_label,f21_train_ss, f21_label,f22_train_ss, f22_label,f3_train_ss, f3_label = prepare_layered_data(x_init,y_init,sensor_selection = True)

    new_f1_train_ss_5 = f1_train_ss
    new_f1_label_5 = f1_label
    new_f21_train_ss_5 = f21_train_ss
    new_f21_label_5 = f21_label
    new_f22_train_ss_5 = f22_train_ss
    new_f22_label_5 = f22_label
    new_f3_train_ss_5 = f3_train_ss
    new_f3_label_5 = f3_label

    # initial training
    ## setting 1:
    clf_1.fit(no_layer_x,no_layer_y)
    ## setting 2:
    clf_2_1.fit(f1_train_ns,f1_label)
    print(set(f21_label))
    clf_2_21.fit(f21_train_ns,f21_label)
    clf_2_22.fit(f22_train_ns,f22_label)
    clf_2_3.fit(f3_train_ns,f3_label)
    ## setting 3:
    clf_3_1.fit(f1_train_ss,f1_label)
    clf_3_21.fit(f21_train_ss,f21_label)
    clf_3_22.fit(f22_train_ss,f22_label)
    clf_3_3.fit(f3_train_ss,f3_label)
    ## setting 4 (incremental):
    clf_4_1.partial_fit(f1_train_ss,f1_label,classes = ['wheeled','unwheeled'])
    if len(f21_train_ss) >0:
        clf_4_21.partial_fit(f21_train_ss,f21_label,classes = ['indoor','bike'])
    if len(f22_train_ss) >0:
        clf_4_22.partial_fit(f22_train_ss,f22_label,classes = ['walk','jog'])
    if len(f3_train_ss) > 0:
        #clf_4_3.fit(f3_train_ss,f3_label)
        clf_4_3.fit(f3_train_ss,f3_label)
    ## setting 5 (incremental):
    '''

    clf_5_1.fit(f1_train_ss,f1_label)
    if len(f21_train_ss) >0:
        clf_5_21.fit(f21_train_ss,f21_label)
    if len(f22_train_ss) >0:
        clf_5_22.fit(f22_train_ss,f22_label)
    if len(f3_train_ss) > 0:
        clf_5_3.fit(f3_train_ss,f3_label)
    '''
    ## data slice loop: with seg_begin and seg_end index, and n times of update, where n =
    seg_begin = 0
    n = len(x_stock)/new_data_size
    seg_begin = 0
    s1_acc = []
    s2_acc = []
    s3_acc = []
    s4_acc = []
    s5_acc = []
    for i in range(n):
        seg_end = seg_begin + new_data_size

        new_x = x_stock[seg_begin:seg_end,:]
        new_y = y_stock[seg_begin:seg_end]

        #print len(new_x),len(new_y)
        if i == 1899:
            print(new_y)
        new_f1_train_ns,new_f1_label,new_f21_train_ns, new_f21_label,new_f22_train_ns, new_f22_label,new_f3_train_ns, new_f3_label = prepare_layered_data(new_x,new_y,sensor_selection = False)
        new_f1_train_ss,new_f1_label,new_f21_train_ss, new_f21_label,new_f22_train_ss, new_f22_label,new_f3_train_ss, new_f3_label = prepare_layered_data(new_x,new_y,sensor_selection = True)
        #scores:
        #1.
        acc1 = calc_nolayer_acc_from_cm(clf_1,new_x,new_y,confusion_matrix1)
        s1_acc.append(acc1)
        #2.
        acc2 = calc_layered_procedure_acc_from_cm(clf_2_1,clf_2_21,clf_2_22,clf_2_3,new_x,new_y,sensor_selection = False,confusion_matrix = confusion_matrix2)
        s2_acc.append(acc2)
        #3.
        acc3 = calc_layered_procedure_acc_from_cm(clf_3_1,clf_3_21,clf_3_22,clf_3_3,new_x,new_y,sensor_selection = True,confusion_matrix = confusion_matrix3)
        s3_acc.append(acc3)
        #4
        acc4 = calc_layered_procedure_acc_from_cm(clf_4_1,clf_4_21,clf_4_22,clf_4_3,new_x,new_y,sensor_selection = True,confusion_matrix = confusion_matrix4)
        s4_acc.append(acc4)
        #5
        #acc5 = calc_layered_procedure_acc_from_cm(clf_5_1,clf_5_21,clf_5_22,clf_5_3,new_x,new_y,sensor_selection = True,confusion_matrix = confusion_matrix5)
        #s5_acc.append(acc5)
        ## update the clf_4
        clf_4_1.partial_fit(new_f1_train_ss,new_f1_label)
        if len(new_f21_train_ss) > 0:
            clf_4_21.partial_fit(new_f21_train_ss,new_f21_label)
        if len(new_f22_train_ss) > 0:
            clf_4_22.partial_fit(new_f22_train_ss,new_f22_label)
        if len(new_f3_train_ss) > 0:
            new_f3_train_ss_5 = np.vstack((new_f3_train_ss_5,new_f3_train_ss))
            new_f3_label_5 = new_f3_label_5 + new_f3_label
            clf_4_3.fit(new_f3_train_ss_5,new_f3_label_5)
            #clf_4_3.partial_fit(new_f3_train_ss,new_f3_label)

        ## update the clf_5
        '''
        new_f1_train_ss_5 = np.vstack((new_f1_train_ss_5,new_f1_train_ss))
        new_f1_label_5 = new_f1_label_5 + new_f1_label
        clf_5_1.fit(new_f1_train_ss_5,new_f1_label_5)
        if len(new_f21_train_ss) > 0:
            new_f21_train_ss_5 = np.vstack((new_f21_train_ss_5,new_f21_train_ss))
            new_f21_label_5 = new_f21_label_5 + new_f21_label
            clf_5_21.fit(new_f21_train_ss_5,new_f21_label_5)
        if len(new_f22_train_ss) > 0:
            new_f22_train_ss_5 = np.vstack((new_f22_train_ss_5,new_f22_train_ss))
            new_f22_label_5 = new_f22_label_5 + new_f22_label
            clf_5_22.fit(new_f22_train_ss_5,new_f22_label_5)
        if len(new_f3_train_ss) > 0:
            new_f3_train_ss_5 = np.vstack((new_f3_train_ss_5,new_f3_train_ss))
            new_f3_label_5 = new_f3_label_5 + new_f3_label
            clf_5_3.fit(new_f3_train_ss_5,new_f3_label_5)
        '''
        seg_begin = seg_end

    return s1_acc,s2_acc,s3_acc,s4_acc#,s5_acc

def plot_layered_learning_results(a1,a2,a3,a4):
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.plot(range(1,len(a1)+1),a4,'b-',label = 'Incremental Classifier - with Sensor Selection',linewidth = 3)
    #plt.plot(range(1,len(a1)+1),a5,'k-',label = 'Incremental Classifier - with Sensor Selection, new hierarchy',linewidth = 3)
    plt.plot(range(1,len(a1)+1),a3,'g-.',label = "Prelearned Classifier - with Sensor Selection",linewidth = 3)
    plt.plot(range(1,len(a1)+1),a1,'r-.',label = "All Sensors, No Hierarchy",linewidth = 3)
    plt.plot(range(1,len(a1)+1),a2,'c-.',label = "All Sensors, with Hierarchy",linewidth = 3)
    plt.legend(loc=0,fontsize = 18)
    plt.xlabel('Iteration', fontsize=18, color='black')

    plt.ylabel('Accuracy', fontsize=18, color='black')
    ##plt.title("Classification Results (iPhone Data) - Online Updating Classifier VS Prelearned Classifier",fontsize = 20)
    plt.show()

if __name__ == '__main__':
    #learn_f3('ip')
    s1_acc,s2_acc,s3_acc,s4_acc = four_setting_sync_timeline('ip',0.5,5,smallSeg = False)
    plot_layered_learning_results(s1_acc,s2_acc,s3_acc,s4_acc)
    results = [s1_acc,s2_acc,s3_acc,s4_acc]
    np.savetxt("layered_learning_results.txt", results, fmt='%1.3f', delimiter=',', newline='\n')
