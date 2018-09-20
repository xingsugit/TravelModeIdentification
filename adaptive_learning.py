import time
from collections import Counter
import timeit
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import pandas as pd
import math
import similarity_calc


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
    global uid_header,label_headeri, uid_index, label_index
    uid_header = 'UID'
    label_header = 'class'
    uid_index = 162
    label_index = 163
    global indoor_mode,wheeled_mode, unwheeled_mode,outdoor_mode
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
    android_winter_data_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
    android_summer_data_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
    iphone_data_file = '~/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'

    df_android_winter = pd.read_csv(android_winter_data_file)
    df_android_summer = pd.read_csv(android_summer_data_file)
    df_iphone = pd.read_csv(iphone_data_file)

    ### select some uids in df_iphone, to see whether the diversity is the reason for online learning no converge
    #df_iphone = df_iphone.loc[df_iphone['UID'].isin([5])]

    df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
    uids = sorted(pd.unique(df_all['UID']))
    return df_all,df_android_winter,df_android_summer,df_iphone


## get similarity rank from simiarity files
def get_similarity_rank(similarity_file):
    similarity_score_xi07 = pd.read_csv(similarity_file,index_col = 0)
    similarity_score = similarity_score_xi07.as_matrix()
    for i in range(len(similarity_score)):
        for j in range(len(similarity_score)):
            if math.isnan(similarity_score[i][j]):
                similarity_score[i][j] = 0
            else:
                similarity_score[i][j] = round(similarity_score[i][j],3)
    ### normalize to 0-1
    max_score = max([max(e) for e in similarity_score])
    for i in range(len(similarity_score)):
        for j in range(len(similarity_score)):
            if i == j:
                similarity_score[i][j] = 0 ##self scoring to lowest
            else:
                similarity_score[i][j] = round(similarity_score[i][j]/(max_score),3)

    similarity_rank = []
    for i in range(len(similarity_score)):
        score_orders = sorted(similarity_score[i])
        #rank =[np.where(score_orders == x)[0][0] for x in similarity_score[0]]
        similarity_rank.append([np.where(score_orders == x)[0][0] for x in similarity_score[i]])
    return similarity_rank

## get the uids that are of high similarity according to the similarity score by similarity_rank_m
def get_top_uids(uid,similarity_rank_m,comparision):
    print('current uid is:'+str(uid))
    similarity_rank = similarity_rank_m[np.where(np.array(uids) == uid)[0][0]]
    top_ranked = sorted(similarity_rank,reverse=True)[:comparision]
    #print top_ranked
    top_similarity_uids_index = [np.where(similarity_rank==x)[0][0] for x in top_ranked]
    results = [uids[i] for i in top_similarity_uids_index]
    return results

## sample data from the dataset.
## volume is the volume of the cm_d.
## modes, uids are the filters.
def sample_data(modes,uids,volume,df):
    uids = [int(x) for x in uids]
    m_d = pd.DataFrame(columns= df.columns)
    for m in modes:
        df_m = df.loc[(df['UID'].isin(uids))& (df['class'] == m)]
        if volume > len(df_m):
            size = len(df_m)
        else:
            size = volume
        rows = np.random.choice(df_m.index, size)
        df_m_vl = df_m.ix[rows]
        m_d = m_d.append(df_m_vl)
    return m_d

## sample data (random): each mode randomly pick another uid data of the same mode
def sample_data_uid_num_limited(uid,modes,uids,topn,df):
    df = df.loc[df['UID'] != uid]
    uids = [int(x) for x in uids]
    m_d = pd.DataFrame(columns = df.columns)
    print('random')
    for m in modes:
        unique_uids = df.loc[df['class'] == m]['UID'].unique()
        if len(unique_uids) < topn:
            choices = len(unique_uids)
        else:
            choices = topn
        pick_uid = np.random.choice(unique_uids,choices,replace = False)
        print(m)
        print(pick_uid)
        pick_uid_m_d = df.loc[df['UID'].isin(pick_uid) &(df['class'] == m)]
        m_d = m_d.append(pick_uid_m_d)
    return m_d

## get weight for xi with respect to target data: tample_weight(xi,tg,C):
def get_sample_weight(xi,tg,C):
    distance_matrix = np.subtract(np.array(tg),np.array(xi))
    #print(distance_matrix)
    dis = np.linalg.norm(distance_matrix,axis = 1)
    #print(dis)
    vfunc = np.vectorize(lambda x:1/(x+C))
    z = sum(vfunc(dis))
    return z

def get_normalizer(d,C):
    summ = 0
    for xi in d:
        summ += get_sample_weight(xi,d,C)
    normalizer = 1/summ
    return normalizer


## return: cm,cm_d,volume (volume is the mean size of each class in cm_d
## cm is the common modes, cm_d is the data of the common modes from the top 3 uids
def get_common_data(uid,top_n_uids,df):
    top_n_uids = [int(x) for x in top_n_uids]
    #df['UID'].astype('str')
    cm = pd.unique(df.loc[df['UID'] == int(uid),'class'])
    cm_similarity = list()
    for u in top_n_uids:
        mode = np.unique(df.loc[df['UID'] == u,'class'])
        cm_similarity = list(set(cm_similarity).union(set(mode)))
    cm = list(set(cm).intersection(set(cm_similarity)))
    print("common mode:")
    print(cm)
    cm_d = df.loc[(df['UID'].isin(top_n_uids))& (df['class'].isin(cm))]
    volume = int(np.mean(pd.value_counts(cm_d[['class']])))
    return cm,cm_d,volume

def get_theoretical_weights(train_x,train_y,test_x,test_y,C):
    training_samples = np.array(train_x,dtype = np.float64)
    training_labels = np.array(train_y,dtype = str)
    test_samples = np.array(test_x,dtype = np.float64)
    test_labels = np.array(test_y,dtype = str)
    weight = []
    ## normalizer:
    nt = get_normalizer(test_samples,C)
    nd = get_normalizer(training_samples,C)
    for i, xi in enumerate(training_samples):
        m = training_labels[i]
        tg = np.take(test_samples,np.where(np.array(test_labels) == m)[0],axis = 0)
        d  = np.take(training_samples,np.where(np.array(training_labels) == m)[0],axis = 0)
        if len(tg) == 0:
            weight_i = -1
        else:
            weight_i_t = nt * get_sample_weight(xi,tg,C)
            weight_i_d = nd * get_sample_weight(xi,d,C)
            weight_i = weight_i_t/weight_i_d
        weight.append(weight_i)
    w0_avg = np.mean([e for e in weight if e > 0])
    weight = [w0_avg if weight[i] < 0 else weight[i] for i in range(len(weight))]
    weight = [wi/np.mean(weight)  for wi in weight]
    return weight

def get_theoretical_weights_class_unknown(train_x,train_y,test_x,test_y,C):
    training_samples = np.array(train_x,dtype = np.float64)
    training_labels = np.array(train_y,dtype = str)
    test_samples = np.array(test_x,dtype = np.float64)
    test_labels = np.array(test_y,dtype = str)
    weight = []
    nt = get_normalizer(test_samples,C)
    nd = get_normalizer(training_samples,C)
    for i, xi in enumerate(training_samples):
        weight_i_t = nt * get_sample_weight(xi,test_samples,C)
        weight_i_d = nd * get_sample_weight(xi,training_samples,C)
        weight_i = weight_i_t/weight_i_d
        weight.append(weight_i)
    w0_avg = np.mean([e for e in weight if e > 0])
    weight = [w0_avg if weight[i] < 0 else weight[i] for i in range(len(weight))]
    weight = [wi/np.mean(weight) for wi in weight]
    return weight


## get weight vector for train_x, with respect of test_x,
def get_weights_obo(train_x,train_y,test_x,test_y,C):
    training_samples = np.array(train_x,dtype = np.float64)
    training_labels = np.array(train_y,dtype = str)
    test_samples = np.array(test_x,dtype = np.float64)
    test_labels = np.array(test_y,dtype = str)
    weight = []
    for i,xi in enumerate(training_samples):
        m = training_labels[i]
        tg = np.take(test_samples,np.where(np.array(test_labels)==m)[0],axis = 0)
        if len(tg) == 0:
            weight_i = -1
        else:
            weight_i = get_sample_weight(xi,tg,C)
        weight.append(weight_i)
    w0_avg = np.mean([e for e in weight if e > 0])
    weight = [w0_avg if weight[i] < 0 else weight[i] for i in range(len(weight))]
    weight = [wi/np.mean(weight)  for wi in weight]
    return weight

## sampling with weight:
def reject_sampling(train_x,train_y,weight,threathold = 1):
    new_train_x = np.array([])
    new_train_y = np.array([])
    for i in range(len(train_x)):
        if weight[i] > threathold:
            new_train_x = np.append(new_train_x,[train_x[i]])
            new_train_y = np.append(new_train_y,train_y[i])
    return train_x,train_y


### performance comparison: weighted samples VS others
def test_weighting_method_single_uid(uid,top_n_uids,C,df,overall_comp):
    rest_uids = list(set(uids) - set(top_n_uids) - set([uid]) )
    cm,cm_d, volume = get_common_data(uid,top_n_uids,df)#cm is the common modes, cm_d is the data of the common modes from the top 3 uids
    print(cm)
    if len(cm) > 1:
        training_df = cm_d#.append(pm_d)
    else:
        pm = list(set(df['class']) - set(cm))   ## the rest modes
        pm_d = sample_data(pm,uids,volume,df)
        training_df = cm_d.append(pm_d)
        cm += pm
    ## random samples of same cm(common modes) but from rest_uids. (cm is already processed to be guaranteed to be at least 2)
    rm_d = sample_data(cm,rest_uids,volume,df)
    if overall_comp:
        control_df = df.loc[df['UID'] != int(uid)]
    else:
        control_df = rm_d

    ## for training data:
    train = training_df.as_matrix()
    train_x = train[:,:-2] #because #2 is the uid
    train_y = train[:,-1]

    ## for control data:
    control = control_df.as_matrix()
    control_x = control[:,:-2]
    control_y = control[:,-1]

    ## for test data
    test_df = df.loc[df['UID'] == int(uid)]
    test = test_df.as_matrix()
    test_x = test[:,:-2]
    test_y = test[:,-1]

    ##

    ### normalize (is there a better way to normalize?):
    #scaler = preprocessing.standardscaler().fit(df.as_matrix()[:,:-2].astype(float))
    ## try to use the scaler of single person's (the training data)
    scaler = preprocessing.StandardScaler().fit(train_x.astype(float))
    train_x_scaled = scaler.transform(train_x.astype(float))
    control_x_scaled = scaler.transform(control_x.astype(float))
    test_x_scaled = scaler.transform(test_x.astype(float))

    ##### SGD data from top 3 similar uids:
    random_state = np.random.RandomState(0)
    clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,class_weight = 'balanced')
    clf.fit(train_x_scaled, train_y)
    tops_y_predict = clf.predict(test_x_scaled)
    tops_acc = sum(tops_y_predict == test_y)*1.0/len(test_y)

    #### SGD data from randomly picked data from the rest uids
    random_state = np.random.RandomState(0)
    clf_rest_uid = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,class_weight = 'balanced')
    clf_rest_uid.fit(control_x_scaled, control_y)
    control_y_predict = clf_rest.predict(test_x_scaled)
    rest_uid_acc = sum(control_y_predict == test_y)*1.0/len(test_y)

    #weight_x_scaled = scaler.transform(weight_x.astype(float))
    weight = get_weights_obo(train_x_scaled,train_y,test_x_scaled,test_y,C)
    random_state = np.random.RandomState(0)
    clf_weighted = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,class_weight = 'balanced')
    clf_weighted.fit(train_x_scaled,train_y,sample_weight = weight)
    weighted_y_predict = clf_weighted.predict(test_x_scaled)
    weighted_acc = sum(weighted_y_predict == test_y)*1.0/len(test_y)


    #weighted_top_n_score = clf_weighted.score(test_x_scaled,test_y)
    #return top_s_score,random_score, weighted_top_n_score
    return tops_acc,rest_uid_acc,weighted_acc


### test the performance of all the uids
def weighted_sampling_performance_test(merge_minor,C,topn):
    global uids
    define_global_variables()
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    if merge_minor:
        similarity_score_file = "../results/user_similarity_scores03_mergeminors.csv"
        df_all.loc[df_all['UID'] == 4,['UID']] = 1
        df_all.loc[df_all['UID'] == 6,['UID']] = 2
        df_all.loc[df_all['UID'] == 7,['UID']] = 3
    else:
        similarity_score_file = '../results/user_similarity_scores03.csv'
    uids = np.array(sorted(pd.unique(df_all['UID'])))
    rank_m = get_similarity_rank(similarity_score_file)
    scores = []
    current_count = 0
    err_so_far = [[0,0,0]]
    sample_size_so_far = [0]
    for uid in uids:
        top_n_uids = get_top_uids(uid, rank_m,topn)
        current_count += len(df_all.loc[df_all['UID'] == int(uid)])
        sample_size_so_far.append(sample_size_so_far[-1] + current_count)
        acc = test_weighting_method_single_uid(uid,top_n_uids,C,df_all,overall_comp = True)
        err_so_far.append([e[1] + current_count * (1-e[0]) for e in zip(acc,err_so_far[-1])])
        scores.append(acc)
    cummulative_error_rate = [[e/x for e in a] for a,x in zip(err_so_far[1:],sample_size_so_far[1:])]
    return scores,cummulative_error_rate


def plot_comparision_results(merge_minor,C,topn):
    scores,cum_err = weighted_sampling_performance_test(merge_minor, C,topn)
    top_s_score = [e[0] for e in scores]
    random_score = [e[1] for e in scores]
    weighted_top_n_score = [e[2] for e in scores]
    top_s_err = [e[0] for e in cum_err]
    random_s_err = [e[1] for e in cum_err]
    weighted_top_n_err = [e[2] for e in cum_err]
    #best_exhaustive = [0.889, 0.996, 0.985, 0.70, 0.847, 0.977, 0.977, 0.967, 0.971, 0.563, 0.908, 0.784, 0.813, 0.828, 0.58]
    matplotlib.rcParams['figure.figsize'] = (150.0, 10.0)
    panel = plt.figure(1, figsize=(20,10))
    plt.tick_params(axis='both', which='major', labelsize=16)
    #plt.plot(range(1,len(vanilla_on_rest)+1),vanilla_on_rest,'.b-',label = 'Learn from the rest, plain data',linewidth = 3)
    #plt.plot(range(1,len(vanilla_on_rest)+1),weighted_on_rest,'*g-.',label = "Learn with weight",linewidth = 3)
    plt.plot(range(1,len(scores)+1),top_s_score,'.r-',label = 'Learn from top 3 similar uids',linewidth = 3)
    plt.plot(range(1,len(scores)+1),random_score,'.b-',label = 'Learn from 3 random uids',linewidth = 3)
    plt.plot(range(1,len(scores)+1),weighted_top_n_score,'.c-',label = 'Learn from weighted top 3 similar uids',linewidth = 3)
    #plt.plot(range(1,len(vanilla_on_rest)+1),best_exhaustive,'.y-',label = 'Best results from exhaustive search',linewidth = 3)
    plt.title('Performance Compare: Learning from plain and weighted data',fontsize = 20)
    plt.legend(loc=0,fontsize = 18)
    plt.xlabel('UIDS', fontsize=18, color='black')
    plt.ylabel('Score', fontsize=18, color='black')
    plt.savefig("acc_uid_weighted_comparison.png")
    #panel = plt.figure(2, figsize=(20,10))
    plt.clf()
    plt.tick_params(axis='both', which='major', labelsize=16)
    #plt.plot(range(1,len(vanilla_on_rest)+1),vanilla_on_rest,'.b-',label = 'Learn from the rest, plain data',linewidth = 3)
    #plt.plot(range(1,len(vanilla_on_rest)+1),weighted_on_rest,'*g-.',label = "Learn with weight",linewidth = 3)
    plt.plot(range(1,len(scores)+1),top_s_err,'.r-',label = 'Learn from top 3 similar uids',linewidth = 3)
    plt.plot(range(1,len(scores)+1),random_s_err,'.b-',label = 'Learn from 3 random uids',linewidth = 3)
    plt.plot(range(1,len(scores)+1),weighted_top_n_err,'.c-',label = 'Learn from weighted top 3 similar uids',linewidth = 3)
    #plt.plot(range(1,len(vanilla_on_rest)+1),best_exhaustive,'.y-',label = 'Best results from exhaustive search',linewidth = 3)
    plt.title('Accumulated Errors: Learning from plain and weighted data',fontsize = 20)
    plt.legend(loc=0,fontsize = 18)
    plt.xlabel('UIDS', fontsize=18, color='black')
    plt.ylabel('Score', fontsize=18, color='black')
    plt.savefig("Accumulated_errors_weighted_comparision.png")





## this function returns the data to learn from with high similarity scores of each mode
## Each uid has a simialrity score for each travel mode
## The function takes topn uids with high similarity of each mode, and extract the data
def get_similar_data_from_mode_uid_pair_rank(uid,topn,df,xi,T,ns_similarity_matrix,size_control = False):
    uids = sorted(pd.unique(df['UID']))
    similar_data = pd.DataFrame(columns= df.columns)
    non_similar_data = pd.DataFrame(columns = df.columns)
    xi_text = str(int(xi*10))
    similarity_score_file = "../data/usr_"+ str(uid) + "_mode_similarity_xi0"+xi_text+".csv"
    print(similarity_score_file)
    similarity_score = pd.read_csv(similarity_score_file)
    modes_for_current_uids = df.loc[df['UID'] == uid]['class'].unique()
    for mode in modes_for_current_uids:
        print(mode)
        if mode in ['walk'] and uid == 31:
            print('add one top')
            top_similar_uids_for_current_mode  = similarity_score.loc[similarity_score[mode].nlargest(topn+1).index]['usr']
            #top_similar_uids_for_current_mode = similarity_calc.high_similar_uids_in_current_mode(df,uid,T,mode,ns_similarity_matrix,topn+1)
        else:
            top_similar_uids_for_current_mode = similarity_score.loc[similarity_score[mode].nlargest(topn).index]['usr']
            #top_similar_uids_for_current_mode = similarity_calc.high_similar_uids_in_current_mode(df,uid,T,mode,ns_similarity_matrix,topn)
        print(top_similar_uids_for_current_mode)
        similar_data_for_current_mode = df.loc[df['UID'].isin(top_similar_uids_for_current_mode) & (df['class'] == mode)]
        #print(len(similar_data_for_current_mode))
        similar_data = similar_data.append(similar_data_for_current_mode)
        volume = len(similar_data_for_current_mode)
        rest_uids = list(set(df['UID'].unique()) - set(top_similar_uids_for_current_mode) - set([uid]))
        ## get compensate data: df_m, of which the mode is current mode and the uids are the rest uids
        df_m = df.loc[df['UID'].isin(rest_uids) & (df['class'] == mode)]
        if len(df_m) == 0:
            print("Lack of data for current mode in the rest uids, put same similarity uid data")
            #df_m = df.loc[df['UID'].isin(list(set(df['UID'].unique()) - set([uid]))) & (df['class'] == mode)]
            #change: 2017.01.16 from above to down
            df_m = similar_data_for_current_mode
        if size_control:
            if volume > len(df_m):
                size = len(df_m)
            else:
                size = volume
            rows = np.random.choice(df_m.index, size)
            df_m = df_m.ix[rows]
        non_similar_data = non_similar_data.append(df_m)
    return similar_data,non_similar_data

### calculate the similarity score
#def calc_similarity_score(uid,xi):




### this function print the information of prediction details of each mode. It compares the truth and the predict.
def classification_info(test_y, predict_y):
    #unique_facts, counts_facts = np.unique(test_y, return_counts=True)
    facts_dict = Counter(test_y)#dict(zip(unique_facts, counts_facts))
    #unique_predict, counts_predict = np.unique(predict_y,return_counts = True)
    predict_dict = Counter(predict_y)#dict(zip(unique_predict,counts_predict))
    ratio = ((key,1.0*predict_dict[key]/facts_dict[key]) for key in facts_dict.keys())
    print("predict/facts summary:")
    print(dict(ratio))
    print("The correct cases:")
    correct =dict({(key,0) for key in facts_dict.keys()})
    for i in range(len(predict_y)):
        if predict_y[i] == test_y[i]:
            correct[test_y[i]] += 1
    for key in correct.keys():
        correct[key] = 1.0 * correct[key]/facts_dict[key]
    print(correct)

def get_indoor_outdoor_label_list(orig_label_list):
    indoor_outdoor_label_list = ['indoor' if e in ['bus','car','subway'] else 'outdoor' for e in orig_label_list]
    return indoor_outdoor_label_list



def get_wheeled_unwheeled_label_list(orig_label_list):
    wheeled_unwheeled_label_list = ['unwheeled' if e in ['walk','jog'] else 'wheeled' for e in orig_label_list]
    return wheeled_unwheeled_label_list



### the hierarchical structure of learning
def hierarchical_learning_withSGD(train_x,train_y,wheeled_first):
    single_class = ['','','',''] ## wheeled class, unwheeled class, outdoor class, indoor class
    if wheeled_first:
        meta_labels = get_wheeled_unwheeled_label_list(train_y)
        random_state = np.random.RandomState(0)
        meta_layer_clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
        meta_layer_clf.fit(train_x, meta_labels) ##generate_results of wheeled and unwheeled
        ## inside wheeled:
        wheeled_x = [train_x[i] for i in range(len(train_y)) if train_y[i] in ['bike','bus','subway','car'] ]
        wheeled_y = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['bike','bus','subway','car'] ]
        if len(set(wheeled_y)) > 1:
            wheeled_clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
            wheeled_clf.fit(wheeled_x,wheeled_y)
        else:
            wheeled_clf = None
            single_class[0] = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['bike','bus','subway','car']][0]
        ## inside unwheeled:
        unwheeled_x = [train_x[i] for i in range(len(train_y)) if train_y[i] in ['walk','jog']]
        unwheeled_y = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['walk','jog']]
        if len(set(unwheeled_y)) > 1:
            unwheeled_clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
            unwheeled_clf.fit(unwheeled_x,unwheeled_y)
        else:
            unwheeled_clf = None
            single_class[1] = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['walk','jog']][0]
        return meta_layer_clf,wheeled_clf,unwheeled_clf,single_class
    else:
        meta_labels = get_indoor_outdoor_label_list(train_y)
        random_state = np.random.RandomState(0)
        meta_layer_clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
        meta_layer_clf.fit(train_x, meta_labels) ##generate_results of wheeled and unwheeled
        ## inside indoor:
        indoor_x = [train_x[i] for i in range(len(train_y)) if train_y[i] in ['car','subway','bus']]
        indoor_y = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['car','subway','bus']]
        if len(set(indoor_y)) > 1:
            indoor_clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
            indoor_clf.fit(indoor_x,indoor_y)
        else:
            indoor_clf = None
            single_class[3] = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['car','bus','subway']][0]
        ## inside outdoor
        outdoor_x = [train_x[i] for i in range(len(train_y)) if train_y[i] in ['bike','walk','jog']]
        outdoor_y = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['bike','walk','jog']]
        if len(set(outdoor_y)) > 1:
            outdoor_clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
            outdoor_clf.fit(outdoor_x,outdoor_y)
        else:
            outdoor_clf = None
            single_class[2] = [train_y[i] for i in range(len(train_y)) if train_y[i] in ['bike','jog','walk']][0]
        return meta_layer_clf,outdoor_clf,indoor_clf,sinlge_class

## this function automatically generate


## the hierarchical structure model to predict
def hierarchical_structure_predict(meta_layer_clf, clf_1,clf_2, test_x,test_y,single_class):
    predict_y = []
    ##single_class: wheeled, unwheeled, outdoor, indoor
    for x in test_x:
        meta_category = meta_layer_clf.predict(x.reshape(1,-1))
        if meta_category =='wheeled':
            ## classifier 1
            if clf_1 != None:
                predict = clf_1.predict(x.reshape(1,-1))
            else:
                predict = [single_class[0]]
        elif meta_category == 'unwheeled':
            if clf_2 != None:
                predict = clf_2.predict(x.reshape(1,-1))
            else:
                predict = [single_class[1]]
        elif meta_category == 'outdoor':
            if clf_1 != None:
                predict = clf_1.predict(x.reshape(1,-1))
            else:
                predict = [single_class[2]]
        else:
            if clf_2 != None:
                predict = clf_2.predict(x.reshape(1,-1))
            else:
                predict = [single_class[3]]
        predict_y.append(predict)

    return predict_y


### wf is: wheeled_first, value: True or False. If True, wheeled/unwheeled will be classified first. If False, then indoor/outdoor
### method: string, used for print classification status. Example: 'Top Similarity'
def learn_predict(train_x,train_y,wf,test_x,test_y,method):
    meta_layer_clf,clf_1,clf_2,single_class = hierarchical_learning_withSGD(train_x,train_y,wheeled_first = wf)
    y_predict = hierarchical_structure_predict(meta_layer_clf,clf_1,clf_2,test_x,test_y,single_class)
    acc = sum([1 if y_predict[i] == test_y[i] else 0 for i in range(len(y_predict))])*1.0/len(test_y)
    print("\033[1;31;40m {} accuracy is {}\033[0;32;40m".format(method, acc))
    classification_info(test_y,y_predict)





### test similarity and weighted learning method with hierarchical learning structure
def test_usr_mode_similarity_hierarchical(uid,topn,C,df,cz = False):
    print('Current uid is:',uid)
    #get data of similarity, as well as non_similar_data.
    top_n_learning_data, non_similar_data = get_similar_data_from_mode_uid_pair_rank(uid,topn,df,size_control = cz)
    cm = pd.unique(df.loc[df['UID'] == int(uid),'class'])
    uids = sorted(pd.unique(df['UID']))
    rest_uids = list(set(uids) - set([uid]))

    ## random samples of the same mode (n users)
    rm_d_topn = sample_data_uid_num_limited(uid,cm,rest_uids,topn,df)


    ## for training data:
    train = top_n_learning_data.as_matrix()
    train_x = train[:,:-2] #because #2 is the uid
    train_y = train[:,-1]

    print('Top n similar learning data summary:')
    print(Counter(train_y))
    ## for control data:
    control = non_similar_data.as_matrix()
    control_x = control[:,:-2]
    control_y = control[:,-1]
    print('control_y unique values:')
    print(set(control_y))
    print('non similar learning data summary:')
    print(Counter(control_y))


    ## for random data:
    random_d = rm_d_topn.as_matrix()
    random_x = random_d[:,:-2]
    random_y = random_d[:,-1]


    ## for test data
    test_df = df.loc[df['UID'] == int(uid)]
    test = test_df.as_matrix()
    test_x = test[:,:-2]
    test_y = test[:,-1]
    print('Test data summary:')
    print(Counter(test_y))
    ### normalize (is there a better way to normalize?):
    print('************************************************************************')
    scaler = preprocessing.StandardScaler().fit(df.as_matrix()[:,:-2].astype(float))
    train_x_scaled = scaler.transform(train_x.astype(float))
    control_x_scaled = scaler.transform(control_x.astype(float))
    test_x_scaled = scaler.transform(test_x.astype(float))
    random_x_scaled = scaler.transform(random_x.astype(float))

    #### topn data
    learn_predict(train_x_scaled,train_y,True,test_x_scaled,test_y,method = 'Top Similarity')

    #### rest_uid data with size control
    learn_predict(control_x_scaled,control_y,True,test_x_scaled,test_y,method = 'Rest UID')

    #### weighted topn

    #### weighted rest_uid

    #### random
    learn_predict(random_x_scaled,random_y,True,test_x_scaled,test_y,method = 'Random Data')
    #### selective random



### the top uids are get via each mode.
def test_usr_mode_similarity(uid,topn,C,df,xi,ratio,ns_similarity_matrix,cz = False):
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
    top_n_learning_data, non_similar_data = get_similar_data_from_mode_uid_pair_rank(uid,topn,df,xi,train_x_addin,ns_similarity_matrix,size_control = cz)
    cm = pd.unique(df.loc[df['UID'] == int(uid),'class'])
    uids = sorted(pd.unique(df['UID']))
    rest_uids = list(set(uids) - set([uid]))

    ## random samples of the same mode (n users)
    rm_d_topn = sample_data_uid_num_limited(uid,cm,rest_uids,topn,df)


    ## for training data:
    train = top_n_learning_data.as_matrix()
    train_x = train[:,:-2] #because #2 is the uid
    train_y = train[:,-1]

    print('Top n similar learning data summary:')
    print(Counter(train_y))
    ## for control data:
    control = non_similar_data.as_matrix()
    control_x = control[:,:-2]
    control_y = control[:,-1]
    print('control_y unique values:')
    print(set(control_y))
    print('non similar learning data summary:')
    print(Counter(control_y))


    ## for random data:
    random_d = rm_d_topn.as_matrix()
    random_x = random_d[:,:-2]
    random_y = random_d[:,-1]


    ### normalize (is there a better way to normalize?):
    print('************************************************************************')
    train_x_scaled = scaler.transform(train_x.astype(float))
    control_x_scaled = scaler.transform(control_x.astype(float))
    random_x_scaled = scaler.transform(random_x.astype(float))
    ##### Here we split test data into two different sets: one part uses as labeled data, for sample reweigting. The other part is for testing.
    ### the test_size in train_test_split is correspond to train_x_addin (which is the ratio in T we use for transfer learning)
    weight = get_theoretical_weights(train_x_scaled,train_y,train_x_addin,train_y_addin,C)
    weight_class_unknown = get_theoretical_weights_class_unknown(train_x_scaled,train_y,train_x_addin,train_y_addin,C)
    train_y = np.append(train_y, train_y_addin,0)
    train_x_scaled = np.append(train_x_scaled,train_x_addin,0)
    max_w_s = max(weight) + 1
    weight += [max_w_s for i in range(len(train_y_addin))]
    max_w_s_class_unknown = max(weight_class_unknown)
    weight_class_unknown +=[max_w_s_class_unknown for i in range(len(train_y_addin))]
    ##### SGD data from top 3 similar uids:
    random_state = np.random.RandomState(0)
    clf = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state)#,class_weight = 'balanced')
    clf.fit(train_x_scaled, train_y)
    tops_y_predict = clf.predict(test_x_scaled)
    tops_acc = sum(tops_y_predict == test_y)*1.0/len(test_y)
    #print("\033[1;31;40m The #%d sensor doesn't exist in file: %s\033[0;32;40m"
    print("\033[1;31;40m top similar data accuracy: %f\033[0;32;40m" % tops_acc)
    classification_info(test_y,tops_y_predict)
    #top_s_score = clf.score(test_x_scaled,test_y)
    #print('top similar data score: %f' % top_s_score)

    #### SGD data from randomly picked from the rest uids
    weight_control = get_theoretical_weights(control_x_scaled,control_y,train_x_addin,train_y_addin,C)
    control_y = np.append(control_y, train_y_addin,0)
    control_x_scaled = np.append(control_x_scaled,train_x_addin,0)
    max_w_r = max(weight_control) + 1
    weight_control += [max_w_r for i in range(len(train_y_addin))]
    #### finish prepare the control size and weight
    random_state = np.random.RandomState(0)
    clf_control= SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state)#,class_weight = 'balanced')

    clf_control.fit(control_x_scaled, control_y)
    control_y_predict = clf_control.predict(test_x_scaled)
    rest_uid_acc = sum(control_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m rest_uid accuracy: %f\033[1;32;40m " % rest_uid_acc)
    classification_info(test_y,control_y_predict)
    #rest_uid_score = clf_control.score(test_x_scaled,test_y)
    #print('rest uid score: %f' % rest_uid_score)

    #### SGD for weighted similarity data
    #weight_x_scaled = scaler.transform(weight_x.astype(float))
    #weight = get_weights_obo(train_x_scaled,train_y,test_x_scaled,test_y,C)
    random_state = np.random.RandomState(0)
    clf_weighted = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
    clf_weighted.fit(train_x_scaled,train_y,sample_weight = weight)
    weighted_y_predict = clf_weighted.predict(test_x_scaled)
    weighted_acc = sum(weighted_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m weighted top similar data accuracy: %f\033[1;32;40m " % weighted_acc)
    classification_info(test_y,weighted_y_predict)
    #clf_weighted_score = clf_weighted.score(test_x_scaled,test_y)
    #print('weighted top similarity data score: %f' % clf_weighted_score)

    #### SGD for basic transfer learning: kernel mean matching (calculate the weight with whole data set, instead of getting the weight class by class
    random_state = np.random.RandomState(0)
    clf_weighted_class_unknown = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
    clf_weighted_class_unknown.fit(train_x_scaled,train_y,sample_weight = weight_class_unknown)
    weighted_class_unknown_y_predict = clf_weighted_class_unknown.predict(test_x_scaled)
    weighted_class_unknown_acc = sum(weighted_class_unknown_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m weighted (class unknown)  top similar data accuracy: %f\033[1;32;40m " % weighted_class_unknown_acc)
    classification_info(test_y,weighted_class_unknown_y_predict)


    #weight_control = get_weights_obo(control_x_scaled,control_y,test_x_scaled,test_y,C)

    random_state = np.random.RandomState(0)
    clf_rest_uid_weighted = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
    clf_rest_uid_weighted.fit(control_x_scaled,control_y,sample_weight = weight_control)
    rest_uid_weighted_y_predict = clf_rest_uid_weighted.predict(test_x_scaled)
    rest_uid_weighted_acc = sum(rest_uid_weighted_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m weighted rest uid data accuracy: %f\033[1;32;40m " % rest_uid_weighted_acc)
    classification_info(test_y,rest_uid_weighted_y_predict)
    #weighted_rest_uid_score = clf_rest_uid_weighted.score(test_x_scaled,test_y)
    #print('weighted rest uid score: %f' % weighted_rest_uid_score)


    ### for each mode randomly pick 3 uids. then it consists the randomly picked training data
    random_y = np.append(random_y, train_y_addin,0)
    random_x_scaled = np.append(random_x_scaled,train_x_addin,0)
    random_state = np.random.RandomState(0)
    clf_random = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')

    clf_random.fit(random_x_scaled,random_y)
    random_y_predict = clf_random.predict(test_x_scaled)
    random_acc = sum(random_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m random data accuracy: %f\033[1;32;40m " % random_acc)
    classification_info(test_y,random_y_predict)
    #random_uid_score = clf_random.score(test_x_scaled,test_y)
    #print('random uid score: %f' % random_uid_score)

    ###### add in the ratio compare:
    ratio_compare_x = train_x_addin
    ratio_compare_y = train_y_addin
    clf_ratio_compare = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,class_weight = 'balanced')
    clf_ratio_compare.fit(ratio_compare_x, ratio_compare_y)
    ratio_compare_y_predict = clf_ratio_compare.predict(test_x_scaled)
    ratio_compare_acc = sum(ratio_compare_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m T inner data accuracy: %f\033[1;32;40m " % ratio_compare_acc)



    ###reject sampling:
    '''
    resampled_train_x_scaled, resampled_train_y = reject_sampling(train_x_scaled,train_y,weight,threathold = 1)
    random_state = np.random.RandomState(0)
    clf_resample =  SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100,class_weight = 'balanced')
    clf_resample.fit(resampled_train_x_scaled, resampled_train_y)
    resampled_y_predict = clf_resample.predict(test_x_scaled)
    resampled_acc = sum(resampled_y_predict == test_y)*1.0/len(test_y)
    print('\033[1;31;40m resampled accuracy: %f\033[1;32;40m ' % resampled_acc)
    classification_info(test_y,resampled_y_predict)
    #resampled_score = clf_resample.score(test_x_scaled,test_y)
    #print('resampled score: %f' % resampled_score)
    '''
    print('************************************************************************')
    return tops_acc,random_acc,weighted_acc,weighted_class_unknown_acc,rest_uid_acc,rest_uid_weighted_acc,ratio_compare_acc
    #return top_s_score, rest_uid_score,clf_weighted_score,weighted_rest_uid_score,random_uid_score#,resampled_scroe



### How would topn selection affect the learning results?
### Is it related to the data size?
def topn_performance_compare(uid,C,merge_minor = True):
    define_global_variables()
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    #uids = sorted(pd.unique(df_all['UID']))
    #scores = []
    if merge_minor:
        df_all.loc[df_all['UID'] == 4,['UID']] = 1
        df_all.loc[df_all['UID'] == 6,['UID']] = 2
        df_all.loc[df_all['UID'] == 7,['UID']] = 3
    max_uids = 5
    acc = []
    for topn in range(1,max_uids):
        acc.append(test_usr_mode_similarity(uid,topn,C,df_all))
    tops_acc = [e[0] for e in acc]
    random_acc = [e[1] for e in acc]
    weighted_tops_acc = [e[2] for e in acc]
    rest_uid_acc = [e[3] for e in acc]
    rest_uid_weighted_acc = [e[4] for e in acc]
    file_name = '../data/usr_' + str(uid) + '_topn_performance_weighted_vs_random_C1dot2.csv'
    result = pd.DataFrame({'topn':range(1,max_uids),'tops_acc':tops_acc,'random_acc':random_acc,'weighted_tops_acc':weighted_tops_acc,'rest_uid_acc':rest_uid_acc,'weighted_rest_uid_acc':rest_uid_weighted_acc})
    result.to_csv(file_name, sep=',')
    print('finish generating file %s' % file_name)



## The main body of the performance test procedure, it iterates all the uids. For each uid, call function <test_usr_mode_similarity> to test the performance in different learning setting.
def test_performance_of_mode_uid_pair_similarity(topn,C,xi,ratio,merge_minor = True,hierarchy = False):
    global uids
    define_global_variables()
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    ns_similarity_matrix = similarity_calc.calculate_ns_similarity(xi)
    acc = []
    if merge_minor:
        df_all.loc[df_all['UID'] == 4,['UID']] = 1
        df_all.loc[df_all['UID'] == 6,['UID']] = 2
        df_all.loc[df_all['UID'] == 7,['UID']] = 3
    uids = sorted(pd.unique(df_all['UID']))
    uu = [1,2,3,5,8,9,10,21,22,23,31,32,33]
    uu = [10]
    if hierarchy:
        for u in uids:
            test_usr_mode_similarity_hierarchical(u,topn,C,df_all,cz = False)
    else:
        for u in uu:
            acc.append(test_usr_mode_similarity(u,topn,C,df_all,xi,ratio,ns_similarity_matrix,cz=True))
        tops_acc = [e[0] for e in acc]
        random_acc = [e[1] for e in acc]
        weighted_tops_acc = [e[2] for e in acc]
        weighted_class_unknown_tops_acc = [e[3] for e in acc]
        rest_uid_acc = [e[4] for e in acc]
        rest_uid_weighted_acc = [e[5] for e in acc]
        ratio_compare_acc = [e[6] for e in acc]
        xi_text = str(xi*10)
        ratio_text = str(int(ratio*100))
        file_name = '../data/T_add_weight1_usr_mode_ratio_' + ratio_text+'_top_'+ str(topn) +'_performance_weighted_vs_random_C0dot003_xi'+xi_text+'.csv'
        result = pd.DataFrame({'uid':uu,'tops_acc':tops_acc,'random_acc':random_acc,'weighted_tops_acc':weighted_tops_acc,'weighted_class_unknown_tops_acc':weighted_class_unknown_tops_acc,'rest_uid_acc':rest_uid_acc,'weighted_rest_uid_acc':rest_uid_weighted_acc,'ratio_compare_acc':ratio_compare_acc})
        result.to_csv(file_name, sep=',')
        print('finish generating file %s' % file_name)

def parameter_effect(df,uid,topn,xi,C,cz = False,merge_minor = True):
    print('current uid is:',uid)
    #get data of similarity, as well as non_similar_data. uid,topn,df,xi,size_control = False
    top_n_learning_data, non_similar_data = get_similar_data_from_mode_uid_pair_rank(uid,topn,df,xi,size_control = cz)
    cm = pd.unique(df.loc[df['UID'] == int(uid),'class'])
    uids = sorted(pd.unique(df['UID']))

    ## for training data:
    train = top_n_learning_data.as_matrix()
    train_x = train[:,:-2] #because #2 is the uid
    train_y = train[:,-1]

    test_df = df.loc[df['UID'] == int(uid)]
    test = test_df.as_matrix()
    test_x = test[:,:-2]
    test_y = test[:,-1]
    print('Test data summary:')
    print(Counter(test_y))
    ### normalize (is there a better way to normalize?):
    print('************************************************************************')
    scaler = preprocessing.StandardScaler().fit(df.as_matrix()[:,:-2].astype(float))
    train_x_scaled = scaler.transform(train_x.astype(float))
    test_x_scaled = scaler.transform(test_x.astype(float))

    ##### SGD data from top 3 similar uids:
    random_state = np.random.RandomState(0)
    weight = get_theoretical_weights(train_x_scaled,train_y,test_x_scaled,test_y,C)
    random_state = np.random.RandomState(0)
    clf_weighted = SGDClassifier(loss = 'hinge',penalty = 'l2',shuffle = True,n_iter = 100,random_state = random_state,class_weight = 'balanced')
    clf_weighted.fit(train_x_scaled,train_y,sample_weight = weight)
    weighted_y_predict = clf_weighted.predict(test_x_scaled)
    weighted_acc = sum(weighted_y_predict == test_y)*1.0/len(test_y)
    print("\033[1;31;40m weighted top similar data accuracy: %f\033[1;32;40m " % weighted_acc)
    classification_info(test_y,weighted_y_predict)
    return weighted_acc

def paramter_performance_tour(uid,merge_minor = True):
    global uids
    define_global_variables()
    df_all,df_android_winter,df_android_summer,df_iphone = prepareTravelData()
    acc = []
    if merge_minor:
        df_all.loc[df_all['UID'] == 4,['UID']] = 1
        df_all.loc[df_all['UID'] == 6,['UID']] = 2
        df_all.loc[df_all['UID'] == 7,['UID']] = 3
    uids = sorted(pd.unique(df_all['UID']))
    topn_list = [1,2,3,4,5,6,7]
    xi_list = [0.3,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.5]
    C_seed = range(-10,10,1)
    C_list = [math.exp(e) for e in C_seed]
    acc = []
    #xi = 0.1
    topn = 4
    C = 0.003
    for xi in xi_list:
        acc.append(parameter_effect(df_all,uid,topn,xi,C,cz = False))
    print("current uid is:" + str(uid))
    print(acc)

if __name__ == '__main__':
    mergeMinor = True
    #uid = 8
    #paramter_performance_tour(uid)
    ratio = 0.02
    xi = 0.2
    C = 0.003
    topn = 4
    #plot_comparision_results(mergeMinor,C,topn)
    #uid = 33
    #topn_performance_compare(uid,C)
    #for ratio in [0.02,0.03,0.04,0.06,0.07,0.08,0.09,0.1]:
    print(ratio)
    test_performance_of_mode_uid_pair_similarity(topn,C,xi,ratio,merge_minor = True)

