import numpy as np
import pandas as pd
import copy
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from multiprocessing import Pool

def prepareGlobalVariable():
    global android_winter_data_file,android_summer_data_file,iphone_data_file
    global df_android_winter,df_android_summer,df_all, uids
    global bike_uids,car_uids,subway_uids,bus_uids,walk_uids,jog_uids
    global travel_modes

    android_winter_data_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
    android_summer_data_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
    iphone_data_file = '~/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
    df_android_winter = pd.read_csv(android_winter_data_file)
    df_android_summer = pd.read_csv(android_summer_data_file)
    df_iphone = pd.read_csv(iphone_data_file)
    df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
    uids = sorted(pd.unique(df_all['UID']))
    new = copy.deepcopy(df_all) ## in python assignment is a bind, not deep copy. Ref: https://docs.python.org/2/library/copy.html
    new['count'] = 1
    x = pd.pivot_table(new, values = ['count'], index = ['UID'],columns=['class'], aggfunc=np.sum)
    ## x is an array of tuple:
    # array([('count', 'bike'), ('count', 'bus'), ('count', 'car'),('count', 'jog'), ('count', 'subway'), ('count', 'walk')], dtype=object)
    bike_uids = x['count']['bike'].dropna().index
    car_uids = x['count']['car'].dropna().index
    subway_uids = x['count']['subway'].dropna().index
    bus_uids = x['count']['bus'].dropna().index
    walk_uids = x['count']['walk'].dropna().index
    jog_uids = x['count']['jog'].dropna().index
    travel_modes = ['bike','car','subway','bus','walk','jog']



def prepareCombination():
    combination = []
    for bku in bike_uids:
        for cu in car_uids:
            for sbu in subway_uids:
                for bu in bus_uids:
                    for wu in walk_uids:
                        for ju in jog_uids:
                            combination.append([bku,cu,sbu,bu,wu,ju])
    return combination



def generate_trainingset(combination_config):
    training_set = []
    for i,e in enumerate(combination_config):
        selection = df_all.loc[(df_all['UID'] == e) & (df_all['class'] == travel_modes[i])].as_matrix()
        training_set += selection.tolist()
    training_set = np.array(training_set)
    x_train = training_set[:,:-1]
    y_train = training_set[:,-1]
    x_train_scaled = preprocessing.scale(x_train.astype(float))
    scaler = preprocessing.StandardScaler().fit(x_train.astype(float))
    ## how to apply scaler: x_test_scaled = scaler.transform(x_test.astype(float))
    return x_train_scaled,y_train,scaler



def usr_mode_model_learning(cmb,uid):
    x_train_scaled,y_train,scaler = generate_trainingset(cmb)
    #print set(y_train)
    test_set = df_all[df_all['UID'] == uid].as_matrix()
    x_test = test_set[:,:-1]
    y_test = test_set[:,-1]
    x_test_scaled = scaler.transform(x_test.astype(float))
    if uid in cmb:
        accuracy = -1
    else:
        clf = SGDClassifier(loss = 'hinge',penalty = 'l1',shuffle = True,n_iter = 100)
        clf.fit(x_train_scaled,y_train)
        y_predict = clf.predict(x_test_scaled)
        accuracy = sum(y_predict == y_test)*1.0/len(y_test)
    return accuracy

def usr_mode_model_learning_all():
    combination = prepareCombination()
    test_accuracy_all = [[-1 for i in uids] for j in range(len(combination))]
    for i,cmb in enumerate(combination):
        print cmb
        for j,uid in enumerate(uids):
            print uid
            accuracy = usr_mode_model_learning(cmb,uid)
            test_accuracy_all[i][j] = round(accuracy,2)
    save_to_file = 'uid_mode_pairwise_validation_'+ date.today().isoformat() +'.csv'
    np.savetxt(save_to_file, test_accuracy_all, delimiter=',',fmt='%1.3f')   # X is an array



def trail():
    fn = '/Users/Xing/Dropbox/TravelData/data/allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-05.csv'
    dataset = pd.read_csv(fn)

def usr_mode_model_learning_multiprocess(i):
    print 'hello world',i

def test_pool_order(x):
    return x

def test_on_pool():
    p = Pool(4)
    comb_pool = p.map(test_pool_order,combination)
    return comb_pool


if __name__ == '__main__':
    prepareGlobalVariable()
    combination = prepareCombination()
    x = test_on_pool()
    print x == combination

