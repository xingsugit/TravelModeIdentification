import pandas as pd
import numpy as np
from sklearn import preprocessing
## calculate the non-sensor similarity between users
## return: A matrix: ns_similarity_matrix
def calculate_ns_similarity(xi):
    user_profile_file = '~/Dropbox/TravelData/data/user_non_factor_profile.csv'
    user_profile = pd.read_csv(user_profile_file)
    uids = user_profile['uid'].unique()
    ns_similarity_matrix = [[5 for i in range(len(uids))] for i in range(len(uids))]
    for i in range(len(uids)):
        for j in range(len(uids)):
            A = user_profile[user_profile['uid'] == uids[i]].values
            B = user_profile[user_profile['uid'] == uids[j]].values
            ns_similarity_matrix[i][j] = sum(np.equal(A,B)[0]) * (1-xi) + (5-sum(np.equal(A,B)[0])) * xi
    return ns_similarity_matrix

def data_center(a):
    ## a is a numpy array
    return a.mean(axis=0)

## it calculate the similarity scores between between current uid and other uids within the 'mode'
## return topn uids with highest similarity score
## here T is normalized, df is not. first generate the scaler for df, and then use scaler to scale each user's data
def high_similar_uids_in_current_mode(df,uid,T,mode,ns_similarity_matrix,topn):
    similarities = []
    uids = sorted(pd.unique(df['UID']))
    scaler = preprocessing.StandardScaler().fit(df.as_matrix()[:,:-2].astype(float))
    rest_uids = [e for e in uids if e!= uid]
    for u in rest_uids:
        u_df = df.loc[(df['UID'] == u) & (df['class'] == mode)]
        if (len(u_df)) > 0:
            #u_data = u_df.as_matrix()[:,:-2]
            u_data = scaler.transform(u_df.as_matrix()[:,:-2].astype(float))
            u_data_center = data_center(u_data)
            T_data_center = data_center(T)
            non_sensor_s = ns_similarity_matrix[np.where(uids==u)[0][0]][np.where(uids==uid)[0][0]]
            sensor_s = np.sqrt(sum((u_data_center[i] - T_data_center[i]) ** 2 for i in range(len(u_data_center))))
            similarities.append(non_sensor_s/sensor_s)
        else:
            similarities.append(-1)
    topn_indexes = np.argsort(similarities)[::-1][:topn]
    topn_uids = [rest_uids[i] for i in topn_indexes]
    return topn_uids
