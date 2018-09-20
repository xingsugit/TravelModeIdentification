import pandas as pd
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


def prepareTravelData(segLen,lag,timestamp):
    define_global_variables()
    path = '~/Dropbox/TravelData/data/'
    if lag == segLen:
        slide = 'noSlide'
    else:
        slide = 'lag' + str(lag)

    android_winter_data_file = path + 'android_winter_allmode_segLen'+str(segLen) + '_' + slide + '_arbpm_normalize_uid_' + timestamp + '.csv'
    android_summer_data_file = path + 'android_summer_allmode_segLen'+str(segLen) + '_' + slide + '_arbpm_normalize_uid_' + timestamp + '.csv'
    iphone_data_file         = path + 'iphone_summer_allmode_segLen'+str(segLen) + '_' + slide + '_arbpm_normalize_uid_' + timestamp + '.csv'
    df_header = acc_headers + rotation_headers + light_headers + pressure_headers + mag_headers + ['UID','label']
    #android_winter_data_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
    #android_summer_data_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
    #iphone_data_file = '~/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
    df_android_winter = pd.read_csv(android_winter_data_file,header = None)
    df_android_summer = pd.read_csv(android_summer_data_file,header = None)
    df_iphone = pd.read_csv(iphone_data_file,header = None)
    df_android_winter.columns = df_header
    df_android_summer.columns = df_header
    df_iphone.columns = df_header
    ### select some uids in df_iphone, to see whether the diversity is the reason for online learning no converge
    #df_iphone = df_iphone.loc[df_iphone['UID'].isin([5])]

    #df_all = pd.concat([df_iphone, df_android_summer,df_android_winter])
    #uids = sorted(pd.unique(df_all['UID']))
    return df_android_winter, df_android_summer, df_iphone
