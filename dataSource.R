################################ Raw Data Readings ###################################
#### iPhone normalized rotated:
iphone_normalized_rotated_data_file = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_normalized_rotated_2016-08-20.csv'
android_summer_rotated_synchronized_data_file = '/Users/Xing/Dropbox/TravelData/android/android_summer_synchronized_rotated_winsorized_normalized_arbpm2016-09-13.csv'

android_winter_rotated_synchronized_data_file = '/Users/Xing/Dropbox/TravelData/android/android_winter_synchronized_rotated_winsorized_normalized_arbpm2016-09-12.csv'
                                                 
################################ Feature Vectors ###################################
#android_winter_fv_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
#android_summer_fv_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
#iphone_fv_file = '~/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'

### note: in segLen = 16 files, the data contains header
###       in segLen = 8 and others, the data doesn't contain header

data_header <<- c('XMAX', 'XMIN', 'XSTND', 'XAVG', 'XOFFSET', 'XFRQ', 'XENERGYSTND', 
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 
                  'XQUANTILE20', 'XQUANTILE40', 'XQUANTILE60', 'XQUANTILE80', 
                  'YMAX', 'YMIN', 'YSTND', 'YAVG', 'YOFFSET', 'YFRQ', 'YENERGYSTND', 
                  'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 
                  'YQUANTILE20', 'YQUANTILE40', 'YQUANTILE60', 'YQUANTILE80', 
                  'ZMAX', 'ZMIN', 'ZSTND', 'ZAVG', 'ZOFFSET', 'ZFRQ', 'ZENERGYSTND', 
                  'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 
                  'ZQUANTILE20', 'ZQUANTILE40', 'ZQUANTILE60', 'ZQUANTILE80', 
                  'ACC_MAGNITUDE_MAX', 'ACC_MAGNITUDE_MIN', 'ACC_MAGNITUDE_STND', 'ACC_MAGNITUDE_AVG', 
                  'ACC_MAGNITUDE_OFFSET', 'ACC_MAGNITUDE_FRQ', 'ACC_MAGNITUDE_ENERGYSTND', 
                  'ACC_MAGNITUDE_0', 'ACC_MAGNITUDE_1', 'ACC_MAGNITUDE_2', 'ACC_MAGNITUDE_3', 'ACC_MAGNITUDE_4', 'ACC_MAGNITUDE_5', 'ACC_MAGNITUDE_6', 'ACC_MAGNITUDE_7', 'ACC_MAGNITUDE_8', 'ACC_MAGNITUDE_9', 
                  'ACC_MAGNITUDE_QUANTILE20', 'ACC_MAGNITUDE_QUANTILE40', 'ACC_MAGNITUDE_QUANTILE60', 'ACC_MAGNITUDE_QUANTILE80',
                  'X_Rotation_MAX', 'X_Rotation_MIN', 'X_Rotation_STND', 'X_Rotation_AVG', 'X_Rotation_OFFSET', 'X_Rotation_FRQ', 'X_Rotation_ENERGYSTND', 
                  'X_Rotation_0', 'X_Rotation_1', 'X_Rotation_2', 'X_Rotation_3', 'X_Rotation_4', 'X_Rotation_5', 'X_Rotation_6', 'X_Rotation_7', 'X_Rotation_8', 'X_Rotation_9', 
                  'X_Rotation_QUANTILE20', 'X_Rotation_QUANTILE40', 'X_Rotation_QUANTILE60', 'X_Rotation_QUANTILE80', 
                  'Y_RotationMAX', 'Y_RotationMIN', 'Y_RotationSTND', 'Y_RotationAVG', 'Y_RotationOFFSET', 'Y_RotationFRQ', 'Y_RotationENERGYSTND', 
                  'Y_Rotation0', 'Y_Rotation1', 'Y_Rotation2', 'Y_Rotation3', 'Y_Rotation4', 'Y_Rotation5', 'Y_Rotation6', 'Y_Rotation7', 'Y_Rotation8', 'Y_Rotation9', 
                  'Y_RotationQUANTILE20', 'Y_RotationQUANTILE40', 'Y_RotationQUANTILE60', 'Y_RotationQUANTILE80', 
                  'Z_RotationMAX', 'Z_RotationMIN', 'Z_RotationSTND', 'Z_RotationAVG', 'Z_RotationOFFSET', 'Z_RotationFRQ', 'Z_RotationENERGYSTND', 
                  'Z_Rotation0', 'Z_Rotation1', 'Z_Rotation2', 'Z_Rotation3', 'Z_Rotation4', 'Z_Rotation5', 'Z_Rotation6', 'Z_Rotation7', 'Z_Rotation8', 'Z_Rotation9', 
                  'Z_RotationQUANTILE20', 'Z_RotationQUANTILE40', 'Z_RotationQUANTILE60', 'Z_RotationQUANTILE80', 
                  'Screen_brightness_MAX', 'Screen_brightness_MIN', 'Screen_brightness_STND', 'Screen_brightness_AVG', 
                  'pressure_MAX', 'pressure_MIN', 'pressure_STND', 'pressure_AVG', 'magEnergy_MAX', 'magEnergy_MIN', 
                  'magEnergy_STND', 'magEnergy_AVG', 'magEnergy_FREQ', 'magEnergy_OFFSET', 'UID', 'label')

android_winter_fv_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
android_summer_fv_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
iphone_fv_file = '~/Dropbox/TravelData/data/iPhone_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-02.csv'


df_as = read.csv(android_summer_fv_file,header = F)
df_aw = read.csv(android_winter_fv_file,header = F)
df_ip = read.csv(iphone_fv_file,header = F)

df_all = rbind.data.frame(df_as,df_aw,df_ip)
df_all$V164 = ifelse(df_all$V163 %in% c('walk','jog'), 'unwheeled','wheeled')
df_wheeled = df_all[which(df_all$V164 == 'wheeled'),]
df_wheeled$V165 = ifelse(df_wheeled$V163 == 'bike','outdoor','indoor')
write.csv(df_all,"~/Dropbox/TravelData/data/alldata_allmode_segLen8_lag2_arbpm_normalized_2017-03-12.arff",row.names = F,col.names = F)
write.csv(df_wheeled,"~/Dropbox/TravelData/data/alldata_wheeled_segLen8_lag2_arbpm_normalized_2017-03-12.arff",row.names = F,col.names = F)
df_indoor = df_all[which(df_all$V163 %in% c('car','bus','subway')),]
write.csv(df_indoor,"~/Dropbox/TravelData/data/alldata_indoor_segLen8_lag2_arbpm_normalized_2017-03-12.arff",row.names = F,col.names = F)
