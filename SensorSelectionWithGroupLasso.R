### dynamic sensor selection

library('msgl')
#library('grpreg')
library('gglasso')
library(MASS)
library('caret')

## binarize y: from text label to 1, -1
y_binarize <-function(y,trail_set)
{
  y_binary = ifelse(y==trail_set[1],1,-1)
  y_binary
  
}





set_common_variables <- function()
{
  #android_winter_data_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
  #android_summer_data_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
  #iphone_data_file = '~/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
  
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
  
  android_winter_data_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
  android_summer_data_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-04.csv'
  iphone_data_file = '~/Dropbox/TravelData/data/iPhone_summer_allmode_segLen8_lag2_arbpm_normalize_uid_2017-03-02.csv'
  
  sd_summer = read.csv(android_summer_data_file,header = F)
  sd_winter = read.csv(android_winter_data_file,header = F)
  si = read.csv(iphone_data_file,header = F)
  ### sd is the total data
  sdata <<- rbind.data.frame(sd_summer,sd_winter,si)
  names(sdata) <- data_header
  
  wheeled <<-c("bike",'bus','car','subway')
  unwheeled <<-c('walk','jog') ## walk = 1, jog = -1
  
  indoor <<-c('bus','subway','car') 
  outdoor<<-c('bike')
  
  level1 <<-c('wheeled','unwheeled')
  level21 <<-c('indoor','bike')
  
  bus_o <<-c('bus','others')
  subway_o <<- c('subway','others')
  car_o <<-c('car','others')
  

  
  ## group: acc: 1:84, rotation:85:147, Screen_brightness: 148:151; Pressure: 152:155; Mag: 156:161
  sensors <<- c('Accelerometer','Gyroscope','Light','Barometer','Magnetometer')
  fv_group <<- c(rep(1,84),rep(2,63),rep(3,4),rep(4,4),rep(5,6))
  
  ##level_label
  lv1 <<- 'level1' ##wheeled/unwheeled
  lv21 <<- 'level21'  ##indoor/outdoor
  
  
  ## add sublevel labels
  sdata[[lv1]] = as.factor(ifelse(sdata$label %in% wheeled,'wheeled','unwheeled'))
  sdata[[lv21]] = as.factor(ifelse(sdata$label %in% wheeled, ifelse(sdata$label %in% indoor,'indoor','bike'), NA))
  
  ## preprocessing at the beginning:
  ## scale:
  ### using library: Caret 
  #### use scale instead
  #preProc  <- preProcess(sdata[,1:161]) ##to  (center, scale)
  #sdata[,1:161] = predict(preProc, sdata[,1:161])
  
  ### standardization of the data:
  sdata[,1:161] = scale(sdata[,1:161])
  # check that we get mean of 0 and sd of 1
  #colMeans(scaled.dat)  # faster version of apply(scaled.dat, 2, mean)
  #apply(scaled.dat, 2, sd)
  
  
  
  ## common container for used and left data
  currentInUse <<- data.frame()
  nextArrival_for_test <<- data.frame()
  data_available <<- sdata
  
  
  ### keep tracking of sensors
  current_sensors <<- c()
  level1_sensors <<- c()
  level21_sensors <<- c()
  level22_sensors <<- c()
  level3_sensors <<- c()
  bus_o_sensors <<- c()
  subway_o_sensors <<- c()
  car_o_sensors <<- c()
  
  ### keep tracking of trails and accuracies: here the test needs to be done at the final level of 6 travel modes
  test_counter <<- 0.0
  hit_counter <<- 0.0
  
  ### accuracy path
  accuracy_path <<- c()
  
  ### sensor_usage
  total_sensors_inuse <<- c()
  
  ### for batch learning (20% for testing, 30% initial training, everytime add 5% data (in total 10 round add until it reaches 50%) )
  
  initial_training_index = createDataPartition(y=sdata[['label']],p=0.2)
  initial_batch<<-sdata[unlist(initial_training_index),]
  
  data_left = sdata[-unlist(initial_training_index),]
  
  test_batch_index = createDataPartition(y = data_left[['label']],p=0.25) ## gives 0.8*0.25 = 0.2
  test_batch <<- data_left[unlist(test_batch_index),]
  add_ups <<- data_left[-unlist(test_batch_index),]
  
}

### prepare data: extract data from whatever left, and put in-> inUse (for training), and put the data left in 
prepareData <-function(col_filter,percent_inUse,percent_nextArrival)
{
  
  data_index = createDataPartition(y = data_available[[col_filter]],p = percent_inUse)
  currentInUse <<- data_available[unlist(data_index),]
  
  data_available = data_available[-unlist(data_index),]
  
  next_arrival_index = createDataPartition(y = data_available[[col_filter]],p = percent_nextArrival)
  nextArrival_for_test <<- data_available[unlist(next_arrival_index),]
  
}

### test prepareData
test_prepareData <-function()
{
  col_filter = lv1
  percent_inUse = 0.3
  percent_nextArrival = 0.1
  prepareData(col_filter,percent_inUse, percent_nextArrival)
  print(nrow(sdata))
  print(nrow(currentInUse))
  print(nrow(nextArrival))
  print(nrow(data_available))
}




### train f1: wheeled/unwheeled
### loss: default is ls
### dlt: delta for hsvm,default is 0.1
### ppmax: the maximum sensors to use (a.k.a the maximum group of beta to select)
### acc_lb: lower_bound accuracy (if the accuracy for current pmax is not satisfied, increase pmax by one until it reaches the maximum sensors)
train_f1 <-function(trainingset,testset,loss = 'ls',dlt = 0.5,ppmax = 2,acc_lb)
{
  ### f1, estimate on level1: wheeled/unwheeled
  
  max_acc = 0
  sensor_group = c()
  x_train = as.matrix(trainingset[,1:161])
  y_train = y_binarize(trainingset[[lv1]],level1)
  x_test = as.matrix(testset[,1:161])
  #y_test = y_binarize(testset[[lv1]],level1)
  y_test = testset[[lv1]]
  sensor_group = c()
  while(max_acc < acc_lb && ppmax < 6 || length(sensor_group) == 0)
  {
    #print(max_acc < acc_lb)
    #print(length(sensor_group) )
    print(paste("current pmax = ",ppmax))
    f1 = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    test_result = predict(f1,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    max_round = f1$dim[2]
    if(loss == 'ls')
    {
      for(j in 1:max_round)
      {
        test_result[,j] = ifelse(test_result[,j]>0,level1[1],level1[2])
      }
      
    }
    current_acc = sum(test_result[,max_round] == y_test)/length(y_test)
    if(max_acc < current_acc){
      max_acc = current_acc
      last_beta = f1$beta[,max_round]
      nonzeros = last_beta != 0
      sensor_group = unique(f1$group[nonzeros])
      sensor_in_use = sensors[sensor_group]
      print(sensor_in_use)
    }
    ppmax = ppmax + 1
  }
  print(max_acc)
  print(sensor_group)
  level1_sensors <<-sensor_group
  f1
}

train_f22<-function(trainingset,testset,loss='ls',dlt = 0.5,ppmax = 2,acc_lb)
{
  trainingset = trainingset[which(trainingset$label %in% unwheeled),]
  testset = testset[which(testset$label %in% unwheeled),]
  max_acc = 0
  sensor_group = c()
  x_train = as.matrix(trainingset[,1:161])
  y_train = y_binarize(trainingset[['label']],unwheeled)
  x_test = as.matrix(testset[,1:161])
  #y_test = y_binarize(testset[['label']],unwheeled)
  y_test = testset[['label']]
  while(max_acc < acc_lb && ppmax < 6 || length(sensor_group) ==0 )
  {
    print(paste("current pmax = ",ppmax))
    f22 = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    test_result = predict(f22,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    max_round = f22$dim[2]
    if(loss == 'ls')
    {
      for(j in 1:max_round)
      {
        test_result[,j] = ifelse(test_result[,j]>0,unwheeled[1],unwheeled[2]) ## from y_binarize, walk is 1, jog is -1
      }
      
    }
    current_acc = sum(test_result[,max_round] == y_test)/length(y_test)
    if(max_acc < current_acc){
      max_acc = current_acc
      last_beta = f22$beta[,max_round]
      nonzeros = last_beta != 0
      sensor_group = unique(f22$group[nonzeros])
      sensor_in_use = sensors[sensor_group]
      print(sensor_in_use)
    }
    ppmax = ppmax + 1
  }
  print(max_acc)
  print(sensor_group)
  level22_sensors <<-sensor_group
  f22
}

test_train_f1 <-function()
{
  
  acc_lb = 0.5
  pmax = 2
  dlt = 0.5
  loss = 'ls'
  
  percent_inUse = 0.2
  percent_nextArrival = 0.01

  #sensor_numbers_usage = c()
  prepareData('label',percent_inUse,percent_nextArrival)
  
  trainingset = currentInUse
  testset = nextArrival_for_test
  f1_test= train_f1(trainingset,testset,loss = 'ls',dlt = 0.5,ppmax = 2,acc_lb)
}

train_f21 <-function(trainingset,testset,loss = 'ls',dlt = 0.5,ppmax = 2, acc_lb)
{
  trainingset = trainingset[which(trainingset$label %in% wheeled),]
  testset = testset[which(testset$label %in% wheeled),]
  max_acc = 0
  sensor_group = c()
  x_train = as.matrix(trainingset[,1:161])
  y_train = y_binarize(trainingset[[lv21]],level21)
  x_test = as.matrix(testset[,1:161])
  #y_test = y_binarize(testset[[lv21]],level21)
  y_test = testset[[lv21]]
  while(max_acc < acc_lb && ppmax < 6 || length(sensor_group) ==0 )
  {
    print(paste("current pmax = ",ppmax))
    f21 = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    test_result = predict(f21,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    max_round = f21$dim[2]
    
    if(loss == 'ls')
    {
      for(j in 1:max_round)
      {
        test_result[,j] = ifelse(test_result[,j]>0,level21[1],level21[2]) ## from y_binarize, walk is 1, jog is -1
      }
      
    }
    
    current_acc = sum(test_result[,max_round] == y_test)/length(y_test)
    if(max_acc < current_acc){
      max_acc = current_acc
      last_beta = f21$beta[,max_round]
      nonzeros = last_beta != 0
      sensor_group = unique(f21$group[nonzeros])
      sensor_in_use = sensors[sensor_group]
      print(sensor_in_use)
    }
    ppmax = ppmax + 1
  }
  print(max_acc)
  print(sensor_group)
  level21_sensors <<-sensor_group
  f21
}

## one vs all
train_f3_OVA <-function(trainingset,testset,loss='ls',dlt = 0.5,ppmax = 2,acc_lb)
{
  ## if multi = 'OVA': one vs all
  f_bus_o = train_f_bus_o(trainingset,testset,loss = loss,dlt = dlt,ppmax = ppmax,acc_lb)
  f_subway_o = train_f_subway_o(trainingset,testset,loss = loss,dlt = dlt,ppmax = ppmax,acc_lb)
  f_car_o = train_f_car_o(trainingset,testset,loss = loss,dlt = dlt,ppmax = ppmax,acc_lb)
  level3_sensors <<-unique(unlist(c(car_o_sensors,subway_o_sensors,bus_o_sensors)))
  list(f_bus_o = f_bus_o,f_subway_o = f_subway_o,f_car_o = f_car_o)
  
}

use_f3_OVA <-function(f3, test_sample)
{
  f_bus_o = f3$f_bus_o
  f_subway_o = f3$f_subway_o
  f_car_o = f3$f_car_o
  ## how to vote? 
  
  vote = c(0,0,0) ## bus,subway,car
  predict_result = c(0,0,0)
  predict_result[1] = predict.gglasso(f_bus_o,test_sample,type = 'class')[f_bus_o$dim[2]]
  predict_result[2] = predict.gglasso(f_subway_o,test_sample,type = 'class')[f_subway_o$dim[2]]
  predict_result[3] = predict.gglasso(f_car_o,test_sample,type = 'class')[f_car_o$dim[2]]
  
  
  if(predict_result[1] == 1)
  {
    vote[1] = vote[1] + 1
  }
  else
  {
    vote[2] = vote[2] + 1
    vote[3] = vote[3] + 1
  }
  if(predict_result[2] == 1)
  {
    vote[2]= vote[2] + 1
    
  }
  else
  {
    vote[3] = vote[3] + 1
    vote[1] = vote[1] + 1
  }
  
  if(predict_result[3] == 1)
  {
    vote[3] = vote[3] + 1
  }
  else
  {
    vote[1] = vote[1] + 1
    vote[2] = vote[2] + 1
  }
  
  indoor[which.max(vote)]
}

test_f3_OVA<-function()
{
  f3 = train_f3_OVA(trainingset,testset,acc_lb = 0.7)
  test_sample = data_available[10000,]
  test_sample_x = as.matrix(test_sample[,1:161])
  predict_level3_mode = use_f3_OVA(f3,test_sample_x)
  test_sample_y = test_sample[['label']]
}

train_f_bus_o <- function(trainingset,testset,loss = 'ls',dlt = 0.5,ppmax = 2,acc_lb)
{
  trainingset = trainingset[which(trainingset$label %in% indoor),]
  testset = testset[which(testset$label %in% indoor),]
  max_acc = 0
  sensor_group = c()
  x_train = as.matrix(trainingset[,1:161])
  y_train = y_binarize(trainingset[['label']],bus_o)
  x_test = as.matrix(testset[,1:161])
  y_test = y_binarize(testset[['label']],bus_o)
  #y_test = testset[['label']]
  sensor_group = c()
  
  while(max_acc < acc_lb && ppmax < 6 || length(sensor_group) == 0)
  {
    print(paste("current pmax = ",ppmax))
    f_bus_o = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    test_result = predict(f_bus_o,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    max_round = f_bus_o$dim[2]
    
    if(loss == 'ls')
    {
      for(j in 1:max_round)
      {
        test_result[,j] = ifelse(test_result[,j]>0,1,-1) ## from y_binarize, walk is 1, jog is -1
      }
      
    }
    
    current_acc = sum(test_result[,max_round] == y_test)/length(y_test)
    if(max_acc < current_acc){
      max_acc = current_acc
      last_beta = f_bus_o$beta[,max_round]
      nonzeros = last_beta != 0
      sensor_group = unique(f_bus_o$group[nonzeros])
      sensor_in_use = sensors[sensor_group]
      print(sensor_in_use)
    }
    ppmax = ppmax + 1
  }
  print(max_acc)
  print(sensor_group)
  bus_o_sensors <<-sensor_group
  f_bus_o
}

train_f_subway_o <- function(trainingset,testset,loss = 'ls',dlt = 0.5,ppmax = 2,acc_lb)
{
  trainingset = trainingset[which(trainingset$label %in% indoor),]
  testset = testset[which(testset$label %in% indoor),]
  max_acc = 0
  sensor_group = c()
  x_train = as.matrix(trainingset[,1:161])
  y_train = y_binarize(trainingset[['label']],subway_o)
  x_test = as.matrix(testset[,1:161])
  y_test = y_binarize(testset[['label']],subway_o)
  #y_test = testset[['label']]
  sensor_group = c()
  
  while(max_acc < acc_lb && pmax < 6 || length(sensor_group) == 0)
  {
    print(paste("current pmax = ",ppmax))
    f_subway_o = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    test_result = predict(f_subway_o,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    max_round = f_subway_o$dim[2]
    if(loss == 'ls')
    {
      for(j in 1:max_round)
      {
        test_result[,j] = ifelse(test_result[,j]>0,1,-1) ## from y_binarize, walk is 1, jog is -1
      }
      
    }
    current_acc = sum(test_result[,max_round] == y_test)/length(y_test)
    if(max_acc < current_acc){
      max_acc = current_acc
      last_beta = f_subway_o$beta[,max_round]
      nonzeros = last_beta != 0
      sensor_group = unique(f_subway_o$group[nonzeros])
      sensor_in_use = sensors[sensor_group]
      print(sensor_in_use)
    }
    ppmax = ppmax + 1
  }
  print(max_acc)
  print(sensor_group)
  subway_o_sensors <<-sensor_group
  f_subway_o
}

train_f_car_o <- function(trainingset,testset,loss = 'ls',dlt = 0.5,ppmax = 2,acc_lb)
{
  trainingset = trainingset[which(trainingset$label %in% indoor),]
  testset = testset[which(testset$label %in% indoor),]
  max_acc = 0
  sensor_group = c()
  x_train = as.matrix(trainingset[,1:161])
  y_train = y_binarize(trainingset[['label']],car_o)
  x_test = as.matrix(testset[,1:161])
  y_test = y_binarize(testset[['label']],car_o)
  #y_test = testset[['label']]
  sensor_group = c()
  
  while(max_acc < acc_lb && ppmax < 6 || length(sensor_group) == 0)
  {
    print(paste("current pmax = ",pmax))
    f_car_o = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    test_result = predict(f_car_o,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)
    max_round = f_car_o$dim[2]
    if(loss == 'ls')
    {
      for(j in 1:max_round)
      {
        test_result[,j] = ifelse(test_result[,j]>0,1,-1) ## from y_binarize, walk is 1, jog is -1
      }
      
    }
    current_acc = sum(test_result[,max_round] == y_test)/length(y_test)
    if(max_acc < current_acc){
      max_acc = current_acc
      last_beta = f_car_o$beta[,max_round]
      nonzeros = last_beta != 0
      sensor_group = unique(f_car_o$group[nonzeros])
      sensor_in_use = sensors[sensor_group]
      print(sensor_in_use)
    }
    ppmax = ppmax + 1
  }
  print(max_acc)
  print(sensor_group)
  car_o_sensors <<-sensor_group
  f_car_o
}

work_procedure<-function()
{
  
    set_common_variables()
    percent_inUse = 0.2
    percent_nextArrival = 0.01
    pmax = 1
    dlt = 0.5
    acc_lb = 0.5
    sensor_numbers_usage = c()
    prepareData('label',percent_inUse,percent_nextArrival)
    loss_method = 'ls'
    
    ## for test need to remove
    ## data_available <<- data_available[1:50,]
    while(nrow(data_available) > 20 && nrow(nextArrival_for_test) > 0)
    {
      print('entering the level1:') 
      f1 = train_f1(currentInUse,nextArrival_for_test,loss = loss_method,dlt = dlt,ppmax = pmax,acc_lb)
      if('wheeled' %in% nextArrival_for_test$level1)
      {
        print('entering the level21:') 
        f21 = train_f21(currentInUse,nextArrival_for_test,loss = loss_method,dlt = dlt,ppmax = pmax,acc_lb)
        if(any(indoor %in% nextArrival_for_test$label))
        {
          print('entering the level3:') 
          f3 = train_f3_OVA(currentInUse,nextArrival_for_test,loss = loss_method,dlt = dlt,ppmax = pmax,0.7)
        }
      }
      if('unwheeled' %in% nextArrival_for_test$level1)
      {
        print('entering the level22:') 
        f22 = train_f22(currentInUse,nextArrival_for_test,loss = loss_method,dlt = dlt,ppmax = pmax,0.7)
      }
      
      
      next_sample = data_available[1,] ## one sample --> later we can change it into a small bunch
      next_sample_train = as.matrix(next_sample[,1:161])
      next_sample_lv1_y = y_binarize(next_sample[[lv1]],level1)
      predict_level1_mode = predict(f1,next_sample_train,group = fv_group,loss = loss_method,type = 'class',delta = dlt,pmax = pmax)
      
      if(next_sample[[lv1]] == 'wheeled')
      {
        ### enter level 21: bike - outdoor
        next_sample_lv21_y = y_binarize(next_sample[[lv21]],level21)
        predict_level21_mode = predict(f21,next_sample_train,type = 'class')
        if(next_sample_lv21_y == 'bike')
        {
          if(next_sample_lv21_y == predict_level21_mode && next_sample_lv1_y == next_sample_lv1_y)
          {
            hit_counter = hit_counter + 1
          }
          current_sensors = union(level1_sensors,level21_sensors)
          total_sensors_inuse <<- c(total_sensors_inuse,length(current_sensors))
        }
        else
        {
          ## enter level 3
          next_sample_lv3_y = next_sample[['label']]
          predict_level3_mode = use_f3_OVA(f3,next_sample_train)
          if(next_sample_lv3_y == predict_level3_mode)
          {
            hit_counter = hit_counter + 1
          } 
          current_sensors = unique(unlist(c(level1_sensors,level21_sensors,level3_sensors)))
          total_sensors_inuse <<- c(total_sensors_inuse,length(current_sensors))
        }
        
      }
      else
      {
        ### enter level 22
        next_sample_lv22_y = y_binarize(next_sample[['label']],unwheeled)
        predict_level22_mode = predict(f22,next_sample_train,type = 'class')
        if(next_sample_lv22_y == predict_level22_mode && next_sample_lv1_y == predict_level1_mode)
        {
          hit_counter <<- hit_counter + 1
        }
        current_sensors = union(level1_sensors,level22_sensors)
        total_sensors_inuse <<- c(total_sensors_inuse,length(current_sensors))
        
      }
      
      currentInUse <<- rbind.data.frame(currentInUse,next_sample)
      data_available <<- data_available[-1,]
      next_arrival_index = createDataPartition(y = data_available[['label']],p = 0.1)
      nextArrival_for_test <<- data_available[unlist(next_arrival_index),]
      test_counter <<- test_counter + 1
      accuracy_path <<- c(accuracy_path,1.0*hit_counter/test_counter)
    }
  
}

work_procedure_for_small_batch<-function()
{
  set_common_variables()
  ##everytime take out 5% of add-ups 
  ## create the partition of add-ups:
  N = 10
  end = 0
  pmax = 1
  dlt = 0.5
  acc_lb = 0.8 ##lower bound
  sensor_numbers_usage = c()
  train_batch  = initial_batch
  loss = 'ls'
  end=0
  for(i in 1:(nrow(add_ups)/N))
  {
    begin = end + 1
    end = as.integer(i * N)
    if(end > nrow(add_ups))
    {
      end = nrow(add_ups)
    }
    current_addup = add_ups[begin:end,]
    
    ### get to the first level
    
    
    f1 = train_f1(train_batch,test_batch,loss = loss,dlt = dlt,ppmax = pmax,acc_lb)
    if('wheeled' %in% train_batch$level1)
    {
      f21 = train_f21(train_batch,test_batch,loss = loss,dlt = dlt,ppmax = pmax,acc_lb)
      if(any(indoor %in% train_batch$label))
      {
        f3 = train_f3_OVA(train_batch,test_batch,loss = loss,dlt = dlt,ppmax = pmax,0.7)
        current_sensors = unique(unlist(c(level1_sensors,level21_sensors,level3_sensors)))
        total_sensors_inuse <<- c(total_sensors_inuse,length(current_sensors))
        
      }
      if('bike' %in% train_batch$label)
      {
        current_sensors = union(level1_sensors,level21_sensors)
        total_sensors_inuse <<- c(total_sensors_inuse,length(current_sensors))
      }
    }
    if('unwheeled' %in% train_batch$level1)
    {
      f22 = train_f22(train_batch,test_batch,loss = loss,dlt = dlt,ppmax = pmax,0.7)
      current_sensors = union(level1_sensors,level22_sensors)
      total_sensors_inuse <<- c(total_sensors_inuse,length(current_sensors))
    }
    
    
    train_batch = rbind.data.frame(train_batch,current_addup)
    print('**************************************************')
    print("Enter next batch, the training batch size increase to:")
    print(nrow(train_batch))
    
    ### end of one process
    
    
  }
  results <- data.frame(f1_acc = acc_f1,f21_acc = acc_f21,f22_acc = acc_f22,f3_acc = (acc_bus_o+acc_car_o+acc_subway_o)/3, sensor_numbers_usage = total_sensors_inuse)
  write.csv(results,"~/Dropbox/TravelData/results/Ls_batch_learning_sensor_usage_acclb0.8.csv",row.names = F)
  results
}

results <- data.frame(f1_acc = acc_f1[-1],f21_acc = acc_f21[-1],f22_acc = acc_f22,f3_acc = (acc_bus_o+acc_car_o+acc_subway_o)/3, indoor_path = total_sensors_inuse[x1],bike_path =total_sensors_inuse[x2],unwheeled_path = total_sensors_inuse[x3] )


# 
# plot(x = seq(1,nrow(results)), y = results$f3_acc, type = "b", col = "red",ylab = "Accuracy",xlab = 'Iteration')#, axes = FALSE)#, xlab = "", ylab = "")
# 
# par(new = TRUE)
# x <- barplot(results$indoor_path, 
#              axes = FALSE,
#              col = "blue", 
#              xlab = "",
#              ylab = "# of Sensors In Use",
#              ylim = c(0, 5) )[, 1]
# axis(4, at = seq(0,5), labels = seq(0,5)))
# ats <- c(seq(0, 5, 1), 6); axis(4, at = ats,label = ats, las = 2)
# axis(3, at = x, labels = NA) 
# 
# 
# mtext(text="Accuracy of Wheeled Indoor path with Sensor Selection through iterations", side = 3, line = 1)
# box()


batch_learning_result = work_procedure_for_small_batch()
write.csv(batch_learning_result,"U:\\TravelData\\Logit_batch_learning_sensor_usage_acclb0.8.csv",row.names = F)


### for plot:
training_index = createDataPartition(y=sdata[[lv1]],p=0.7)
training_data = sdata[unlist(initial_training_index),]
test_data = sdata[-unlist(initial_training_index),]
#f1 = train_f1(currentInUse,nextArrival_for_test,loss = loss_method,dlt = dlt,ppmax = pmax,acc_lb)
x_train = as.matrix(training_data[,1:161])
y_train = y_binarize(training_data[[lv1]],level1)
x_test = as.matrix(test_data[,1:161])
#y_test = y_binarize(testset[[lv1]],level1)
y_test = test_data[[lv1]]
f1 = gglasso(x = x_train,y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = 5)
test_result = predict(f1,x_test,group = fv_group,loss = loss,delta = dlt,pmax = ppmax)

