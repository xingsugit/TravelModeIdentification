## Library
library(hash)
library(MASS)
############## Util Functions ###########
get_v1 <- function(x)
{
  slice_index = regexpr("\\.", x)[1]
  first_v = substring(x,1,slice_index+6)
  first_v
}

get_v2 <- function(x)
{
  slice_index = regexpr("\\.", x)[1]
  second_v = substring(x,slice_index+7,nchar(x))
  second_v
}

######## End of Util Functions #########

########### Global Parameters #####
## iphone data stream 
  ## current_time, activity, calibrated_magnetic_x,calibrated_magnetic_y,calibrated_magnetic_z,
  ## uncal_magnetic_x,uncal_magnetic_y,uncal_magnetic_z
  ## gravity_x,gravity_y,gravity_z,
  ## magnetic_heading,
  ## calib_rotateRate.x,calib_rotateRate.y,calib_rotateRate.z, //from deviceMotion, bias has been calibrated
  ## raw_rotateRate.x,raw_rotateRate.y,raw_rotateRate.z, //from gyroscope, raw gyro data.
  ## pitch,roll,yaw, //Euler angles (roll, pitch, and yaw values). 
  ## r1,r1,...,r9, //rotation matrix ( calc: standard = np.dot(read,rm))??
  ## accelerate.x,accelerate.y,accelerate.z] //The acceleration that the user is giving to the device.
  ## pressure //The recorded pressure, in kilopascals.
  ## screen_brigtness //between 0.0 - 1.0


# Rotation Unit: radians
## Pitch: A pitch is a rotation around a lateral axis that passes through the device from side to side.
## Roll: A roll is a rotation around a longitudinal axis that passes through the device from its top to bottom.
## Yaw: A yaw is a rotation around an axis that runs vertically through the device. It is perpendicular to the body of the device, 
##     with its origin at the center of gravity and directed toward the bottom of the device.

## current reference frame: xManageticNorthZVertical
## Describes a reference frame in which the Z axis is vertical and the X axis points toward magnetic north. 
## Note that using this reference frame may require device movement to calibrate the magnetometer.



### for Magnetic:
## This is the “raw” magnetic-field value, unlike the calibrated value of the magneticField 
## property of CMDeviceMotion which filters out the bias introduced by the device and, in some cases, its surrounding fields. More accurate.

## For rotation: 
## A CMRotationRate structure contains data specifying the device’s rate of rotation around three axes. 
## The value of this property contains a measurement of gyroscope data whose bias has been removed by Core Motion algorithms. 
## The identically name property of CMGyroData, on the other hand, gives the raw data from the gyroscope. 
## The structure type is declared in CMGyroData.h.
name_list = c('timestamp','label','cal_mag_x','cal_mag_y','cal_mag_z','raw_mag_x','raw_mag_y','raw_mag_z','gravity_x','gravity_y',
              'gravity_z','mag_heading','calib_rotationRate_x','calib_rotationRate_y','calib_rotationRate_z','raw_rotationRate_x','raw_rotationRate_y',
              'raw_rotationRate_z','angle_pitch','angle_roll','angle_yall', paste0('rm',seq(1,9)),'acc_x','acc_y',"acc_z",'pressure','screen_brightness')

uid_hash = hash()

sensors_name_list = c('raw_mag_x','raw_mag_y','raw_mag_z','rotate_acc_x','rotate_acc_y',"rotate_acc_z",
                      'rotate_rotation_rate_x','rotate_rotation_rate_y','rotate_rotation_rate_z','pressure','screen_brightness')
sensor_group_length_vector = c(3,3,3,1,1)
########## End of Global Parameters ########
## steps to process the data
## 1. deal with V12 and split it into two values
## 1.1 extract the uid and add it into a new column
## 2. deal with freq, make freq 16 into freq 5.
## 3. remove all the dummys
## 4. from the valid data, get the format 



extract_uid <- function(f_name)
{
  print("extract UID")
  uid = substring(f_name,5,40)
  # sample: "010FDD1A-4C28-4C61-A74E-74458DCAE5FE"
  if(! has.key(uid,uid_hash))
  {
    uid_hash[[uid]] = length(uid_hash) + 1
  }
  
  uid_hash[[uid]]
}


## deal with the bug in one version of the app: there's a ',' missing between 12th and 13th variable, 
## so the 12th value is actually a combination of 12th and 13th value. This function seperate them. 
## @temp: the data source that is read in file a data file. If temp contains 34 columns, then the function will process it and make it 15 columns in total
## and if the data source has 35 columns, then the data source keeps the original shape.
## Then the column name is assigned to the data. 

preprocess_data<-function(temp) 
{
  print("process 12th variable and set variable names")
  if(ncol(temp) == 34)
  {
    temp$V12 = as.character(temp$V12)
    temp$V35 <- sapply(temp$V12, get_v2)
    temp$V12 <- sapply(temp$V12,get_v1)
    temp <- temp[,c(1:12,35,13:34)]
  }
  
  names(temp) = name_list
  temp
}



remove_warm_up_data<-function(temp)
{
  temp = temp[which(temp$pressure != '(null)'),]
  if(nrow(temp) != 0) ##some of the data doesn't have pressure (Arthur's phone)
  {
    start_index = max(min(which(temp[,3] != 0),min(which(temp[,4] != 0)),min(which(temp[,5] != 0))))
    if(start_index > 1)
    {
      temp = temp[-(1:start_index),]
    }
  }
  temp
}
remove_rm_zeros<-function(temp)
{

  indexes = which(temp$rm1 ==0 & temp$rm2 == 0 & temp$rm3 == 0 & temp$rm4 == 0 & temp$rm5 == 0 & temp$rm6 == 0 & temp$rm7 == 0 & temp$rm8 == 0 & temp$rm9 == 0)
  temp = temp[-indexes,]
 
  temp
}

remove_dummy_data<-function(temp) ### remove dummy, 2 seconds after dummy, which is 10 readings, and then 2 seconds before stop of the data.
{
  print("remove dummy data")
  ##remove all dummy data first
  temp = remove_warm_up_data(temp)
  #temp = remove_rm_zeros(temp)
  if(nrow(temp) != 0)
  {
    temp = temp[which(temp$label != 'dummy'),]
    temp$timestamp = as.numeric(temp$timestamp)
    ## if there're more than 1 travel mode in one data
    label1 = temp$label[1:(nrow(temp)-1)]
    label2 = temp$label[2:nrow(temp)]
    label_change_indexes = which(label1 != label2) + 1
    print(label_change_indexes)
    if(length(label_change_indexes) > 0)
    {
      segment = data.frame()
      begin = 11
      for(i in label_change_indexes)
      {
        end = i - 10
        print(paste("begin = ",begin,"end =",end))
        ## each extracted segment, reset the timestamps and let it start from 0.
        temp[begin:end,"timestamp"] = temp[begin:end,"timestamp"] - temp[begin,"timestamp"]
        segment = rbind.data.frame(segment,temp[begin:end,])
        begin = i + 10
      }
      segment = rbind.data.frame(segment,temp[begin:(nrow(temp)-10),])
      segment
    }
    else
    {
      temp = temp[11:(nrow(temp)-10),]
      temp[,"timestamp"] = temp[,"timestamp"] - temp[1,"timestamp"]
      temp
    }
    
  }
  else
  {
    temp
  }
  
}




process_sampling_frequency<-function(temp)
{
  print("process the sampling")
  new_temp = temp[seq(1,nrow(temp),3),]
  new_temp
}
## Workflow: 
generate_full_data_file <-function()
{
  data_path = '~/Dropbox/TravelData/data/iPhoneCollection/'
  data_files = list.files(data_path,pattern = '*csv')
  total_data = data.frame()
  setwd(data_path)
  for(f in data_files)
  {
    print(f)
    #1. readin file
    temp = read.csv(f,skip = 2,header = F,stringsAsFactors=F)
    #2. decide whether need to reduce the sampling frequency:
    slice_index2 = regexpr("_2016", f)
    slice_index = regexpr("freq", f)
    freq = substring(f,slice_index+4,slice_index2-1)
    if(freq == '16')
    {
      temp = process_sampling_frequency(temp)
    }
    #3. preprocess to fix the 12th variable bug
    sample_new = preprocess_data(temp)
    #4. remove dummy data and warmup data. and rm matrix == 0
    sample_refined = remove_dummy_data(sample_new)
    
    #5. add uid in to the data and add it to the total data pool
    ## here we need to check sample_refined size is > 0 because there's data contains no pressure
    ## For now, we omit his data.
    if(nrow(sample_refined) >0 )
    {
      sample_refined$uid = extract_uid(f)
      total_data = rbind.data.frame(total_data,sample_refined)
    }
    print(paste('finish processing file:',f))
  }
  
  total_data$timestamp = as.numeric(total_data$timestamp)
  total_data$label = as.factor(total_data$label)
  for(i in 3:ncol(total_data))
  {
    print(i)
    total_data[, i] <- as.numeric(total_data[,i])
  }
  
  
  total_data = total_data[complete.cases(total_data),]
  ## remove jessie's bus data (it is labeled with wrong mode)
  total_data = total_data[-which(total_data$uid == 1 & total_data$label == 'bus'),]
  write_to = paste0("~/Dropbox/TravelData/data/iPhoneCollection_",Sys.Date(),".csv")
  print(paste0("Write the processed original data into file: ",write_to))
  write.csv(total_data,write_to,row.names = F)
  total_data
}
# ---------------------- This rotate function is deprecated. because of inefficiency ------------------------# 
### rotate the phone axis to the reference framework (x magnetic north z vertical)
rotate_acc_to_reference<-function(acc,rm) ## acc is c(acc_x,acc_y,acc_z), rm is (rm1~9)
{
  acc = as.numeric(acc)
  rm = as.numeric(rm)
  ## note that for the rotation matrix, t(rm) = solve(rm). but to make it clear, we still use solve(rm)
  ## unit of acc and gravity is G (1G = -9.8m/s^2)
  acc_observed = matrix(acc,nrow = 1,ncol = 3)
  rotation_matrix = matrix(rm,nrow = 3,ncol = 3)
  ## acc_reference %*% rotation_matrix = acc_observed
  acc_reference = acc_observed %*% solve(rotation_matrix)
  acc_reference
}
rotate_acc_in_dataframe<-function(df)
{
  result = data.frame()
  
  for(i in 1:nrow(df))
  {
    print(i)
    curr_result = rotate_acc_to_reference(df[i,c('acc_x','acc_y','acc_z')],df[i,paste0('rm',seq(1,9))])
    result = rbind.data.frame(result,curr_result)
  }
  names(result) = c("rotate_acc_x","rotate_acc_y","rotate_acc_z")
  df =  cbind.data.frame(df,result)
  df
}
# ----------------------------- deprecated rotation functions end here ------------------------------------# 

# ----------------------------- Parallel Version of rotation function -------------------------------------#
rotate_acc_to_reference2 <- function(x,y,z,rm1,rm2,rm3,rm4,rm5,rm6,rm7,rm8,rm9)
{
  acc = as.numeric(c(x,y,z))
  rm = as.numeric(c(rm1,rm2,rm3,rm4,rm5,rm6,rm7,rm8,rm9))
  ## note that for the rotation matrix, t(rm) = solve(rm). but to make it clear, we still use solve(rm)
  ## unit of acc and gravity is G (1G = -9.8m/s^2)
  acc_observed = matrix(acc,nrow = 1,ncol = 3)
  rotation_matrix = matrix(rm,nrow = 3,ncol = 3)
  ## acc_reference %*% rotation_matrix = acc_observed
  acc_reference = acc_observed %*% solve(rotation_matrix)
  acc_reference
}

rotate_acc_in_dataframe2<-function(df)
{
  result = mapply(rotate_acc_to_reference2,df$acc_x,df$acc_y,df$acc_z,df$rm1,df$rm2,df$rm3,df$rm4,df$rm5,df$rm6,df$rm7,df$rm8,df$rm9)
  result = t(result)
  rotated_acc = data.frame(rotate_acc_x = result[,1],rotate_acc_y = result[,2],rotate_acc_z = result[,3])
  df =  cbind.data.frame(df,rotated_acc)
  df
}

rotate_rtRate_in_dataframe2<-function(df) ##rotate the rotation rate 
{
  result = mapply(rotate_acc_to_reference2,df$calib_rotationRate_x,df$calib_rotationRate_y,df$calib_rotationRate_z,df$rm1,df$rm2,df$rm3,df$rm4,df$rm5,df$rm6,df$rm7,df$rm8,df$rm9)
  result = t(result)
  rotated_rt = data.frame(rotate_rotation_rate_x = result[,1],rotate_rotation_rate_y = result[,2],rotate_rotation_rate_z = result[,3])
  df =  cbind.data.frame(df,rotated_rt)
  df
}

## ------------------------------ end of parallel version of rotation function ----------------------------------#

### normalize function (normalize by group)
normalize_by_group<-function(m,len_vectors)
{
  begin = 1
  mmax_vec = c()
  mmin_vec = c()
  for(l in len_vectors)
  {
    end = begin+l-1
    mmax = max(m[,begin:end])
    mmin = min(m[,begin:end])
    
    mmax_vec = c(mmax_vec,rep(mmax,l))
    mmin_vec = c(mmin_vec,rep(mmin,l))
    m[,begin:end] = as.matrix(m[,begin:end]) %*% (diag(l)* 1/(mmax - mmin)) - mmin/(mmax - mmin)
    
    begin = end + 1
  }
  print(paste0("mmin_vec = c(",paste(as.character(mmin_vec), collapse=", "),')'))
  print(paste0("mmax_vec = c(",paste(as.character(mmax_vec), collapse=", "),')'))
  m
}
## winsorize by cutting off 0.025 at each side, quantile is calculated by sensor group
winsorize_by_group <-function(x,group_length_vector,fraction=.025)
{
  if(length(fraction) != 1 || fraction < 0 ||
     fraction > 0.5) {
    stop("bad value for 'fraction'")
  }
  begin = 1 
  for(l in group_length_vector)
  {
    print(l)
    end = begin + l -1
    group_value_vector = c()
    for(i in begin:end)
    {
      group_value_vector = c(group_value_vector,x[,i])
    }
    
    lim <- quantile(group_value_vector, probs=c(fraction, 1-fraction))
    for (i in begin:end)
    {
      x[x[,i] < lim[1],i] = lim[1]
      x[x[,i] > lim[2],i] = lim[2]
    }
    begin = end + 1
  }
  x
}

## generate the normalized data 
generate_norm_data_with_label_uid <-function(df,cols,norm_group_length_vector,iswinsorize = T)
{
  m = df[,cols]
  ## insert rotate here
  
  ## 
  if(iswinsorize)
  {
    m = winsorize_by_group(m,norm_group_length_vector)
  }
  m$magnitude_acc = sqrt(m$rotate_acc_x**2 + m$rotate_acc_y**2 + m$rotate_acc_z **2)
  m$magnitude_mag = sqrt(m$raw_mag_x**2 + m$raw_mag_y**2 + m$raw_mag_z **2)
  norm_group_length_vector = c(norm_group_length_vector,1,1)
  norm_m = normalize_by_group(m,norm_group_length_vector)
  ## put the label and uid back
  result_m = cbind.data.frame(timestamp = df$timestamp,acc_x = norm_m$rotate_acc_x,acc_y= norm_m$rotate_acc_y,acc_z = norm_m$rotate_acc_z,
                              rotation_rate_x = norm_m$rotate_rotation_rate_x,rotation_rate_y = norm_m$rotate_rotation_rate_y, rotation_rate_z = norm_m$rotate_rotation_rate_z,
                              magnitude_acc = norm_m$magnitude_acc,magnitude_mag = norm_m$magnitude_mag, pressure = norm_m$pressure,
                              screen_brightness = norm_m$screen_brightness,label = df$label,uid = df$uid)
  result_m
}

extract_cols_from_unnormalized_data <-function(df,cols,winsorize_group_len_vector,iswinsorize = T)
{
  m = df[,cols]
  if(iswinsorize)
  {
    m = winsorize_by_group(m,winsorize_group_len_vector)
  }
  m$magnitude_acc = sqrt(m$rotate_acc_x**2 + m$rotate_acc_y**2 + m$rotate_acc_z **2)
  m$magnitude_mag = sqrt(m$raw_mag_x**2 + m$raw_mag_y**2 + m$raw_mag_z **2)
  
  result_m = cbind.data.frame(timestamp = df$timestamp,acc_x = m$rotate_acc_x,acc_y= m$rotate_acc_y,acc_z = m$rotate_acc_z,
                              rotation_rate_x = m$rotate_rotation_rate_x,rotation_rate_y = m$rotate_rotation_rate_y, rotation_rate_z = m$rotate_rotation_rate_z,
                              magnitude_acc = m$magnitude_acc,magnitude_mag = m$magnitude_mag, pressure = m$pressure,
                              screen_brightness = m$screen_brightness,label = df$label,uid = df$uid)
  result_m
  
}

total_data = generate_full_data_file()
total_data_remove_zerorms = remove_rm_zeros(total_data)
total_data_rotate_acc = rotate_acc_in_dataframe2(total_data_remove_zerorms)
total_data_rotate = rotate_rtRate_in_dataframe2(total_data_rotate_acc) ##rotate the calib_rotation rate along x,y,z
### ------------- normalize the data and write it into csv file ---------------### 
normalized_data = generate_norm_data_with_label_uid(total_data_rotate,sensors_name_list,sensor_group_length_vector)
write_to_normalized = paste0("~/Dropbox/TravelData/data/iPhoneCollection_5sensors_normalized_rotated_",Sys.Date(),".csv")
print(paste0("Write the processed normalized data into file: ",write_to_normalized))
write.csv(normalized_data,write_to_normalized,row.names = F)
### --------------- write the processed normalized data ----------------- ### 
processed_unnormalized_5sensors_data = extract_cols_from_unnormalized_data(total_data_rotate,sensors_name_list,sensor_group_length_vector,iswinsorize = T)
write_to_5sensors_nonormalize = paste0("~/Dropbox/TravelData/data/iPhoneCollection_5sensors_raw_rotated_",Sys.Date(),".csv")
print(paste0("Write the processed raw data into file: ",write_to_5sensors_nonormalize))
write.csv(processed_unnormalized_5sensors_data,write_to_5sensors_nonormalize,row.names = F)

# ------------------------------ Test Code: define the value of test first ------------------------------------#
test = F
test_group_winsorize <-function()
{
  xx = data.frame(aa = seq(1,50),bb = seq(51,100),cc = seq(0.2,10,0.2),dd = seq(10.25,22.5,0.25),ee = seq(1,50))
  yy = winsorize_by_group(xx,c(2,2,1))
}

if(test)
{
  test_group_winsorize()
}

### decide whether need to reduce the sampling frequency:
if(test)
{
  f = data_files[52]
  slice_index2 = regexpr("_2016", f)
  slice_index = regexpr("freq", f)
  freq = substring(f,slice_index+4,slice_index2-1)
  temp = read.csv(paste0(data_path,f),skip = 1,header = F)
  if(freq == '16')
  {
    temp = process_sampling_frequency(temp)
  }
}


### test the data which contains more than one activity
if(test)
{
  f_name="UID_5FF08F2A-893E-4C2D-876C-7653FEA27B6A_freq5_20160809031445.csv"
  temp = read.csv(paste0(data_path,f_name),skip = 1,header = F)
  sample_new = preprocess_data(temp)
  sample_refined = remove_dummy_data(sample_new)
}
### test:
if(test)
{
  sample = read.csv(paste0(data_path,data_files[58]),skip = 1,header = F)
  sample_new = preprocess_data(sample)
  sample_refined = remove_dummy_data(sample_new)
  sample_refined$pressure = as.numeric(as.character(sample_refined$pressure))
  plot(sample_refined$pressure,type = 'l',col = 'red')
  par(new = T)
  plot(sample_refined$acc_y,col = 'blue',axes =F,xlab=NA, ylab=NA,type = 'l')
  axis(side = 4)
}

if(test)
{
  
  zz = rotate_acc_to_reference(test_data[100,]) 
  
  #sample_new is the one to use for test for batch rotation
  test_data = sample_new
  test_data = remove_rm_zeros(test_data)
  result = data.frame()
  #= mapply(rotate_acc_to_reference,test_data[,c('acc_x','acc_y','acc_z')],test_data[,paste0('rm',seq(1,9))])
  for(i in 1:nrow(test_data))
  {
    print(i)
    curr_result = rotate_acc_to_reference(test_data[i,c('acc_x','acc_y','acc_z')],test_data[i,paste0('rm',seq(1,9))])
    result = rbind.data.frame(result,curr_result)
  }
  names(result) = c("rotate_acc_x","rotate_acc_y","rotate_acc_z")
  test_data = cbind.data.frame(test_data,result)
}
