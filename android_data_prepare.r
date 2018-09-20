
#### Sep. 10th, 2016 ########## 
#### process android sensor readings:
#### 1. reading in data in each file
#### 2. select the rows that contain critical sensors 
#### 3. change the format from long shape into wide shape
#### 4. syncronize the data along timeline
####  - 4.0 For each tick, include: gravity,gyroscope,acc_linear,rotation_vector,light, pressure,magnetic
####  - 4.1 Fill in light data (sparse)
####  - 4.2 Fill in pressure data (sparse)
#### 5. rotate
#### 6. put all together and save it into a csv file. each row contains: 
####    [timestamp, v1,v2,v3,....vn, label, uid]

library(hash)

### scrach board 
### test before write into functions
set_global_variable<-function()
{
  uid_hash <<- hash()
  ## set location
  path <<- '~/Dropbox/TravelData/SensorWiseWinterData/winter'
  flist <<- list.files(path, pattern = '*.csv')
  essential_sensors <<- c("TYPE_LIGHT","TYPE_GYROSCOPE",
                          "TYPE_MAGNETIC_FIELD","TYPE_ROTATION_VECTOR","TYPE_PRESSURE",
                          "TYPE_GRAVITY","TYPE_LINEAR_ACCELERATION")
  
  motion_sensors <<- c("TYPE_GYROSCOPE","TYPE_ROTATION_VECTOR","TYPE_GRAVITY","TYPE_LINEAR_ACCELERATION")
  total_android_data <<- data.frame(timestamp = numeric(0),acc_x = numeric(0),acc_y= numeric(0),acc_z = numeric(0),
                                    rotation_rate_x = numeric(0),rotation_rate_y = numeric(0), rotation_rate_z = numeric(0),
                                    magnitude_acc = numeric(0),magnitude_mag = numeric(0), pressure = numeric(0),
                                    screen_brightness = numeric(0),label = character(0),uid = integer(0),stringsAsFactors = F)
  label_name <<- c('walk','jog','bike','bus','subway','car')
  ## numerical cols:
  num_cols <<- c('timestamp','acc_x','acc_y','acc_z',
               "rotation_rate_x","rotation_rate_y","rotation_rate_z",
               'magnitude_acc','magnitude_mag','pressure','screen_brightness')
  android_raw_winter <<- paste0('~/Dropbox/TravelData/android/android_raw_summer_synchronized_rotated_arbpm',Sys.Date(),'.csv')
  android_winter_synchronized_winsorized_normalized <<-paste0("~/Dropbox/TravelData/android/android_summer_synchronized_rotated_winsorized_normalized_arbpm",Sys.Date(),".csv")
  sensor_group_col_names <<- c('acc_x','acc_y','acc_z','rotation_rate_x','rotation_rate_y','rotation_rate_z','magnitude_acc','magnitude_mag','pressure','screen_brightness')
  sensor_group_length_vector <<- c(3,3,1,1,1,1)
}


get_name_from_fname <- function(x)
{
  slice_index = regexpr("_", x)[1]
  name = substring(x,1,slice_index-1)
  print("Finish getting the volunteer's name")
  name
}
get_uid <-function(name,uid_hash)
{
  if(! has.key(name,uid_hash))
  {
    uid_hash[[name]] = length(uid_hash) + 1
  }
  # print('Finish getting uid')
  uid_hash[[name]]
}

add_label <-function(orig_df)
{
  act = orig_df[which(orig_df$Name == 'ACT'),]
  critical_sensor_reading <- orig_df[which(orig_df$Name %in% essential_sensors),]
  print('Finish extracting data from essential sensors')
  critical_sensor_reading$label = NA
  if (nrow(act) == 1 )
  {
    critical_sensor_reading = critical_sensor_reading[1000:(nrow(critical_sensor_reading)-1000),]
    critical_sensor_reading$label = label_name[as.numeric(act[1,'Value'])]
  }
  else
  {
    for(i in 1:(nrow(act)-1) )
    {
      start = act[i,'TimeStamp']
      end = act[i+1,'TimeStamp']
      print(end-start)
      if (end - start > 2000)
      {
        critical_sensor_reading[which(critical_sensor_reading$TimeStamp < end &critical_sensor_reading$TimeStamp > start),'label'] = label_name[as.numeric(act[i,'Value'])]  
        # cut off the first second reading and last second reading (200 *5)
        critical_sensor_reading = critical_sensor_reading[-which(critical_sensor_reading$TimeStamp > start & critical_sensor_reading$TimeStamp <(start+1000)),]
        critical_sensor_reading = critical_sensor_reading[-which(critical_sensor_reading$TimeStamp >(end-1000) & critical_sensor_reading$TimeStamp < end),]
      }
      else
      {
        critical_sensor_reading = critical_sensor_reading[-which(critical_sensor_reading$TimeStamp > start & critical_sensor_reading$TimeStamp <end),]
      }
  }
    start = end
    critical_sensor_reading[which(critical_sensor_reading$TimeStamp > start),'label'] = label_name[as.numeric(act[i+1,'Value'])]
    ## cut the first and last second of readings
    critical_sensor_reading = critical_sensor_reading[-which(critical_sensor_reading$TimeStamp > start & critical_sensor_reading$TimeStamp < (start+1000)),]
    critical_sensor_reading = critical_sensor_reading[-which(critical_sensor_reading$TimeStamp >(max(critical_sensor_reading$TimeStamp)-1000)),]
    
  }
  print('finish adding label, removing the data from head and tail of each acc')
  critical_sensor_reading
}

convert_to_wide_shape<-function(long_df)
{
  ## 1. change it to factor
  #long_df$ValueName = as.factor(as.numeric(long_df$ValueName))
  #!duplicated(magnetic$timeStamps)
  long_df_remove_duplicated = long_df[-duplicated(long_df[,c('TimeStamp','ValueName','Name')]),]
  ## ? what is idvar? (can't be TimeStamp coz there're readings from different sensors but at same TimeStamp)
  wide_df = reshape(long_df_remove_duplicated,timevar = 'ValueName',v.names = 'Value',direction='wide',idvar = c('TimeStamp','Name'))
  
  wide_df = wide_df[complete.cases(wide_df),]
  wide_df
  
}
rotate_gravity <-function(gx,gy,gz,rx,ry,rz)
{
  g = c(gx,gy,gz)
  rm = calc_rotation_matrix(c(rx,ry,rz))
  ## note that for the rotation matrix, t(rm) = solve(rm). but to make it clear, we still use solve(rm)
  ## unit of acc and gravity is G (1G = -9.8m/s^2)
  g_observed = matrix(g,nrow = 1,ncol = 3)
  rotation_matrix = matrix(rm,nrow = 3,ncol = 3)
  ## g_observe %*% rotation_matrix = g_reference
  g_reference = g_observed %*% rotation_matrix
  as.vector(g_reference)
}

calc_rotation_matrix<-function(rv)
{
  rm = c()
  q1 = rv[1]
  q2 = rv[2]
  q3 = rv[3]
  q0 = 1 - q1**2 - q2**2 - q3**2
  q0 = ifelse(q0 > 0, sqrt(q0), 0)
  
  sq_q1 = 2 * q1 * q1
  sq_q2 = 2 * q2 * q2
  sq_q3 = 2 * q3 * q3
  q1_q2 = 2 * q1 * q2
  q3_q0 = 2 * q3 * q0
  q1_q3 = 2 * q1 * q3
  q2_q0 = 2 * q2 * q0
  q2_q3 = 2 * q2 * q3
  q1_q0 = 2 * q1 * q0
  
  rm = c(rm,(1 - sq_q2 - sq_q3))
  rm = c(rm,(q1_q2 - q3_q0))
  rm = c(rm, (q1_q3 + q2_q0))
  rm = c(rm,(q1_q2 + q3_q0))
  rm = c(rm,(1 - sq_q1 - sq_q3))
  rm = c(rm, (q2_q3 - q1_q0))
  rm = c(rm,(q1_q3 - q2_q0))
  rm = c(rm,(q2_q3 + q1_q0))
  rm = c(rm,(1 - sq_q1 - sq_q2))
  #   /  R[ 0]   R[ 1]   R[ 2]   \
  #   |  R[ 3]   R[ 4]   R[ 5]   |
  #   \  R[ 6]   R[ 7]   R[ 8]   /
  rm
}

synchronize_and_fill_data<-function(wide_df,uid)
{
  wide_df$Value.0 = as.numeric(wide_df$Value.0)
  wide_df$Value.1 = as.numeric(wide_df$Value.1)
  wide_df$Value.2 = as.numeric(wide_df$Value.2)
  ### idea: each tick (within 200ms) must contain 4 essential readings: gyroscope, acc, rotation_vector, gravity. 
  ### for missing light, pressure, mag, fill in with the average of adjucent 2 values
  ### for the readings that contains more than 1, only take the first one.
  ### how to define one tick. (where to begin, where to end?)
  #### for now assume gyroscope is always the first one. 
  ## 1. find the first row contains gyroscope reading:
  start = min(which(wide_df$Name == 'TYPE_GYROSCOPE'))
  ## set the first <current index>
  current_index = start
  ## prepare the first value of environment sensors (as back up to fill in at the begining:
  ### pressure
  pressure_first_index = min(which(wide_df$Name == 'TYPE_PRESSURE'))
  pressure_fill = wide_df[pressure_first_index,'Value.0']
  ### mag (need magnitude)
  mag_first_index = min(which(wide_df$Name == 'TYPE_MAGNETIC_FIELD'))
  mag_magnitude_fill = sqrt(wide_df[mag_first_index,'Value.0']**2
                            + wide_df[mag_first_index,'Value.1']**2
                            + wide_df[mag_first_index,'Value.2']**2)
  ### light
  light_first_index = min(which(wide_df$Name == 'TYPE_LIGHT'))
  light_fill = wide_df[light_first_index,'Value.0']
  
  ## initialize the dataframe with flat shape. (same as iphone data, except for the field 'light' which in iphone is Screen_Brightness)
  sync_readings = data.frame(timestamp = numeric(0),acc_x = numeric(0),acc_y= numeric(0),acc_z = numeric(0),
                             rotation_rate_x = numeric(0),rotation_rate_y = numeric(0), rotation_rate_z = numeric(0),
                             magnitude_acc = numeric(0),magnitude_mag = numeric(0), pressure = numeric(0),
                             screen_brightness = numeric(0),label = character(0),uid = integer(0),stringsAsFactors=F) 

  
  #a[nrow(a)+1,] <- c(5,6)
  while(current_index < nrow(wide_df)- 100)
  {
    #print(current_index)
    ## find the range of one tick:
    tick_start_timestamp = wide_df[current_index,'TimeStamp']
    tick_end_relative = max(min(which(wide_df[current_index:nrow(wide_df),'TimeStamp']- tick_start_timestamp > 99)), min(which(wide_df[current_index:nrow(wide_df),'Name'] == 'TYPE_GYROSCOPE')))
    tick_end = tick_end_relative + current_index -1
    
    current_label = wide_df[current_index,'label']
    ## check if the tick is valid: contain <acc,gyro, rotation_vector,gravity>
    valid = all(motion_sensors %in% wide_df[current_index:tick_end,'Name']) && (wide_df[tick_end,'label']==current_label )
    if(!'TYPE_GRAVITY' %in% wide_df[current_index:tick_end,'Name'])
    {
      print("missing gravity in current tick")
    }
    if(valid)## contain all motion sensors, put the data in data frame
    {
      ## rotation_vector 
      rotation_vector_index = current_index - 1 + min(which(wide_df[current_index:tick_end,'Name'] == 'TYPE_ROTATION_VECTOR'))
      rv = wide_df[rotation_vector_index,c('Value.0','Value.1',"Value.2")]
      ## gravity 
      gravity_index = current_index - 1 + min(which(wide_df[current_index:tick_end,'Name'] == "TYPE_GRAVITY"))
      gravity = wide_df[gravity_index,c('Value.0','Value.1',"Value.2")]
      ## acc
      acc_index = current_index - 1 +min(which(wide_df[current_index:tick_end,'Name'] == 'TYPE_LINEAR_ACCELERATION'))
      acc = wide_df[acc_index,c('Value.0','Value.1',"Value.2")]
      
      ## acc_rotate
      acc_rotate = rotate_gravity(acc$Value.0,acc$Value.1,acc$Value.2,rv$Value.0,rv$Value.1,rv$Value.2)
      ## acc_magnitude
      acc_mag = sqrt(acc_rotate[1]**2 + acc_rotate[2]**2 + acc_rotate[3]**2)
      ## gyro
      gyro_index = current_index - 1 +min(which(wide_df[current_index:tick_end,'Name'] == "TYPE_GYROSCOPE"))
      gyro = wide_df[gyro_index,c('Value.0','Value.1',"Value.2")]
      ## magnetic_magnitude
      mag_in_tick = current_index - 1 + which(wide_df[current_index:tick_end,'Name'] == 'TYPE_MAGNETIC_FIELD')
      if(length(mag_in_tick) == 0)
      {
        magnetic_magnitude = mag_magnitude_fill
      }
      else
      {
        mag_index = min(mag_in_tick)
        magnetic_magnitude = sqrt(wide_df[mag_index,'Value.0']**2 
                                  + wide_df[mag_index,'Value.1']**2
                                  + wide_df[mag_index,'Value.2']**2)
      }
      
      ## pressure
      pressure_in_tick = current_index - 1 + which(wide_df[current_index:tick_end,'Name'] == 'TYPE_PRESSURE')
      if(length(pressure_in_tick) == 0)
      {
        pressure = pressure_fill
      }
      else
      {
        pressure_index = min(pressure_in_tick)
        pressure = wide_df[pressure_index,'Value.0']
      }
      ## light
      light_in_tick = current_index - 1 + which(wide_df[current_index:tick_end,'Name'] == 'TYPE_LIGHT')
      if (length(light_in_tick) == 0)
      {
        light = light_fill
      }
      else
      {
        light_index = min(light_in_tick)
        light = wide_df[light_index,'Value.0']
      }
      
      ### - replace the fill with current value
      mag_magnitude_fill = magnetic_magnitude
      pressure_fill = pressure
      light_fill = light
      
      
      ### put the data in the record
      sync_readings[nrow(sync_readings)+1,] = c(tick_start_timestamp,acc_rotate[1],acc_rotate[2],acc_rotate[3],
                                                gyro$Value.0,gyro$Value.1,gyro$Value.2,acc_mag,magnetic_magnitude,pressure,
                                                light,current_label,uid)
      
      
    }
    ### go to next tick 
    ## if (at the case: label is not the same in current tick,then find the index, which is the new label and start from gyroscope)
    if(wide_df[tick_end,'label']!=current_label)
    {
      current_index = max(min(tick_end + which(wide_df[(tick_end+1):nrow(wide_df),'Name'] == 'TYPE_GYROSCOPE')), min(current_index - 1 + which(wide_df[(current_index+1):nrow(wide_df),'label']!=current_label)))
    }
    ## else: the rest case. regular, or current tick doesn't contain the motion sensors
    else
    {
      current_index = min(tick_end + which(wide_df[(tick_end+1):nrow(wide_df),'Name'] == 'TYPE_GYROSCOPE'))
    }
  }
  sync_readings
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

for(f in flist)
{
  print(f)
  name = get_name_from_fname(f)
  uid = get_uid(name,uid_hash)
}

generate_raw_rotated_synchronized_winter_android_data <-function()
{
  
  for(f in flist)
  {
    print(f)
    name = get_name_from_fname(f)
    uid = get_uid(name,uid_hash)
    d = read.csv(paste0(path,f),stringsAsFactors = F)
    print('finish reading data')
    labeled_df = add_label(d)
    c_wide_df = convert_to_wide_shape(labeled_df)
    print('nrow of wide_df:')
    print(nrow(c_wide_df))
    if(all(essential_sensors %in% unique(c_wide_df$Name)))
    {
      current_result = synchronize_and_fill_data(c_wide_df,uid)
      print(nrow(current_result))
      total_android_data = rbind.data.frame(total_android_data,current_result)
    }
    else
    {
      print(paste0("Data in ", f, " missing ", essential_sensors[which(!essential_sensors %in% unique(c_wide_df$Name))]))
    }
  }
  
  final_results_winter = total_android_data
  
  final_results_winter[,num_cols] = lapply(num_cols, function(x) as.numeric(final_results_winter[,x]))
  # final_results_winter$label = as.integer(final_results_winter$label)
  # final_results_winter$uid = as.integer(final_results_winter$uid)
  write.csv(final_results_winter,android_raw_winter,row.names = F)
  final_results_winter
}

generate_norm_data<-function(raw_rotated_data,isWinsorize)
{
  
  if(isWinsorize)
  {
    raw_rotated_data[,sensor_group_col_names] = winsorize_by_group(raw_rotated_data[,sensor_group_col_names],sensor_group_length_vector)
  }
  
  raw_rotated_data[,sensor_group_col_names] = normalize_by_group(raw_rotated_data[,sensor_group_col_names],sensor_group_length_vector)
  raw_rotated_data
  
}
########## !!!! Set Summer Winter Parameter for the file names ##############################
set_global_variable()
## get all data, synchronize, rotate.
final_results_winter = generate_raw_rotated_synchronized_winter_android_data()
## winsorize, normalize
norm_winsorized_rotated_data = generate_norm_data(final_results_winter,isWinsorize = T)
## map act num to real act type
# 1 walking
# 2 Running
# 3 Bicycling
# 4 Riding a bus
# 5 Using the subway
# 6 Driving or riding a car



norm_winsorized_rotated_data$uid = as.numeric(norm_winsorized_rotated_data$uid) + 30
## remove professor's he's subway data: might be typo/mistake (and too less)
norm_winsorized_rotated_data = norm_winsorized_rotated_data[-which(norm_winsorized_rotated_data$uid == 31 & norm_winsorized_rotated_data$label == 'subway'),]
## write to file.
write.csv(norm_winsorized_rotated_data,android_winter_synchronized_winsorized_normalized,row.names = F)

################ Test Section ##################
test_add_label <-function()
{
  d = read.csv(paste0(path,flist[34]),stringsAsFactors = F)
  labeled_df = add_label(d)
}
test_synchronize_and_fill_data <-function()
{
  uid = 33
  test = synchronize_and_fill_data(wide_df,uid)
}
test_convert_to_wide_shape<-function()
{
  wide_df = convert_to_wide_shape(labeled_df)
}
