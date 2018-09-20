file_dir = '~/Dropbox/TravelData/CrossPersonLearning/'

file_list = list.files(path = file_dir,pattern = "*csv")
f1 = file_list[1]  ##1-2000
f2 = file_list[11] ##2000 - 4000
f3 = file_list[13] ##4000 - 8000 
f4 = file_list[18] ##8000 - 12000
f5 = file_list[6]  ##12000 - 32000
f6 = file_list[12] ##32000 - 40000
f7 = file_list[14] ##40000 - 50000
f8 = file_list[15] ##50000 - 60000
f9 = file_list[16] ##60000 - 65000
f10 = file_list[17] ##65000 - 80000
f11 = file_list[19] ##80000 - 90000
f12 = file_list[20] ##90000 - 100000
f13 = file_list[2] ##100000 - 105000
f14 = file_list[3] ##105000 - 115000
f15 = file_list[4] ##115000 - 125000
f16 = file_list[7] ##125000 - 130000
f17 = file_list[8] ##130000 - 135000
f18 = file_list[9] ##135000 - 140000
f19 = file_list[10] ##140000 - end

file_order = c(1,11,13,18,6,12,14,15,16,17,19,20,2,3,4,7,8,9,10)
learning_results = data.frame()
for(i in file_order)
{
  fname = paste0(file_dir,file_list[i])
  print(fname)
  d = read.csv(fname,header = F)
  learning_results = rbind.data.frame(learning_results,d)
  
}

uids = paste0('usr',c(1,2,3,4,5,6,7,8,9,10,21,22,23,31,32,33,34,35))
names(learning_results) = uids
write.csv(learning_results,"~/Dropbox/TravelData/results/cross_person_mode_learning_results.csv",row.names = F)
# ----- Define a function for plotting a matrix ----- #
myImagePlot <- function(x, ...){
  min <- min(x)
  max <- max(x)
  yLabels <- rownames(x)
  xLabels <- colnames(x)
  title <-c()
  # check for additional function arguments
  if( length(list(...)) ){
    Lst <- list(...)
    if( !is.null(Lst$zlim) ){
      min <- Lst$zlim[1]
      max <- Lst$zlim[2]
    }
    if( !is.null(Lst$yLabels) ){
      yLabels <- c(Lst$yLabels)
    }
    if( !is.null(Lst$xLabels) ){
      xLabels <- c(Lst$xLabels)
    }
    if( !is.null(Lst$title) ){
      title <- Lst$title
    }
  }
  # check for null values
  if( is.null(xLabels) ){
    xLabels <- c(1:ncol(x))
  }
  if( is.null(yLabels) ){
    yLabels <- c(1:nrow(x))
  }
  
  layout(matrix(data=c(1,2), nrow=1, ncol=2), widths=c(4,1), heights=c(1,1))
  
  # Red and green range from 0 to 1 while Blue ranges from 1 to 0
  ColorRamp <- rgb( seq(0,1,length=256),  # Red
                    seq(0,1,length=256),  # Green
                    seq(1,0,length=256))  # Blue
  ColorLevels <- seq(min, max, length=length(ColorRamp))
  
  # Reverse Y axis
  reverse <- nrow(x) : 1
  yLabels <- yLabels[reverse]
  x <- x[reverse,]
  
  # Data Map
  par(mar = c(3,5,2.5,2))
  image(1:length(xLabels), 1:length(yLabels), t(x), col=ColorRamp, xlab="",
        ylab="", axes=FALSE, zlim=c(min,max))
  if( !is.null(title) ){
    title(main=title)
  }
  axis(BELOW<-1, at=1:length(xLabels), labels=xLabels, cex.axis=0.7)
  axis(LEFT <-2, at=1:length(yLabels), labels=yLabels, las= HORIZONTAL<-1,
       cex.axis=0.7)
  
  # Color Scale
  par(mar = c(3,2.5,2.5,2))
  image(1, ColorLevels,
        matrix(data=ColorLevels, ncol=length(ColorLevels),nrow=1),
        col=ColorRamp,
        xlab="",ylab="",
        xaxt="n")
  
  layout(1)
}
### Plot the matrix of learning results
myImagePlot(learning_results, xLabels=names(learning_results), title=c("UID_Mode_CrossLearning_Validation"), zlim=c(0,1)) 
### read in the usr-mode setting and put setting and learning results together
combinations = read.csv("~/Dropbox/TravelData/CrossPersonLearning/user_mode_cross_learning_modecombinations.txt",header = F)
names(combinations) = c('bike','car','subway','bus','walk','jog')
cross_person_mode_setting_results = cbind.data.frame(combinations,learning_results)
### check which setting gets the highest classification accuracy for each uid:
# task - uid-wise
top_index<-function(arr,top_n)
{
  arr_ordered = arr[order(arr,decreasing = T)]
  top_n_index = which(arr >= arr_ordered[top_n])
  top_n_index
}
## 1. For each uid (col), plot the acc distribution.

top_acc_modeComb_uid = data.frame()
par(mfrow = c(6,3))
par(mar = c(3.0,3.0,3.0,3.0),mgp = c(2,0.7,0))
for(uid in uids)
{
  print(uid)
  top_10_index = top_index(cross_person_mode_setting_results[[uid]],10)
  current_uid_top10 = cross_person_mode_setting_results[top_10_index,c(names(combinations),uid)]
  current_uid_top10$uid = uid
  names(current_uid_top10) = c(names(combinations),'acc','uid')
  top_acc_modeComb_uid = rbind.data.frame(top_acc_modeComb_uid,current_uid_top10)
  ## plot
  
  #plot(1:nrow(cross_person_mode_setting_results),cross_person_mode_setting_results[[uid]],type = 'l',xlab = 'Settings',ylab = 'Accuracy',main = uid,ylim = c(0,1))
}

## 2. for each uid, what's the highest top 10 settings

for(uid in uids)
{
  print(uid)
  current_uid_record = top_acc_modeComb_uid[which(top_acc_modeComb_uid$uid == uid),]
  for(mode in names(combinations))
  {
    print(mode)
    print(unique(current_uid_record[[mode]]))
  }
}
uid_top_acc_bike = matrix(0, nrow = length(uids), ncol = length(uids))
uid_top_acc_car = matrix(0, nrow = length(uids), ncol = length(uids))
uid_top_acc_subway = matrix(0, nrow = length(uids), ncol = length(uids))
uid_top_acc_bus = matrix(0, nrow = length(uids), ncol = length(uids))
uid_top_acc_walk = matrix(0, nrow = length(uids), ncol = length(uids))
uid_top_acc_jog = matrix(0, nrow = length(uids), ncol = length(uids))

### row: each uid, col: learn from which uid
for(i in 1: length(uids))
{
  current_uid_record = top_acc_modeComb_uid[which(top_acc_modeComb_uid$uid == uids[i]),]
  ## bike
  mode = 'bike'

  uids_for_training_for_current_mode = unique(current_uid_record[[mode]])
  for(j in uids_for_training_for_current_mode)
  {
    col_num = which(uids == paste0('usr',j))
    uid_top_acc_bike[i,col_num] = 1
  }
  ## car
  mode = 'car'
  uids_for_training_for_current_mode = unique(current_uid_record[[mode]])
  for(j in uids_for_training_for_current_mode)
  {
    col_num = which(uids == paste0('usr',j))
    uid_top_acc_car[i,col_num] = 1
  }
  ## subway
  mode = 'subway'

  uids_for_training_for_current_mode = unique(current_uid_record[[mode]])
  for(j in uids_for_training_for_current_mode)
  {
    col_num = which(uids == paste0('usr',j))
    uid_top_acc_subway[i,col_num] = 1
  }
  ## bus
  mode = 'bus'
  uids_for_training_for_current_mode = unique(current_uid_record[[mode]])
  for(j in uids_for_training_for_current_mode)
  {
    col_num = which(uids == paste0('usr',j))
    uid_top_acc_bus[i,col_num] = 1
  }
  ## walk
  mode = 'walk'
  record_matrix = uid_top_acc_bike
  uids_for_training_for_current_mode = unique(current_uid_record[[mode]])
  for(j in uids_for_training_for_current_mode)
  {
    col_num = which(uids == paste0('usr',j))
    uid_top_acc_walk[i,col_num] = 1
  }
  ## jog
  mode = 'jog'
  
  uids_for_training_for_current_mode = unique(current_uid_record[[mode]])
  for(j in uids_for_training_for_current_mode)
  {
    col_num = which(uids == paste0('usr',j))
    uid_top_acc_jog[i,col_num] = 1
  }
}

### then do the binary plot
### read in the usr-mode setting and put setting and learning results together
combinations = read.csv("~/Dropbox/TravelData/results/user_mode_cross_learning_modecombinations.txt",header = F)
cross_person_mode_setting_results_file = '~/Dropbox/TravelData/results/cross_person_mode_learning_results.csv'
learning_results = read.csv(cross_person_mode_setting_results_file)
uids = paste0('usr',c(1,2,3,4,5,6,7,8,9,10,21,22,23,31,32,33,34,35))
#names(learning_results) = uids
names(combinations) = c('bike','car','subway','bus','walk','jog')
cross_person_mode_setting_results = cbind.data.frame(combinations,learning_results)

### check which setting gets the highest classification accuracy for each uid:
# task - uid-wise
top_index<-function(arr,top_n)
{
  arr_ordered = arr[order(arr,decreasing = T)]
  top_n_index = which(arr >= arr_ordered[top_n])
  top_n_index
}
## 1. For each uid (col), plot the acc distribution.
## 2. for each uid, what's the highest top 10 settings
top_acc_modeComb_uid = data.frame()
par(mfrow = c(6,3))
par(mar = c(3.0,3.0,3.0,3.0))
for(uid in uids)
{
  print(uid)
  top_10_index = top_index(cross_person_mode_setting_results[[uid]],10)
  current_uid_top10 = cross_person_mode_setting_results[top_10_index,c(names(combinations),uid)]
  current_uid_top10$uid = uid
  names(current_uid_top10) = c(names(combinations),'acc','uid')
  top_acc_modeComb_uid = rbind.data.frame(top_acc_modeComb_uid,current_uid_top10)
  ## plot
  
  #plot(1:nrow(cross_person_mode_setting_results),cross_person_mode_setting_results[[uid]],type = 'l',xlab = 'Settings',ylab = 'Accuracy',main = uid,ylim = c(0,1))
}

### usr1:
### female, 
usr1_mode = c('car','subway','walk')
usr_1 = top_acc_modeComb_uid[which(top_acc_modeComb_uid$uid == 'usr1'),c(usr1_mode,'uid')]
for(mode in usr1_mode)
{
  print(mode)
  print(unique(usr_1[[mode]]))
}

### usr5:
#### female

print_similarity<-function(uid,uid_mode)
{
  usr_info = top_acc_modeComb_uid[which(top_acc_modeComb_uid$uid == uid),c(uid_mode,'acc','uid')]
  for(mode in uid_mode)
  {
    print(mode)
    print(unique(usr_info[[mode]]))
  }
}
usr5_mode = c('bike','car','subway','walk','jog','bus')
print_similarity('usr5',usr5_mode)
usr8_mode = c('car','subway','walk')
print_similarity('usr8',usr8_mode)
usr1_mode = c('car','subway','walk')
print_similarity('usr1',usr1_mode)

usr31_mode = c('car','walk','jog')
print_similarity('usr31',usr31_mode)

usr22_mode = c('bike','car','subway')
print_similarity('usr22',usr22_mode)

usr9_mode = c('subway','walk')
print_similarity('usr9',usr9_mode)

usr10_mode = c('car','subway')
print_similarity('usr10',usr9_mode)

usr2_mode = c('car','walk')
print_similarity('usr2',usr2_mode)


### try to see the similarity between car 1 and car 8, (bike 32, 22)
### walk (1, 3) (5,8,9)
### subway  (5,9)

iphone_data_file = 'U:\\TravelData\\iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
android_summer_data_file = 'U:\\TravelData\\Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
android_winter_data_file = 'U:\\TravelData\\Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv' 
normalized_ipohne_data = 'U:\\TravelData\\iPhoneCollection_5sensors_normalized_rotated_2016-08-20.csv'
raw_iphone_data = 'U:\\TravelData\\iPhoneCollection_5sensors_raw_rotated_2016-08-21.csv'

iphone_fv = read.csv(iphone_data_file)
android_winter_fv = read.csv(android_winter_data_file)
android_summer_fv = read.csv(android_summer_data_file)
raw_iphone = read.csv(raw_iphone_data)
normalized_iphone = read.csv(normalized_ipohne_data)




### check uid 1 and uid 8 -- car
uid_1_car_raw = raw_iphone[which(raw_iphone$uid == 1 & raw_iphone$label == 'car'),]
uid_8_car_raw = raw_iphone[which(raw_iphone$uid == 8 & raw_iphone$label == 'car'),]
uid_5_car_raw = raw_iphone[which(raw_iphone$uid == 5 & raw_iphone$label == 'car'),]
accx5 = uid_5_car_raw[1:10000,'acc_x']
accx1 = uid_1_car_raw[1:10000,'acc_x']
accx8 = uid_8_car_raw[1:10000,'acc_x']
nns = names(uid_1_car_raw)
cor_parwise_car = data.frame()
compare_two_uid_cor <-function(u1,u2,mode)
{
  u1_raw = normalized_iphone[which(normalized_iphone$uid == u1 &normalized_iphone$label == mode ),]
  u2_raw = normalized_iphone[which(normalized_iphone$uid == u2 &normalized_iphone$label == mode ),]
  max_length = min(nrow(u1_raw),nrow(u2_raw))
  cor_row = c()
  for (name in nns[2:11])
  {
    print(name)
    u1_raw_current_col = u1_raw[1:max_length,name]
    u2_raw_current_col = u2_raw[1:max_length,name]
    cor_row = c(cor_row,var(u1_raw_current_col,u2_raw_current_col))
  }
  names(cor_row) = c(nns[2:11])
  cor_row
  
}
cor_15 = compare_two_uid_cor('1','5','car')
cor_18 = compare_two_uid_cor('1','8','car')
cor_58 = compare_two_uid_cor('5','8','car')



short_names = c('accx','accy','accz','rrx','rry','rrz','accm','magm','p','bri')
par(mar=c(4,3,3,3))

col_base = 15
plot.new()
for(i in 1:10)
{
  for (j in 1:10)
  {
    cor = compare_two_uid_cor(i,j,'car')
    par(new = TRUE)
    plot(cor,type = 'l', xlab="", ylab="",xaxt="n",ylim=c(-0.5,0.5),col = col_base)
    col_base = col_base + 10
  }
}
axis(1, at=1:10, labels =short_names)

plot(cor_58,type = 'l', xlab="", ylab="",xaxt="n",ylim=c(-0.5,0.5))
axis(1, at=1:10, labels =short_names)
par(new = TRUE)
plot(cor_18,type = 'l', xlab="", ylab="",xaxt="n",col = 'red',ylim=c(-0.5,0.5))
axis(1, at=1:10, labels =short_names)
par(new = TRUE)
plot(cor_15,type = 'l', xlab="", ylab="",xaxt="n",col = 'blue',ylim=c(-0.5,0.5))
axis(1, at=1:10, labels =short_names)

par(mfrow = c(2,1))
plot(uid_1_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'red')
par(new = T)
plot(uid_8_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'blue')

plot(uid_1_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'red')
par(new = T)
plot(uid_5_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'blue')

plot(uid_1_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'red')
par(new = T)
plot(uid_8_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'blue')

plot(uid_1_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'red')
par(new = T)
plot(uid_5_car_raw$magnitude_acc,type = 'l',xlim = c(0,60000),ylim=c(0,1),col = 'blue')


#### Try clustering first:
fspace = rbind.data.frame(iphone_fv,android_summer_fv,android_winter_fv)
## step 1: standardize 
fspace_scaled = scale(fspace[,names(fspace)[1:161]])

# check that we get mean of 0 and sd of 1
colMeans(fspace_scaled)  # faster version of apply(scaled.dat, 2, mean)
apply(fspace_scaled, 2, sd)

fspace_scaled$UID = fspace$UID
fspace_scaled$label = fspace$class


kmean_results = kmeans(fspace_scaled, 30, iter.max = 20, nstart = 1,
       algorithm = c("Hartigan-Wong", "Lloyd", "Forgy",
                     "MacQueen"), trace=FALSE)

kmean_results_uid_label = data.frame(kmean = unlist(kmean_results$cluster),UID = fspace$UID,mode = fspace$class)




kmean_results_5 = kmeans(fspace_scaled, 5, iter.max = 20, nstart = 1,
                       algorithm = c("Hartigan-Wong", "Lloyd", "Forgy",
                                     "MacQueen"), trace=FALSE)
kmean_results5_uid_label = data.frame(kmean = unlist(kmean_results_5$cluster),UID = fspace$UID,mode = fspace$class)

kmean_results_6 = kmeans(fspace_scaled[which(fspace$class %in% c('walk','jog')),1:84], 6, iter.max = 20, nstart = 1,
                         algorithm = c("Hartigan-Wong", "Lloyd", "Forgy",
                                       "MacQueen"), trace=FALSE)

kmean_mode = data.frame(kmean_results_6$cluster,fspace[which(fspace$class %in% c('walk','jog')),c('UID')])


##### Begining of similarity Score calculation (User vs User)

#### Similarity Calculation.
iphone_data_file = '~/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
android_summer_data_file = '~/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
android_winter_data_file = '~/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv' 
normalized_ipohne_data = '~/Dropbox/TravelData/data/iPhoneCollection_5sensors_normalized_rotated_2016-08-20.csv'
raw_iphone_data = '~/Dropbox/TravelData/data/iPhoneCollection_5sensors_raw_rotated_2016-08-21.csv'

iphone_fv = read.csv(iphone_data_file)
android_winter_fv = read.csv(android_winter_data_file)
android_summer_fv = read.csv(android_summer_data_file)
raw_iphone = read.csv(raw_iphone_data)
normalized_iphone = read.csv(normalized_ipohne_data)






fspace = rbind.data.frame(iphone_fv,android_summer_fv,android_winter_fv)
fspace[which(fspace$UID == 4),'UID'] = 1
fspace[which(fspace$UID == 6),'UID'] = 2
fspace[which(fspace$UID == 7),'UID'] = 3

## step 1: standardize 
fspace_scaled = scale(fspace[,names(fspace)[1:161]])
modes =  sort(unique(fspace$class))

uids = as.character(sort(unique(fspace$UID)))
uid_mode = table(fspace$UID, fspace$class)
## step 2, build the data center profile for each user.
### the data frame is a shape of [uid, mode, <feature centers>]
data_center_profiles = data.frame(matrix(ncol = 163, nrow = 0))
names(data_center_profiles) = c('uid','mode',names(fspace)[1:161])



for(uid in uids)
{
  for(mode in modes)
  {
    print(paste(uid,mode))
    if(uid_mode[uid,mode] >0)
    {
      ##calculate the center of current user at current mode
      data_space = fspace_scaled[which(fspace$UID == uid & fspace$class ==mode),]
      data_space_center = apply(data_space,2, mean)
      data_center_profiles[nrow(data_center_profiles)+1,] <- c(uid,mode,data_space_center)
    }
  }
}




###Step3: build the non-data factor matrix 
user_profile = data.frame(uid = character(0),gender = character(0),city = character(0),phone_model = character(0),season = character(0),stringsAsFactors=FALSE)
user_profile[nrow(user_profile)+1,] = c('1','f','ny','ip','s')
user_profile[nrow(user_profile)+1,] = c('2','f','bl','ip','s')
user_profile[nrow(user_profile)+1,] = c('3','f','ny','ip','s')
#user_profile[nrow(user_profile)+1,] = c('4','m','ny','ip','s')
user_profile[nrow(user_profile)+1,] = c('5','f','ny','ip','s')
#user_profile[nrow(user_profile)+1,] = c('6','f','ny','ip','s')
#user_profile[nrow(user_profile)+1,] = c('7','m','ny','ip','s')
user_profile[nrow(user_profile)+1,] = c('8','f','ny','ip','s')
user_profile[nrow(user_profile)+1,] = c('9','f','ny','ip','s')
user_profile[nrow(user_profile)+1,] = c('10','f','ny','ip','s')
user_profile[nrow(user_profile)+1,] = c('21','m','bl','a','w')
user_profile[nrow(user_profile)+1,] = c('22','m','bl','a','w')
user_profile[nrow(user_profile)+1,] = c('23','m','bl','a','w')
user_profile[nrow(user_profile)+1,] = c('31','m','bl','a','s')
user_profile[nrow(user_profile)+1,] = c('32','m','bl','a','s')
user_profile[nrow(user_profile)+1,] = c('33','f','bl','a','s')
user_profile[nrow(user_profile)+1,] = c('34','m','bl','a','s')
user_profile[nrow(user_profile)+1,] = c('35','m','bl','a','s')


##step 4: calculate the non-data factor matrix
A = matrix(rep(0,length(uids)**2),ncol = length(uids),nrow = length(uids))
rownames(A) = uids
colnames(A) = uids
## define \xi = 0.3
xi = 0.2
non_data_factorization <-function(u1_val,u2_val)
{
  print(u1_val)
  print(u2_val)
  if(u1_val == u2_val)
  {
    
    1-xi
  }
  else
  {
    xi
  }
}


for(u1 in uids)
{
  for(u2 in uids)
  {
    print(u1)
    print(u2)
    factor_score = sum(mapply(non_data_factorization,user_profile[which(user_profile$uid == u1),],user_profile[which(user_profile$uid == u2),]))
    A[as.character(u1),as.character(u2)] = factor_score
  }
}

data_center_profiles = read.csv("/Users/Xing/Dropbox/TravelData/data/data_center_profiles.csv")

## step 5. calculate the similarity score matrix
ss = matrix(rep(0,length(uid)**2),ncol = length(uids),nrow = length(uids))
rownames(ss) = uids
colnames(ss) = uids
for(u1 in uids)
{
  for(u2 in uids)
  {
    if(u1 != u2)
    {
      print(paste(u1,',',u2))
      ## calculate the average distance within common modes they have
      common_modes = which( (uid_mode[u1,] >0) & (uid_mode[u2,]>0))
      print('common modes:')
      print(common_modes)
      sum_distance = 0
      for(m in common_modes)
      {
        print(paste('m=',m))
        print("current mode:")
        print(modes[m])
        data_center_current_mode_u1 =as.numeric(as.vector(data_center_profiles[which(data_center_profiles$uid == u1 & data_center_profiles$mode == modes[m]), names(fspace)[1:161]]))
        
        data_center_current_mode_u2 =as.numeric(as.vector(data_center_profiles[which(data_center_profiles$uid == u2 & data_center_profiles$mode == modes[m]), names(fspace)[1:161]]))
        
        distance_u1_u2_m = sqrt(sum((data_center_current_mode_u1-data_center_current_mode_u2)**2))
        print(distance_u1_u2_m)
        sum_distance = sum_distance + distance_u1_u2_m
        
      }
      avg_distance = sum_distance/length(common_modes)
      score_u1_u2 = A[u1,u2]/avg_distance
      ss[u1,u2] = score_u1_u2
    }
    
      
  }
}



write.csv(ss,'user_similarity_scores_05312017.csv')
ss_selective = ss[c(1,2,3,4,6,7,8,9,10,12,13,14),c(1,2,3,4,6,7,8,9,10,12,13,14)]
write.csv(ss_selective,'user_similarity_scores_05312017_icdm.csv')
## for icdm 2017 keep ids: 1,2,3,5,8,9,10,21,22,31,32,33


###### Here we begin the similarity computation based on each mode (of each user)
## data_center_profiles has the information of uid-mode center (already merged)
## user_profile has the user's profile (ID, gender, city, os, season)

## begin merge_minor: merge data data from those UIDs which only have one single mode to other people's data pool
## 4 -> 1
## 6 -> 2
## 7 -> 3
## data_center_profiles already merged

data_center_profiles = read.csv("/Users/Xing/Dropbox/TravelData/data/data_center_profiles.csv")
uid_mode = table(data_center_profiles$uid, data_center_profiles$mode)
data_center_profiles[,3:ncol(data_center_profiles)] = sapply(data_center_profiles[,3:ncol(data_center_profiles)],MARGIN = c(1,2),FUN = as.numeric)
uids = unique(data_center_profiles$uid)
modes =  sort(unique(data_center_profiles$mode))


xi = 0.3
A = matrix(rep(0,length(uids)**2),ncol = length(uids),nrow = length(uids))
rownames(A) = uids
colnames(A) = uids
## define \xi = 0.3
non_data_factorization <-function(u1_val,u2_val)
{
  print(u1_val)
  print(u2_val)
  if(u1_val == u2_val)
  {
    
    1-xi
  }
  else
  {
    xi
  }
}


for(u1 in uids)
{
  for(u2 in uids)
  {
    print(u1)
    print(u2)
    factor_score = sum(mapply(non_data_factorization,user_profile[which(user_profile$uid == u1),],user_profile[which(user_profile$uid == u2),]))
    A[as.character(u1),as.character(u2)] = factor_score
  }
}

for(u1 in uids)
{

  xi_text = paste0("xi0",as.character(xi*10))
  file_name = paste0('usr_',u1,'_mode_similarity_',xi_text,'.csv')
  print(file_name)
  similarity_df = data.frame(usr = uids[-which(uids==u1)],car = NA,bus =NA,subway = NA,bike = NA,walk = NA,jog = NA,stringsAsFactors=FALSE)
  
  
  for(u2 in uids[-which(uids==u1)])
  {

    common_modes = which( (uid_mode[as.character(u1),] >0) & (uid_mode[as.character(u2),]>0))
    for(m in common_modes)
    {
      
      data_center_current_mode_u1 =as.numeric(as.vector(data_center_profiles[which(data_center_profiles$uid == u1 & data_center_profiles$mode == modes[m]), names(data_center_profiles)[3:163]]))
      
      data_center_current_mode_u2 =as.numeric(as.vector(data_center_profiles[which(data_center_profiles$uid == u2 & data_center_profiles$mode == modes[m]), names(data_center_profiles)[3:163]]))
      
      
      distance_u1_u2_m = sqrt(sum((data_center_current_mode_u1-data_center_current_mode_u2)**2))
      if(modes[m] == 'car' & u2 == 2)
      {
        print(data_center_current_mode_u2)
        print(distance_u1_u2_m)
      }
      
      similarity_u1_u2 = A[as.character(u1),as.character(u2)]/distance_u1_u2_m
      
      similarity_df[which(similarity_df$usr == u2),as.character(modes[m])] = similarity_u1_u2
    }
  }

  write.csv(similarity_df,file = paste0("/Users/Xing/Dropbox/TravelData/data/",file_name),row.names = F)
}

## check the variance of the each mode to see whether it is related to the learning accuracy:
data_variance_profiles = data.frame(matrix(ncol = 163, nrow = 0))
names(data_variance_profiles) = c('uid','mode',names(fspace)[1:161])


for(uid in uids)
{
  for(mode in modes)
  {
    print(paste(uid,mode))
    if(uid_mode[uid,mode] >0)
    {
      ##calculate the center of current user at current mode
      data_space = fspace_scaled[which(fspace$UID == uid & fspace$class ==mode),]
      data_variance = apply(data_space,2, var)
      data_variance_profiles[nrow(data_variance_profiles)+1,] <- c(uid,mode,data_variance)
    }
  }
}

squared_sum = function(a)
{
  sqrt(sum(a))
}


data_variance_profiles[,3:163] = lapply(data_variance_profiles[,3:163],as.numeric)


data_variance_profiles$dev = apply(data_variance_profiles[,3:163],1,squared_sum)
write.csv(data_variance_profiles,'~/Dropbox/TravelData/data/data_variance_profile.csv',row.names = F)


##### Notes:
usr1_similarity_file = paste0('~/Dropbox/TravelData/data/usr_1_mode_similarity_xi03.csv')
usr1_similarity = read.csv(usr1_similarity_file)


variance_profile = read.csv('~/Dropbox/TravelData/data/data_variance_profile.csv')
