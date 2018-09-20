
## calc_weight is a function to calculate the loss between two measure of dataset with weight mechanism applied.
cal_weight <-function(sr,tg,eta = 0.001)
{
  tg_measure = set_measure(tg,sr,rep(1,nrow(tg))) ## the first component of the objective function, weight is 1 (no weight)
  print('target data measure:')
  print(tg_measure)
  beta = rep(1,nrow(sr))
  sr_measure = set_measure(sr,tg,beta)
  print("source data measure:")
  print(sr_measure)
  loss = abs(tg_measure - sr_measure)
  for(i in 1:nrow(sr))
  {
    
    betai_list = c(1)
    lossi_list = c(loss)
    plot_file_name = paste0("beta_",i," change_vs_loss.png")
    ub_beta = Inf
    lb_beta = 0
    decrease = TRUE ## default: increase beta
    d_beta = 0.1
    while(loss >= eta && beta[i] > 0 && d_beta >0.01)
    {
      if(decrease)
      {
        print('decrease')
        lb_beta = beta[i]
        if(beta[i] + d_beta >= ub_beta)
        {
          d_beta = d_beta/2
        }
        if(beta[i] + 2*d_beta < ub_beta)
        {
          d_beta = d_beta * 2
        }
        print(paste('d_beta=',d_beta))
        beta[i] = beta[i] + d_beta
        
      }
      
      else
      {
        print('increase')
        ub_beta = beta[i]
        if(beta[i] - d_beta <= lb_beta)
        {
          d_beta = d_beta/2
        }
        if(beta[i] - 2*d_beta > lb_beta)
        {
          d_beta = d_beta *2
        }
        print(paste('d_beta=',d_beta))
        beta[i] = beta[i] - d_beta
      }
      sr_measure = set_measure(sr,tg,beta)
      new_loss = abs(tg_measure - sr_measure)
      decrease = loss > new_loss
      if(decrease)
      {
        loss = new_loss
      }
      print(paste0('loss:',loss))
      betai_list = c(betai_list,beta[i])
      lossi_list = c(lossi_list,loss)
    }
    beta[i] = lb_beta
    png(file=plot_file_name, width = 895, height = 855, units = "px")
    plot(x = sort(betai_list), y = lossi_list[order(betai_list)],main = paste0('Beta_',i," value VS loss"),xlab = "beta",ylab = 'loss',type ='l')
    dev.off()
  } 
  
  beta 
}

### if we ignore S and think: support of T is contained in support of D, 
calc_absolute_distance<-function(sr,tg,c)
{
  t_v_ij_matrix = calc_v_matrix(tg,c)
  t_normalizer = 1.0/sum(t_v_ij_matrix)
  ##calculate the measure of T:
  t_px = t_normalizer * colSums(t_v_ij_matrix) # each column adds up to the value of px_i
  t_normx =  apply(tg, 1, function(x) norm(as.matrix(x),type='f'))
  t_measure = t_normx %*% t_px
  
  
  ##calculate the mean distance between T and D: 
  s = rbind.data.frame(sr,tg)
  dist_s = as.matrix(dist(s)) 
  ### select distance between di - sj, remove t
  dist_s = dist_s[(1+nrow(sr)):nrow(s),1:nrow(sr)] ## column i is the distance of x_i to others in s
  dt_v_ij_matrix = 1/(dist_s+1.0/c)
  ## normx is the row norm of x. 
  normx_d =  apply(sr, 1, function(x) norm(as.matrix(x),type='f'))
  mean_distance = normx_d %*% colSums(dt_v_ij_matrix)
  
  ## minimizer:
  loss = abs(t_measure - mean_distance)
  loss
}


test_calc_mean_distance<-function()
{
  iphone_data_file = '/Users/Xing/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
  android_summer_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
  android_winter_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv' 
  normalized_ipohne_data = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_normalized_rotated_2016-08-20.csv'
  raw_iphone_data = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_raw_rotated_2016-08-21.csv'
  
  iphone_fv = read.csv(iphone_data_file)
  android_winter_fv = read.csv(android_winter_data_file)
  android_summer_fv = read.csv(android_summer_data_file)
  raw_iphone = read.csv(raw_iphone_data)
  normalized_iphone = read.csv(normalized_ipohne_data)
  
  ## step2: normalize the data
  merge_minor = T
  
  fspace = rbind.data.frame(iphone_fv,android_summer_fv,android_winter_fv)
  if(merge_minor)
  {
    fspace[which(fspace$UID == '4'),c('UID')] = 1
    fspace[which(fspace$UID == '6'),c('UID')] = 2
    fspace[which(fspace$UID == '7'),c('UID')] = 3
  }
  ## step 1: standardize 
  fspace_scaled = scale(fspace[,names(fspace)[1:161]])
  modes =  sort(unique(fspace$class))
  uids = as.character(sort(unique(fspace$UID)))
  ## try one mode: car
  test_mode = 'car'
  target_uid = '2'
  source_uid = '1'
  
  sr = fspace_scaled[which(fspace$UID == target_uid & fspace$class ==test_mode),]
  tg = fspace_scaled[which(fspace$UID == source_uid & fspace$class ==test_mode),]
  
  t_v_ij_matrix = calc_v_matrix(tg,c)
  t_normalizer = 1.0/sum(t_v_ij_matrix)
  ##calculate the measure of T:
  t_px = t_normalizer * colSums(t_v_ij_matrix) # each column adds up to the value of px_i
  t_normx =  apply(tg, 1, function(x) norm(as.matrix(x),type='f'))
  t_measure = t_normx %*% t_px
  
  
  ##calculate the mean distance between T and D: 
  s = rbind.data.frame(sr,tg)
  dist_s = as.matrix(dist(s)) 
  ### select distance between di - sj, remove t
  dist_s = dist_s[(1+nrow(sr)):nrow(s),1:nrow(sr)] ## column i is the distance of x_i to others in s
  dt_v_ij_matrix = 1/(dist_s+1.0/c)
  ## normx is the row norm of x. 
  normx_d =  apply(sr, 1, function(x) norm(as.matrix(x),type='f'))
  mean_distance = normx_d %*% colSums(dt_v_ij_matrix)
  
  ## minimizer:
  loss = abs(t_measure - mean_distance)
  
}

calc_mean_distance_rank<-function()
{
  iphone_data_file = '/Users/Xing/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
  android_summer_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
  android_winter_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv' 
  normalized_ipohne_data = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_normalized_rotated_2016-08-20.csv'
  raw_iphone_data = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_raw_rotated_2016-08-21.csv'
  
  iphone_fv = read.csv(iphone_data_file)
  android_winter_fv = read.csv(android_winter_data_file)
  android_summer_fv = read.csv(android_summer_data_file)
  raw_iphone = read.csv(raw_iphone_data)
  normalized_iphone = read.csv(normalized_ipohne_data)
  uid_mode = table(fspace$UID, fspace$class)
  ## step2: normalize the data
  merge_minor = T
  
  fspace = rbind.data.frame(iphone_fv,android_summer_fv,android_winter_fv)
  if(merge_minor)
  {
    fspace[which(fspace$UID == '4'),c('UID')] = 1
    fspace[which(fspace$UID == '6'),c('UID')] = 2
    fspace[which(fspace$UID == '7'),c('UID')] = 3
  }
  ## step 1: standardize 
  fspace_scaled = data.frame(scale(fspace[,names(fspace)[1:161]]))
  modes =  sort(unique(fspace$class))
  uids = as.character(sort(unique(fspace$UID)))
  
  c = 1.2
  ss = matrix(rep(0,length(uids)**2),ncol = length(uids),nrow = length(uids))
  rownames(ss) = paste0('su',uids) ##source uid: row
  colnames(ss) = paste0('tu',uids) ##target uid: col
  for(u1 in uids)
  {
    for(u2 in uids)
    {
      if(u1 != u2)
      {
        print(paste(u1,',',u2))
        ## calculate the average distance within common modes they have
        common_modes = which((uid_mode[u1,] >0) & (uid_mode[u2,]>0))
        print(common_modes)
        sum_mean_distance = 0
        if(length(common_modes) == 0)
        {
          ss[which(uids==u1),which(uids==u2)] = Inf
        }
        else
        {
          for(m in common_modes)
          {
            sr = fspace_scaled[which(fspace$UID == u1 & fspace$class ==modes[m]),]
            tg = fspace_scaled[which(fspace$UID == u2 & fspace$class ==modes[m]),]
            ##loss = calc_absolute_distance(sr,tg,c) ##v1
            loss = loss_in_t_learn_from_d(sr,tg,c) ##v2
            sum_mean_distance = sum_mean_distance + loss/(nrow(sr) + nrow(tg))
          }
          ss[which(uids==u1),which(uids==u2)] = sum_mean_distance
        }
        
      }
    }
  }
  write.csv(ss,paste0('weighted_loss_version2_c',c,'.csv'))
  ss
}



calculate_weight<-function()
{
  for(u in uids)
  {
    fspace_scaled[,paste0('weight_learnfrom_u',u)] <- NA
  }
  
  
  for(su in uids)
  {
    col_name = paste0('weight_learnfrom_u',su)
    ss = read.csv("~/Dropbox/TravelData/src/weighted_loss_version2_c1.2.csv",header = T,row.names = 1)
    for(tu in uids)
    {
      
      if(su != tu)
      {
        common_modes = which((uid_mode[su,] >0) & (uid_mode[tu,]>0))
        if(length(common_modes) > 0)
        {
          for(m in common_modes)
          {
            sr = fspace_scaled[which(fspace$UID == su & fspace$class ==modes[m]),]
            tg = fspace_scaled[which(fspace$UID == tu & fspace$class ==modes[m]),]
            ##loss = calc_absolute_distance(sr,tg,c) ##v1
            fspace_scaled[row.names(tg),c(col_name)] = t_weighted_learn_from_d(sr[,1:161],tg[,1:161],c)
          }
        }
      }
    }
  }
  write.csv(fspace_scaled,paste0('allphones_allmode_segLen16_noSlide_arbpm_normalize_uid_with_learning_weight_c',c,'_2016-12_24.csv'))
  fspace_scaled
}


test_calc_mean_distance<-function()
{
  ss = matrix(rep(0,length(uids)**2),ncol = length(uids),nrow = length(uids))
  rownames(ss) = paste0('su',uids) ##source uid: row
  colnames(ss) = paste0('tu',uids) ##target uid: col
  u1  = '1'
  u2 = '2'
  print(paste(u1,',',u2))
  ## calculate the average distance within common modes they have
  common_modes = which((uid_mode[u1,] >0) & (uid_mode[u2,]>0))
  print(common_modes)
  sum_mean_distance = 0
  for(mode in common_modes)
  {
    sr = fspace_scaled[which(fspace$UID == u1 & fspace$class ==mode),]
    tg = fspace_scaled[which(fspace$UID == u2 & fspace$class ==mode),]
    mean_distance = calc_mean_distance(sr,tg,c=1.2)
    sum_mean_distance = sum_mean_distance + mean_distance/(nrow(sr) + nrow(tg))
  }
  ss[as.integer(u1),as.integer(u2)] = sum_mean_distance
}

test_cal_weight<-function()
{
  ## step 1. get the data:
  iphone_data_file = '/Users/Xing/Dropbox/TravelData/data/iPhone_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv'
  android_summer_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_summer_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-13.csv'
  android_winter_data_file = '/Users/Xing/Dropbox/TravelData/data/Android_winter_allmode_segLen16_noSlide_arbpm_normalize_uid_2016-09-12.csv' 
  normalized_ipohne_data = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_normalized_rotated_2016-08-20.csv'
  raw_iphone_data = '/Users/Xing/Dropbox/TravelData/data/iPhoneCollection_5sensors_raw_rotated_2016-08-21.csv'
  
  iphone_fv = read.csv(iphone_data_file)
  android_winter_fv = read.csv(android_winter_data_file)
  android_summer_fv = read.csv(android_summer_data_file)
  raw_iphone = read.csv(raw_iphone_data)
  normalized_iphone = read.csv(normalized_ipohne_data)
  
  ## step2: normalize the data
  merge_minor = T
  
  fspace = rbind.data.frame(iphone_fv,android_summer_fv,android_winter_fv)
  if(merge_minor)
  {
    fspace[which(fspace$UID == '4'),c('UID')] = 1
    fspace[which(fspace$UID == '6'),c('UID')] = 2
    fspace[which(fspace$UID == '7'),c('UID')] = 3
  }
  ## step 1: standardize 
  fspace_scaled = scale(fspace[,names(fspace)[1:161]])
  modes =  sort(unique(fspace$class))
  uids = as.character(sort(unique(fspace$UID)))
  ## try one mode: car
  test_mode = 'car'
  
  for(target_uid in uids)
  {
    print(paste("target_uid:",target_uid))
    for(source_uid in uids[-which(target_uid==uids)])
    {
      print(paste("source_uid:",source_uid))
      tg = fspace_scaled[which(fspace$UID == target_uid & fspace$class ==test_mode),]
      sr = fspace_scaled[which(fspace$UID == source_uid & fspace$class ==test_mode),]
      if(nrow(tg) != 0 && nrow(sr) != 0)
      {
        print("exist common mode, calculating loss and beta now......")
        beta = cal_weight(sr,tg)
        print(beta)
      }
    }
  }
}



##### toy data examples for reweighting ##### 
calc_v_matrix<-function(ds,c)
{
  dist_s = as.matrix(dist(ds)) ## distance of 
  v_ij_matrix = 1/(dist_s+1.0/c)
  v_ij_matrix
}


x_1 = runif(200, min=70, max=100)
x_2 = runif(200, min=70, max=100)
plot(x = x_1,y = x_2,xlim = c(0,100),ylim = c(0,100),col = 'red')


y_1 = runif(1000, min=0, max= 100)
y_2 = runif(1000, min=0, max = 100)

par(new = TRUE)
plot(x = y_1,y = y_2,xlim = c(0,100),ylim = c(0,100),col = 'blue')




## get N? by sum(P) = 1

#v_ij_matrix = calc_v_matrix(ds,c)
#normalize_var = 1.0/sum(v_ij_matrix)

calc_wds_from_t<-function(d,t,c)
{
  t_v_ij_matrix = calc_v_matrix(t,c)
  t_normalizer = 1.0/sum(t_v_ij_matrix)
  
  d_v_ij_matrix = calc_v_matrix(d,c)
  d_normalizer = 1.0/sum(d_v_ij_matrix)
  
  t_x_v_ij_matrix = calc_v_matrix(rbind.data.frame(d,t),c)
  t_x_v_ij_matrix = t_x_v_ij_matrix[(1+nrow(d)):nrow(t_x_v_ij_matrix),1:nrow(d)]
  d_x_v_ij_matrix = calc_v_matrix(d,c)
  
  
  px_t = colSums(t_x_v_ij_matrix) * t_normalizer
  px_d = colSums(d_x_v_ij_matrix) * d_normalizer
  
  weight =  px_t/px_d
  weight
}

t = data.frame(x1 = x_1,x2 = x_2)
d = data.frame(x1 = y_1,x2 = y_2)
c = 0.003
weighted = calc_wds_from_t(d,t,c)
d$weight = weighted
d$weight = as.factor(d$weight)
n = length(unique(d$weight))
colPlate = rainbow(n, s = 1, v = 1, start = 0.1, end = max(1, n - 1)/n, alpha = 1)

plot(x = d$x1,y = d$x2,xlim = c(-100,100),ylim = c(-100,100),col = colPlate[d$weight])
par(new=TRUE)
plot(x = t$x1,y = t$x2,xlim = c(-100,100),ylim = c(-100,100),col = 'yellow')
