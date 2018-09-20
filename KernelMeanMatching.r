###################### Kernel Mean Matching ########################
## calculate the measure of a probability distributio of set d. 
## c is a constant that is bigger than 1. 
## comp_space is the dataset that together with d, it consists the whole space s
## beta is the weight. if it is the target set, then set the value inside beta to be 1
set_measure<-function(d,comp_space,beta,c = 1.2)
{
  ## d is current, t is the compensate 
  s = rbind.data.frame(d,comp_space)
  
  dist_s = as.matrix(dist(s)) 
  ### select distance between di - sj, remove t
  dist_s = dist_s[,1:nrow(d)] ## column i is the distance of x_i to others in s
  v_ij_matrix = 1/(dist_s+1.0/c)
  normalize_var = 1.0/sum(v_ij_matrix) ## sum of matrix = sum(colSums(matrix))
  
  px = normalize_var *colSums(v_ij_matrix) # each column adds up to the value of px_i
  ## normx is the row norm of x. 
  normx =  apply(d, 1, function(x) norm(as.matrix(x),type='f'))
  measure = (normx * beta) %*% px
  as.numeric(measure)
}

## calculate the v_ij_matrix
calc_v_matrix<-function(ds,c)
{
  dist_s = as.matrix(dist(ds)) 
  v_ij_matrix = 1/(dist_s+1.0/c)
  v_ij_matrix
}
### simplex version doesn't construct an S.
set_measure_simplex<-function(d, c = 1.2)
{
  ## d is current, t is the compensate 
  v_ij_matrix = calc_v_matrix(d,c)
  normalize_var = 1.0/sum(v_ij_matrix) ## sum of matrix = sum(colSums(matrix))
  px = normalize_var *colSums(v_ij_matrix) # each column adds up to the value of px_i
  ## normx is the row norm of x. 
  normx =  apply(d, 1, function(x) norm(as.matrix(x),type='f'))
  measure = normx  %*% px
  as.numeric(measure)
}

test_set_measure<-function()
{
  ##toy data
  sr = matrix(c(1,1,1,1,2,2,2,2,3,3,3,3),nrow = 3,byrow=T)
  tg = matrix(c(4,4,4,4,5,5,5,5),nrow = 2,byrow = T)
  
  t_v_ij_matrix = calc_v_matrix(tg,c = 1.2)
  t_normalizer = 1.0/sum(t_v_ij_matrix)
  
  ##calculate the mean distance: 
  s = rbind.data.frame(sr,tg)
  c=1.2
  dist_s = as.matrix(dist(s)) 
  ### select distance between di - sj, remove t
  dist_s = dist_s[(1+nrow(d)):nrow(s),1:nrow(d)] ## column i is the distance of x_i to others in s
  dt_v_ij_matrix = 1/(dist_s+1.0/c)
  ## normx is the row norm of x. 
  normx_d =  apply(sr, 1, function(x) norm(as.matrix(x),type='f'))
 
  mean_distance = normx_d %*% colSums(dt_v_ij_matrix)
}



loss_in_t_learn_from_d<-function(d,t,c)
{
  t_v_ij_matrix = calc_v_matrix(t,c)
  t_normalizer = 1.0/sum(t_v_ij_matrix)
  
  d_v_ij_matrix = calc_v_matrix(d,c)
  d_normalizer = 1.0/sum(d_v_ij_matrix)
  
  normx =  apply(t, 1, function(x) norm(as.matrix(x),type='f'))
  
  t_x_v_ij_matrix = calc_v_matrix(rbind.data.frame(t,t),c)
  t_x_v_ij_matrix = t_x_v_ij_matrix[(1+nrow(t)):nrow(t_x_v_ij_matrix),1:nrow(t)]
  d_x_v_ij_matrix = calc_v_matrix(rbind.data.frame(t,d),c)
  d_x_v_ij_matrix = d_x_v_ij_matrix[(1+nrow(t)):nrow(d_x_v_ij_matrix),1:nrow(t)]
  px_t = colSums(t_x_v_ij_matrix) * t_normalizer
  px_d = colSums(d_x_v_ij_matrix) * d_normalizer
  
  weighted_x_loss = normx * (1 - px_t/px_d)
  px = t_normalizer *colSums(t_v_ij_matrix) # each column adds up to the value of px_i
  ## normx is the row norm of x. 
  
  loss = abs(weighted_x_loss  %*% px)
  
  loss
}

t_weighted_learn_from_d<-function(d,t,c)
{
  t_v_ij_matrix = calc_v_matrix(t,c)
  t_normalizer = 1.0/sum(t_v_ij_matrix)
  
  d_v_ij_matrix = calc_v_matrix(d,c)
  d_normalizer = 1.0/sum(d_v_ij_matrix)
  
  normx =  apply(t, 1, function(x) norm(as.matrix(x),type='f'))
  
  t_x_v_ij_matrix = calc_v_matrix(rbind.data.frame(t,t),c)
  t_x_v_ij_matrix = t_x_v_ij_matrix[(1+nrow(t)):nrow(t_x_v_ij_matrix),1:nrow(t)]
  d_x_v_ij_matrix = calc_v_matrix(rbind.data.frame(t,d),c)
  d_x_v_ij_matrix = d_x_v_ij_matrix[(1+nrow(t)):nrow(d_x_v_ij_matrix),1:nrow(t)]
  px_t = colSums(t_x_v_ij_matrix) * t_normalizer
  px_d = colSums(d_x_v_ij_matrix) * d_normalizer
  
  weighted_x = t * px_t/px_d
  weighted_x
}

calculate_weight_with_min_loss<-function(topn)
{
  ## ss is the loss matrix. 
  ss = read.csv("~/Dropbox/TravelData/src/weighted_loss_version2_c1.2.csv",header = T,row.names = 1)
  weighted_target_data = data.frame(matrix(ncol = 163, nrow = 0))
  colnames(weighted_target_data) = names(fspace)
  for(tu in uids)
  {
    print(tu)
    top_min_loss_su = which(ss[,which(uids == tu)] %in% sort(ss[,which(uids==tu)])[2:(2+topn-1)])
    sus = uids[top_min_loss_su]
    
    common_modes = which((uid_mode[tu,] >0) & (colSums(uid_mode[sus,]) > 0 ))
    
    if(length(common_modes) > 0)
    {
      for(m in common_modes)
      {
        sr = fspace_scaled[which(fspace$UID %in% sus & fspace$class ==modes[m]),]
        tg = fspace_scaled[which(fspace$UID == tu & fspace$class ==modes[m]),]
        ##loss = calc_absolute_distance(sr,tg,c) ##v1
        weighted_t = t_weighted_learn_from_d(sr[,1:161],tg[,1:161],c)
        weighted_t$UID = tu
        weighted_t$class = modes[m]
        weighted_target_data = rbind.data.frame(weighted_target_data,weighted_t)
      }
    }
    
  }
  write.csv(weighted_target_data,paste0('allphones_allmode_segLen16_noSlide_arbpm_normalize_uid_top_',topn,'_weightedt_c',c,'_2016-12_24.csv'),row.names = F)
  weighted_target_data
}
topn = 4
weighted_t = calculate_weight_with_min_loss(topn)

### First shift the Domain data ds with weight calculated by: pt(dxi)/pd(dxi), which will generate weighted ds
### calculated the expectation of wds (weighted ds)
### calcualted the distance between E(tr) and E(wds)
### get the minimum
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
  weighted_d = d * weight
}

calc_loss_between_t_and_wds<-function(d,t,c)
{
  weighted_d = calc_wds_from_t(d,t,c)
  wd_measure = set_measure_simplex(weighted_d, c = 1.2)
  t_measure = set_measure_simplex(t,c)
  loss = abs(t_measure-wd_measure)
  loss
}


cal_nondata_factor_matrix <-function(xi)
{
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
  xi = 0.3
  non_data_factorization <-function(u1_val,u2_val,xi=0.3)
  {
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
      factor_score = sum(mapply(non_data_factorization,user_profile[which(user_profile$uid == u1),],user_profile[which(user_profile$uid == u2),]))
      A[u1,u2] = factor_score
    }
  }
  A
}

A = cal_nondata_factor_matrix()

generate_loss_matrix_from_t_and_weightedD<-function(add_factors = F)
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
  
  #c = 1.2
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
            loss = calc_loss_between_t_and_wds(sr,tg,c)
            sum_mean_distance = sum_mean_distance + loss/(nrow(sr) + nrow(tg))
          }
          ss[which(uids==u1),which(uids==u2)] = sum_mean_distance
        }
        
      }
    }
  }
  ### add_factors: 
  if(add_factors)
  {
    ss = ss/log(A)
    file_name = paste0('weighted_loss_version3_c',c,'_addFactors.csv')
  }
  else
  {
    file_name = paste0('weighted_loss_version3_c',c,'.csv')
  }
  write.csv(ss,file_name)
  ss
}

generate_mode_wise_loss_matrix_from_t_and_weightedD<-function(add_factors = F)
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
  
  rownames(ss) = paste0('su',uids) ##source uid: row
  colnames(ss) = paste0('tu',uids) ##target uid: col
  for(m in modes)
  {
    file_to_save = paste0('/Users/Xing/Dropbox/TravelData/src/modewise_weighted_loss_',m)
    ss = matrix(rep(0,length(uids)**2),ncol = length(uids),nrow = length(uids))
    for(u1 in uids)
    {
      for(u2 in uids)
      {
        if(u1 != u2)
        {
          print(paste(u1,',',u2))
         
          if((uid_mode[u1,m] >0) & (uid_mode[u2,m]>0))
          {
              sr = fspace_scaled[which(fspace$UID == u1 & fspace$class ==m),]
              tg = fspace_scaled[which(fspace$UID == u2 & fspace$class ==m),]
              ##loss = calc_absolute_distance(sr,tg,c) ##v1
              loss = calc_loss_between_t_and_wds(sr,tg,c)
              
              ss[which(uids==u1),which(uids==u2)] = loss
          }
          else
          {
            ss[which(uids==u1),which(uids==u2)] = Inf
          }
          
        }
      }
    }
    if(add_factors)
    {
      ss = ss/log(A)
      file_to_save = paste0(file_to_save,'_addFactors.csv')
    }
    else
    {
      file_to_save = paste0(file_to_save,'.csv')
    }
    print(paste("write to file:",file_to_save))
    write.csv(ss,file_to_save)
  }
  
}
ss_jog = read.csv("/Users/Xing/Dropbox/TravelData/src/modewise_weighted_loss_subway_addFactors.csv",header = T,row.names = 1)

ss = generate_loss_matrix_from_t_and_weightedD(add_factors = T)

prepare_topn_weighted_learning_data_from_modewise_loss_ranking<-function(topn,addFactors = T)
{
  for(u in uids)
  {
    print("current user is:")
    print(u)
    ##1.each user has a weighted source file
    file_to_write = paste0("~/Dropbox/TravelData/data/uid_",u,"_top_",topn,"_weighted_learning_data.csv")
    ##2.create an empty data frame to put the weighted data in, and later save to the learning file
    weighted_learning_data =  data.frame(matrix(ncol = 163, nrow = 0))
    colnames(weighted_learning_data) = names(fspace)
    ##2.from each mode, get the uids with top n minimum loss from the mode ranking file 
    travel_modes = which(uid_mode[u,] >0)
    for(m in travel_modes)
    {
      print(paste("modes is: ",m))
      if(addFactors)
      {
        mode_wise_ranking_file = paste0('/Users/Xing/Dropbox/TravelData/src/modewise_weighted_loss_',modes[m],'_addFactors.csv')
      }
      else
      {
        mode_wise_ranking_file = paste0('/Users/Xing/Dropbox/TravelData/src/modewise_weighted_loss_',modes[m],'.csv')
      }
      
      weighted_loss_rank = read.csv(mode_wise_ranking_file,header = T,row.names = 1)
      ##get uids with top min loss
      top_min_loss_su = uids[which(weighted_loss_rank[,paste0('X',u)] %in% sort(weighted_loss_rank[,which(uids==u)],decreasing = TRUE)[2:(2+topn-1)])]
      for(su in top_min_loss_su)
      {
        sr = fspace_scaled[which(fspace$UID == su & fspace$class == modes[m]),1:161]
        tg = fspace_scaled[which(fspace$UID == u & fspace$class == modes[m]),1:161]
        weighted_sr = calc_wds_from_t(sr,tg,c)
        weighted_sr$UID = su
        weighted_sr$class = modes[m]
        weighted_learning_data = rbind.data.frame(weighted_learning_data,weighted_sr)
      }
    }
    print(names(weighted_learning_data))
    print(paste0("Write to file:",file_to_write))
    write.csv(weighted_learning_data,file_to_write,row.names = F)
  }
  
}
prepare_topn_weighted_learning_data_from_modewise_loss_ranking(topn=3)


calc_data_center<-function()
{
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
  data_center_profiles
}

data_center_profiles = calc_data_center()
write.csv(data_center_profiles,"/Users/Xing/Dropbox/TravelData/data/data_center_profiles.csv",row.names = F)



