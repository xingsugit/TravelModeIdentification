library('msgl')
#library('grpreg')
library('gglasso')
library(MASS)
library('caret')
set_common_variables <-function(trail_set)
{
  sd_file = '~/Dropbox/TravelData/data/allmode_segLen16_noSlide_arbpm_normalize_2016-08-21.csv'
  sd = read.csv(sd_file)
  sd = sd[which(sd$class %in% trail_set),]
  feature_list <<-names(sd)
  ## split
  training_index = createDataPartition(y = sd$class,p = 0.8)
  training_data = sd[unlist(training_index),]
  test_data = sd[-unlist(training_index),]
  
  x_train <<- training_data[,1:161]
  y_train <<- training_data[,162]
  x_test <<- test_data[,1:161]
  y_test <<- test_data[,162]
  
  wheeled <<-c("bike",'bus','car','subway')
  unwheeled <<-c('walk','jog')
  
  indoor <<-c('car','bus','subway')
  outdoor<<-c('bike')
  
  level1 <<-c('wheeled','unwheeled')
  level2 <<-c('indoor','outdoor')
  ## scale:
  ### using library: Caret
  preProc  <- preProcess(x_train)
  x_training_scaled <<- predict(preProc, x_train)
  x_test_scaled     <<- predict(preProc, x_test)
  
  ## group: acc: 1:84, rotation:85:147, Screen_brightness: 148:151; Pressure: 152:155; Mag: 156:161
  sensors <<- c('Accelerometer','Gyroscope','Light','Barometer','Magnetometer')
  fv_group <<- c(rep(1,84),rep(2,63),rep(3,4),rep(4,4),rep(5,6))
}

y_binarize <-function(trail_set)
{
  y_train <<- ifelse(y_train==trail_set[1],1,-1)
  y_test <<- ifelse(y_test==trail_set[1],1,-1)
}

hierarchicle <-function(lv)
{
  if(lv == 1)
  {
    y_train<<-ifelse(y_train %in% wheeled,'wheeled','unwheeled')
    y_test <<-ifelse(y_test %in% wheeled,'wheeled','unwheeled')
  }
  else 
  {
    if(lv == 21) ## wheeled
    {
      stopifnot(length(unique(y_train)) == 4)
      y_train <<-ifelse(y_train %in% indoor,'indoor','outdoor')
      y_test <<-ifelse(y_test %in% indoor,'indoor','outdoor')
    }
    if(lv == 22) ##unwheeled
    {
      stopifnot(length(unique(y_train)) == 2)
      stopifnot("walk" %in% unique(y_train)) 
      stopifnot("jog" %in% unique(y_train)) 
    }
    if(lv == 31)
    {
      stopifnot(length(unique(y_train)) == 2)
      stopifnot("bus" %in% unique(y_train)) 
      stopifnot("car" %in% unique(y_train))
    }
    if(lv == 32)
    {
      stopifnot(length(unique(y_train)) == 2)
      stopifnot("bus" %in% unique(y_train)) 
      stopifnot("subway" %in% unique(y_train))
    }
    if(lv == 33)
    {
      stopifnot(length(unique(y_train)) == 2)
      stopifnot("car" %in% unique(y_train)) 
      stopifnot("subway" %in% unique(y_train))
    }
  }
}

#example:
## trail_set = c("walk","jog")
## lvnum: 1,21,22,31,32,33
## lv:1,2,3
## maintitle = "Walk VS Jog"

gglasso_train_plot<-function(trail_set,lvnum,maintitle,loss ='ls')
{
  set_common_variables(trail_set)
  hierarchicle(lvnum)
  y_binarize(unique(y_train))
  if(loss == 'hsvm')
  {
    dlt = 0.7
  }
  trail = gglasso(x = as.matrix(x_training_scaled),y = y_train,group = fv_group,loss = loss,delta = dlt,pmax = 2)#,lambda = seq(0.01,0.9,length.out=100))
  result = predict(trail,x_test_scaled,group = fv_group,loss = loss,type = 'class',delta = dlt,pmax = 2)
  
  if(loss == 'ls')
  {
    for(j in 1:100)
    {
      result[,j] = ifelse(result[,j]>0,unique(y_train)[1],unique(y_train)[2])
    }
    
  }
  
  
  
  plot(trail) # plots the coefficients against the log-lambda sequence 
  plot(trail,group=TRUE) # plots group norm against the log-lambda sequence 
  plot(trail,group = TRUE,log.l=FALSE) # plots against the lambda sequence
  
  sensor_num = 0
  updated_round = c()
  sensor_used = list()
  ii = 1
  accuracy_list = c()
  for(i in 1:100)
  {
    acc = sum(result[,i] == y_test)/length(y_test)
    accuracy_list = c(accuracy_list,acc)
    current_beta = trail$beta[,i]
    indexes = which(abs(current_beta) > 0.00001)
    si = unique(fv_group[indexes])
    sensor_selection = sensors[si]
    current_num = length(sensor_selection)
    if(current_num > sensor_num)
    {
      print(paste0(i,', Sensor selection:'))
      print(sensor_selection)
      sensor_used[[ii]] = sensor_selection
      sensor_num = current_num
      updated_round = c(updated_round,i)
      ii = ii+1
    }
    
  }
  
  plot(accuracy_list,type = 'l',col = 'blue',main=maintitle)
  hh = 10
  for(i in 1:sensor_num)
  {
    abline(v = updated_round[i], lty="dotted")
    #text(hh,accuracy_list[updated_round[i]],cat(sensor_used[[i]],sep = ','))
    hh = hh + 20
  }
  par(new = T)
  plot(trail$lambda,type = 'o',col = 'red',axes=F, xlab=NA, ylab=NA)
  axis(side = 4)
  mtext(side = 4, line = 3, 'lambda')
  
}

learning_with_group_lasso_regularization <- function()
{
  ##trail 1: car vs bus
  ##trail 2: bus vs others
  ##trail 3: subway vs others
  ##trail 4: bike vs others
  ##trail 5: walk vs others
  ##trail 6: jog vs others

  ##trail 7: subway vs bus
  trail_set7 = c('subway','bus')
  set_common_variables(trail_set7)
  y_binarize(trail_set7)
  trail7 = gglasso(x = as.matrix(x_training_scaled),y = y_train,group = fv_group,loss = 'sqsvm')
  result7 = predict(trail7,x_test_scaled,group = fv_group,loss = 'sqsvm',type = 'class')
  plot(trail7) # plots the coefficients against the log-lambda sequence 
  plot(trail7,group=TRUE) # plots group norm against the log-lambda sequence 
  plot(trail7,group = TRUE,log.l=FALSE) # plots against the lambda sequence
  
  
  ## trail 8: wheeled & unwheeled
  trail_set8 = c('subway','bus','car','bike','walk','jog')
  lvnum8 = 1 #level of trail set #8
  maintitle8 = "Wheeled VS Unwheeled"
  gglasso_train_plot(trail_set8,lvnum8,maintitle8,loss ='hsvm')

  ##################### hierachiecal ########################################
  ### trail9: indoor, outdoor
  trail_set9 = c("subway", "bus"  ,  "car"  ,  "bike")
  lvnum9 = 21
  maintitle9 = "Indoor VS Outdoor"
  gglasso_train_plot(trail_set9,lvnum9,maintitle9,loss ='ls')
  
  ### trail 10: walk, jog
  trail_set10 = unwheeled
  lvnum10 = 22
  maintitle10 = "Walk VS Jog"
  gglasso_train_plot(trail_set10,lvnum10,maintitle10,loss ='ls')
    
  ### trail 11: 
  trail_set11 = c('bus','car')
  lvnum11 = 31
  maintitle11 = "Bus VS Car"
  gglasso_train_plot(trail_set11,lvnum11,maintitle11,loss ='ls')

  
  
  ### trail 12:
  trail_set12 = c('bus','subway')
  lvnum12 = 32
  maintitle12 = "Bus VS Subway"
  gglasso_train_plot(trail_set12,lvnum12,maintitle12,loss ='ls')
  
 
  ### trail 13:
  trail_set13 = c('car','subway')
  lvnum13 = 33
  maintitle13 = "Car VS Subway"
  gglasso_train_plot(trail_set13,lvnum13,maintitle13,loss ='ls')
  
  
  ## answer the following questions:
  # 1. In the second plot, how to mark (or know which line represent which group.)
  ### scratch: 
  trail7_cv_training = rbind(x_training_scaled,x_test_scaled)
  trail7_cv_test = c(y_train,y_test)
  trail7_cv = cv.gglasso(as.matrix(trail7_cv_training),trail7_cv_test,group = fv_group,pred.loss = 'misclass',nfolds = 10,loss = 'sqsvm')
  
  # 2. In the predict function, which lambda is in use. 
  # 3. what's the accuracy VS. group size (in current setting?)
  # 4. should I try customized lambda values? 
  # 5. Setting: dfmax: the maximum number of groups in this model. pmax: limit the maximum number of the groups ever to be nonzero.
  
  # optional: what's the special of this group lasso algorithm? 
  
  ## Summarize:
  #1. How is the performance for each pairwise classification? 
  # optional: OVO and OVR? 
  
  
  
  
 #####-------------------------------- msgl ------------------------------### 
  trail_set21 = c('subway','bus','car','bike','walk','jog')
  set_common_variables(trail_set21)
  config <- msgl.algorithm.config()
  hierarchicle(1)
  lambda = msgl.lambda.seq(as.matrix(x_training_scaled),y_train,grouping = fv_group,alpha = 0,standardize = F,lambda.min = 0.001)
  
  #lambda = trail7$lambda
  msgl_fit = msgl(as.matrix(x_training_scaled),classes = y_train,grouping = fv_group,alpha = 0,standardize = F,lambda = lambda,algorithm.config = config)
  
  unique_groups = unique(features(msgl_fit))
  
   
  
  
  predict_y = predict(msgl_fit,x_test_scaled)
  
  ### plot
  predict_accuracy_path = c()
  for(i in 1:100)
  {
    predict_class = predict_y$classes[,i]
    accuracy = 1.0*sum(predict_class == y_test)/length(y_test)
    predict_accuracy_path = c(predict_accuracy_path,accuracy)
  }
 
 feature_numbers_list = c()
 for(i in 1:100)
 {
   feature_nu = length(features(msgl_fit)[[i]])
   feature_numbers_list = c(feature_numbers_list,feature_nu)
 }
 par(mar = c(3,5,3,5))
 plot(predict_accuracy_path,type = 'l')  
 par(new = T)
 plot(feature_numbers_list,type = 'o',col = 'red',axes=F, xlab=NA, ylab=NA)
 axis(side = 4)
 mtext(side = 4, line = 3, 'Numbers of features selected')
 
 for(group in unique_groups)
 {
   
   indexes = which(feature_list %in% group)
   sensors_index = unique(fv_group[indexes])
   sensor_selection = sensors[sensors_index]
   print('Sensor selection:')
   print(sensor_selection)
 }
 
 
 ### -------------------------- gmlnet------------------------------------- ######
 
 
 
  ######## ---------------------- grplasso --------------------------------  ######## 
  lambda = lambdamax(x = as.matrix(x_training_scaled),y = y_train,index = fv_group,model = LinReg(),center=F,standardize = F)* 0.5^(0:50)
  fit = grplasso(x = as.matrix(x_training_scaled), y = y_train, index = fv_group,model = LinReg(),lambda = lambda,center=F,standardize = F)
  pred <- predict(fit)
  pred.resp <- predict(fit, type = "response")
}

