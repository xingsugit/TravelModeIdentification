## set location
flist = list.files(path = './winter/')


### scrach board 
### test before write into functions
d = read.csv(paste0(getwd(),'/winter/',flist[1]))




essential_sensors = c("ACT","TYPE_LIGHT","TYPE_PROXIMITY","TYPE_ACCELEROMETER","TYPE_GYROSCOPE",
                      "TYPE_GYROSCOPE_UNCALIBRATED","TYPE_MAGNETIC_FIELD_UNCALIBRATED",
                      "TYPE_MAGNETIC_FIELD","TYPE_ROTATION_VECTOR","TYPE_PRESSURE","TYPE_GRAVITY",
                      "TYPE_LINEAR_ACCELERATION","TYPE_GAME_ROTATION_VECTOR")

d_essential = d[which(d$Name %in% essential_sensors),]
## there exist same timestamp different readings, remove the duplicated readings and leave the one that 
## occured at last.
d_essential$count = 1

d_agg = aggregate(count~TimeStamp+Name+ValueName,d_essential,sum)

duplicate = d_agg[which(d_agg$count > 1),c("TimeStamp","Name","ValueName")]


library(prodlim)
duplicate_index = row.match(duplicate,d_essential[,c(1,2,3)])
d_essential_unique = d_essential[-duplicate_index,c("TimeStamp","Name","ValueName","Value")]

d_essential_unique$count = 1
agg_d_u = aggregate(count~TimeStamp+Name+ValueName,d_essential_unique,sum)


d_acc = d_essential_unique[which(d_essential_unique$Name == "TYPE_ACCELEROMETER" & d_essential_unique$ValueName == 0),]
d_mag = d_essential_unique[which(d_essential_unique$Name == "TYPE_MAGNETIC_FIELD_UNCALIBRATED" & d_essential_unique$ValueName == '0'),]
d_p = d_essential_unique[which(d_essential_unique$Name == 'TYPE_PRESSURE' & d_essential_unique$ValueName==0),]
d_magc = d_essential_unique[which(d_essential_unique$Name == "TYPE_MAGNETIC_FIELD" &d_essential_unique$ValueName == 0),]
d_light = d_essential_unique[which(d_essential_unique$Name == 'TYPE_LIGHT' & d_essential_unique$ValueName == 1 ),]
d_gyro = d_essential_unique[which(d_essential_unique$Name == 'TYPE_GYROSCOPE' & d_essential_unique$ValueName == 0),]
acc_diff =diff(d_acc$TimeStamp)
mag_diff = diff(d_mag$TimeStamp)
p_diff = diff(d_p$TimeStamp)
magc_diff = diff(d_magc$TimeStamp)
light_diff = diff(d_light$TimeStamp)
gyro_diff = diff(d_gyro$TimeStamp)

single_value = d_essential_unique[which(d_essential_unique$ValueName == 0),]
s_diff = diff(single_value$TimeStamp)

d_acc = d_acc[sort(d_acc$TimeStamp),]
d_wide = reshape(d_acc,v.names = 'Name',timevar = 'ValueName',idvar = 'TimeStamp',direction="wide")

acc_bike = d_essential_unique[which(d_essential_unique$Name =="TYPE_LINEAR_ACCELERATION" ),]

testt = acc_bike
testt$count = 1

tag = aggregate(count ~ TimeStamp + Name,testt,sum)

acc_bike_wide = reshape(acc_bike,timevar = "ValueName",idvar = 'TimeStamp',direction='wide')


d_acc_y = d_acc[which(d_acc$ValueName == 1),]
plot(d_acc_y$Value,type = 'l')

### In the original sensor reading data, there exist readings with same timestamp but different values. The function
### removes the duplicated readings (for each timestamp that contains more than one reading, it keeps the latest record)  

remove_duplicate<-function(df)
{
  df$count = 1
  df_agg = aggregate(count~TimeStamp+Name+ValueName,df,sum)
  duplicate = df_agg[which(df_agg$count > 1),c("TimeStamp","Name","ValueName")]
  if(nrow(duplicate)>0)
  {
    duplicate_index = row.match(duplicate,df[,c(1,2,3)])
    df_unique = df[-duplicate_index,c("TimeStamp","Name","ValueName","Value")]
  }
  else
  {
    df_unique = df
  }
  df_unique
}

## read the files in (all files under the path directory, remove all the duplicated data, and then save it as new file, ending with "_remove_duplicate')
batch_process_files<-function(path)
{
  flist = list.files(path)
  for (i in 1:length(flist))
  {
    file_name = flist[i]
    print(paste("Now process file:", file_name))
    d = read.csv(paste0(getwd(),'/winter/',file_name))
    d_essential = d[which(d$Name %in% essential_sensors),]
    df_unique = remove_duplicate(d_essential)
    file_to_write = paste0(substr(file_name,0,(nchar(file_name)-4)),"_remove_duplicate",substr(file_name,nchar(file_name)-3,nchar(file_name)))
    write.table(df_unique,file = paste0(path,file_to_write),sep = ',',quote = FALSE,col.names = T,row.names = F)
    print(paste("Finish writing to file:",file_to_write))
  }
}



winsorize <- function (x, fraction=.05)
{
  if(length(fraction) != 1 || fraction < 0 ||
     fraction > 0.5) {
    stop("bad value for 'fraction'")
  }
  lim <- quantile(x, probs=c(fraction, 1-fraction))
  x[ x < lim[1] ] <- lim[1]
  x[ x > lim[2] ] <- lim[2]
  x
}

## Observe the single sensor's data (to see how discriminative it is, what is the general analysis)

pressure_file = 'pressure_winter_all.csv'
light_file = 'light_winter_all.csv'
fn = read.csv(light_file,header = F)


names(fn) = c('TimeStamp','value','label','UID')
fn$label = as.factor(fn$label)

fn_hernan = fn[which(fn$UID == 1),]
fn_he = fn[which(fn$UID == 2),]
fn_he = fn_he[order(fn_he$label),]
fn_ni = fn[which(fn$UID == 3),]
fn_ni = fn_ni[order(fn_ni$label),]
fn_zhenhua = fn[which(fn$UID == 4),]
fn_zhenhua = fn_zhenhua[order(fn_zhenhua$label),]

layout(matrix(c(1,2,3,4),4,1,byrow = TRUE),heights=c(1,3,3,3),widths = rep.int(1,4))
## 3. plot the common legend, set the page title
par(mar=c(2.0,2.0,2.5,1.5))
col_plate = c('red','green','blue','cyan','black','purple','yellow')
plot(1, type = "n", axes=FALSE, xlab="", ylab="")
legend(x = "top",inset = 0,
       legend = levels(fn$label), 
       col=col_plate, lwd=5, cex=1.5, horiz = TRUE,bty = "n")

title("Winter Pressure",cex.main = 1.5)
plim = c(995,1014)
plot(fn_zhenhua$value,col = col_plate[fn_zhenhua$label],type = 'h',main = "zhenhua's")

plot(x=fn_ni$TimeStamp, y = fn_ni$value,col = col_plate[fn_ni$label],type = 'p',main = "Ni's",ylim = plim,pch=16,cex = 0.6)

plot(x=fn_hernan$TimeStamp,y = fn_hernan$value,col = col_plate[fn_hernan$label],type = 'p',main = "Hernan's",ylim = plim,pch=16,cex = 0.6)

plot(x=fn_he$TimeStamp,y=fn_he$value,col = col_plate[fn_he$label],type = 'p',main = "He's",ylim = plim,pch=16,cex = 0.6)


par(mfrow = c(1,1))
plot(fn$value,col = col_plate[fn$label],type = 'p',pch=16,cex = 0.6)

plot(fn$value,col = col_plate[fn$label],type = 'p',pch=16,cex = 0.6,ylim = c(0,10000))

legend(x = "top",inset = 0,
       legend = levels(fn$label), 
       col=col_plate, lwd=5, cex=1.5, horiz = TRUE,bty = "n")


### scrach of Sep. 8th 2016 check on how this rotation vector-> rotationMatrix work #### 
fname = '/Users/Xing/Dropbox/TravelData/DataCleaning/data Nexus 5 20140703170114.csv'
fd = read.csv(fname,stringsAsFactors = F)
fd_magnetic = fd[which(fd$Name == "TYPE_MAGNETIC_FIELD" & fd$ValueName == 1),]
fd_rotation = fd[which(fd$Name == 'TYPE_ROTATION_VECTOR' & fd$ValueName == "1"),]
fd_gravity = fd[which(fd$Name == 'TYPE_GRAVITY' &fd$ValueName == '1'),]
fd_acc = fd[which(fd$Name == 'TYPE_LINEAR_ACCELERATION' & fd$ValueName =='1'),]
plot(fd_rotation$TimeStamp,type = 'l',col = 'red')
par(new = T)
plot(fd_acc$TimeStamp,type = 'l',col = 'blue')

magnetic = data.frame(timeStamps = fd_magnetic$TimeStamp,
                      x = as.numeric(as.character(fd[which(fd$Name == 'TYPE_MAGNETIC_FIELD' & fd$ValueName == '0'),'Value'])),
                      y = as.numeric(as.character(fd[which(fd$Name == 'TYPE_MAGNETIC_FIELD' & fd$ValueName == '1'),'Value'])),
                      z = as.numeric(as.character(fd[which(fd$Name == 'TYPE_MAGNETIC_FIELD' & fd$ValueName == '2'),'Value'])))

magnetic = magnetic[!duplicated(magnetic$timeStamps), ]
dense_reading_index = which(diff(magnetic$timeStamps) < 18) + 1
magnetic = magnetic[-dense_reading_index,]
magnetic = magnetic[which(magnetic$timeStamps > gravity[1,'timeStamps']),]
## make gravity the standard:
fd = fd[!duplicated(fd$TimeStamp),]

critical_sensors = fd[which(fd$Name %in% c('TYPE_MAGNETIC_FIELD','TYPE_GRAVITY',
                                           'TYPE_ROTATION_VECTOR','TYPE_LINEAR_ACCELERATION',
                                           'TYPE_PRESSURE','TYPE_LIGHT') & fd$ValueName == 0),]

pressure_diff = diff(critical_sensors[which(critical_sensors$Name == 'TYPE_PRESSURE'),'TimeStamp'])





rotation_vector = data.frame(timeStamps = fd_rotation$TimeStamp,
                             x = as.numeric(as.character(fd[which(fd$Name == 'TYPE_ROTATION_VECTOR' & fd$ValueName == "0"),'Value'])),
                             y = as.numeric(as.character(fd[which(fd$Name == 'TYPE_ROTATION_VECTOR' & fd$ValueName == "1"),'Value'])),
                             z = as.numeric(as.character(fd[which(fd$Name == 'TYPE_ROTATION_VECTOR' & fd$ValueName == "2"),'Value'])))




gravity = data.frame(timeStamps = fd_gravity$TimeStamp,
                     x = as.numeric(as.character(fd[which(fd$Name == 'TYPE_GRAVITY' & fd$ValueName == "0"),'Value'])),
                     y = as.numeric(as.character(fd[which(fd$Name == 'TYPE_GRAVITY' & fd$ValueName == "1"),'Value'])),
                     z = as.numeric(as.character(fd[which(fd$Name == 'TYPE_GRAVITY' & fd$ValueName == "2"),'Value'])))


rotation_vector = rotation_vector[-which(rotation_vector$timeStamps == 47834),]
## 47834 is the extra rotation vector

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

### rotation: experiment done with gravity. it is different from iphone
### in iphone: g_reference = g_observed * inverse(rotation_matrix) --->solve(rotation_matrix)
### in android: g_reference = g_observed * rotation_matrix 
### android: after rotate, z is pointing up. value is 9.8

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
  g_reference
}

g1 = gravity[1,]
r1 = rotation_vector[1,]

g_ref = rotate_gravity(g1,r1)

rotate_gravity_in_dataFrame<-function(gravity,rotation_vector)
{
  result = mapply(rotate_gravity,gravity$x,gravity$y,gravity$z,rotation_vector$x,rotation_vector$y,rotation_vector$z)
  result = t(result)
  rotated_gravity = data.frame(rotate_gravity_x = result[,1],rotate_gravity_y = result[,2],rotate_gravity_z = result[,3])
  rotated_gravity
}
rotated_gravity = rotate_gravity_in_dataFrame(gravity,rotation_vector)

