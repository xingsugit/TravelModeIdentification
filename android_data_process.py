import numpy as np
#import matplotlib.pyplot as plt
import pickle
#from scipy import stats
import random
#import rotationUtility
import sys
import os
from scipy.stats import mstats as mstats
from sklearn import preprocessing
## Android phone collected data format: [timestamp,type,axis,value]
# 173,TYPE_ACCELEROMETER,0,-0.49320554733276367
# 173,TYPE_ACCELEROMETER,1,3.2363622188568115
# 173,TYPE_ACCELEROMETER,2,10.509946823120117

## Data Description:

## This file process the android data files following the steps:
## 1. read in data from android file,
## 2. remove the data that collected within certain time period (distance of timestamps are too small)
## name									value field		description
## TYPE_ACCELEROMETER					0,1,2			[x,y,z]
## TYPE_GAME_ROTATION_VECTOR			0,1,2,3			?(in document it has 3 field, but in data 4
## TYPE_GRAVITY							0,1,2			[x,y,z]
## TYPE_GYROSCOPE						0,1,2			[x,y,z] - angular speed around x,y,z, radiance/s
## TYPE_GYROSCOPE_UNCALIBRATED			0,1,2,3,4,5		[x,y,z,drift_x,drift_y,drift_z]
## TYPE_LIGHT							0,1,2			only 1 value is valid (depend on device)
## TYPE_LINEAR_ACCELERATION				0,1,2			[x,y,z] m/s2, not including gravity
## TYPE_MAGNETIC_FIELD					0,1,2			[x,y,z] microTesla
## TYPE_MAGNETIC_FIELD_UNCALIBRATED		0,1,2,3,4,5		[x_uncal,y_u,z_u,x_bias,y_bias,z_bias]
## TYPE_PRESSURE						[0,1,2]			0: value (1,2 might be noise?)
## TYPE_PROXIMITY						[0,1,2]			0: proximity measured in cm
## TYPE_ROTATION_VECTOR					[0,1,2]			x,y,z (rotation_speed * sin(theta/2)
## TYPE_STEP_COUNTER					0
## TYPE_STEP_DETECTOR					0
## WEATHER								[...]
## ACT									0				activity
## LOCATION_GPS							LAT,LON,SPE,ALT,...
## LOCATION_NETWORK						... ...


## getSourceData
def getSourceData(file_name, start, end,uid): #change back to (file,timeRatio,start )
    # Jul. 23rd, 2014: add timeStamps at last, e.g. gravity from [[x],[y],[z]] to [[x],[y],[z],[timeStamp]]
    if (os.path.isfile(file_name)) == False:
        print "file doesn't exit", file_name
        sys.exit()
    activity_dict = {1:'walk',2:'jog',3:'bike',4:'bus',5:'subway',6:'car',7:'none'}
    gravity 			= [[],[],[],[],[],[]] #[x],[y],[z], [timestamp],[act],[ID]
    rotation 			= [[],[],[],[],[],[]]#[x],[y],[z],[timestamp],[act],[ID]
    game_rotation 		= [[],[],[],[],[],[]]#[x],[y],[z],[timestamp],[act][ID]
    uncalibrated_gyroscope 	= [[],[],[],[],[],[],[],[],[]] #[x],[y],[z],[xd],[yd],[zd],[timestamp],[act],[id]
    magnetic 			= [[],[],[],[],[],[]] #[x],[y],[z],[timestamp],[act],[id]
    light 			= [[],[],[],[]] #[v],[timestamp],[act],[id]
    gyroscope 			= [[],[],[],[],[],[]]#[x],[y],[z],[timestamp],[act],[id]
    pressure 			= [[],[],[],[]]#[v],[timestamp],[act],[id]
    magnetic_uncalibrated 	= [[],[],[],[],[],[],[],[],[]] #[x],[y],[z],[xb],[yb],[zb],[timestamp],[act],[id]
    acceleration 		= [[],[],[],[],[],[]]  #[x],[y],[z],[timestamp],[act],[id]
    linear_acceleration 	= [[],[],[],[],[],[]] #[x],[y],[z],[timestamp],[act],[id]

    print "begin read source data"
    source = np.genfromtxt(file_name, delimiter = ',', dtype = 'str',skiprows=1,invalid_raise = False )
    sorted_source = sorted(source,key = lambda x:int(x[0]))
    print "finish reading. Source file length is " + str(len(source))
    finishAddingFirst = False
    activity = ""
    for line in sorted_source:
        if start <int(line[0]) < end:
	    if line[1] == "ACT":
		print 'activity code:', line[3]
	        activity_code = int(float(line[3])) ## get current activity
		activity = activity_dict[activity_code]
            if line[1] == 'TYPE_GRAVITY':
                if finishAddingFirst == True:
                    gravity[int(line[2])].append(float(line[3]))
                    if int(line[2]) == len(gravity) - 4: ## reach the last value of current category.e.g. gravity on z axis. 6-4 = 2, which is the index of z value
                        finishAddingFirst = False
			gravity[4].append(activity)
			gravity[5].append(uid)
                else:
                    if len(gravity[3]) >0 and int(line[0]) == gravity[3][-1]: ## duplicated record
                        continue
                    else:
                         gravity[int(line[2])].append(float(line[3]))
                         gravity[3].append(int(line[0]))
                         finishAddingFirst = True
            if line[1] == 'TYPE_ROTATION_VECTOR':
		if finishAddingFirst == True:
	            rotation[int(line[2])].append(float(line[3]))
		    if int(line[2]) == len(rotation) - 4:
			finishAddingFirst = False
			rotation[4].append(activity)
			rotation[5].append(uid)
		else:
		    if len(rotation[3]) > 0 and int(line[0]) == rotation[3][-1]:
			continue
	            else:
			rotation[int(line[2])].append(float(line[3]))
			rotation[3].append(int(line[0]))
			finishAddingFirst = True
            if line[1] == 'TYPE_GAME_ROTATION_VECTOR':
                if finishAddingFirst == True:
                    game_rotation[int(line[2])].append(float(line[3]))
                    if int(line[2]) == len(game_rotation) - 4:
                        finishAddingFirst = False
			game_rotation[4].append(activity)
			game_rotation[5].append(uid)
                else:
                    if len(game_rotation[3]) >0 and int(line[0]) == game_rotation[3][-1]:
                        continue
                    else:
                        game_rotation[int(line[2])].append(float(line[3]))
                        game_rotation[3].append(int(line[0]))
                        finishAddingFirst = True
                        #print "game rotation set to true,", finishAddingFirst
            if line[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
				if finishAddingFirst == True:
					uncalibrated_gyroscope[int(line[2])].append(float(line[3]))
					if int(line[2]) == len(uncalibrated_gyroscope) - 4:
						finishAddingFirst = False
						uncalibrated_gyroscope[7].append(activity)
						uncalibrated_gyroscope[8].append(uid)
				else:
					if len(uncalibrated_gyroscope[6]) > 0 and int(line[0]) == uncalibrated_gyroscope[6][-1]:
						continue
					else:
						uncalibrated_gyroscope[int(line[2])].append(float(line[3]))
						uncalibrated_gyroscope[6].append(float(line[0]))
						finishAddingFirst = True
            if line[1] == 'TYPE_MAGNETIC_FIELD':
                if finishAddingFirst == True:
                    magnetic[int(line[2])].append(float(line[3]))
                    if int(line[2]) == len(magnetic) - 4:
                        finishAddingFirst = False
			magnetic[4].append(activity)
			magnetic[5].append(uid)
                else:
                    if len(magnetic[3]) >0 and int(line[0]) == magnetic[3][-1]:
                        continue
                    else:
                        magnetic[int(line[2])].append(float(line[3]))
                        magnetic[3].append(int(line[0]))
                        finishAddingFirst = True
            if line[1] == 'TYPE_LIGHT':
                if line[2] != '1':
                    continue
                else:
                    if len(light[0]) == 0 or light[1][-1] != int(line[0]):
                        light[0].append(float(line[3]))
                        light[1].append(int(line[0]))
			light[2].append(activity)
			light[3].append(uid)
            if line[1] == 'TYPE_GYROSCOPE':
				if finishAddingFirst == True:
					gyroscope[int(line[2])].append(float(line[3]))
					if int(line[2]) == len(gyroscope) - 4:
						finishAddingFirst = False
						gyroscope[4].append(activity)
						gyroscope[5].append(uid)
				else:
					if len(gyroscope[3]) > 0 and int(line[0]) == gyroscope[3][-1]:
						continue
					else:
						gyroscope[int(line[2])].append(float(line[3]))
						gyroscope[3].append(int(line[0])) ## timestamp
						finishAddingFirst = True
            if line[1] == 'TYPE_PRESSURE' and line[2] == '0':
				if len(pressure[0]) == 0 or pressure[1][-1] != int(line[0]):
					pressure[0].append(float(line[3]))
		    			pressure[1].append(int(line[0]))
					pressure[2].append(activity)
					pressure[3].append(uid)
            if line[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
                #print "begin uncalibrated magnetic"
                if finishAddingFirst == True:
                    magnetic_uncalibrated[int(line[2])].append(float(line[3]))
                    if int(line[2]) == len(magnetic_uncalibrated) - 4:
                        finishAddingFirst = False
			magnetic_uncalibrated[7].append(activity)
			magnetic_uncalibrated[8].append(uid)
                else:
                    if len(magnetic_uncalibrated[6]) >0 and int(line[0]) == magnetic_uncalibrated[6][-1]:
                        continue
                    else:
                         magnetic_uncalibrated[int(line[2])].append(float(line[3]))
                         magnetic_uncalibrated[6].append(int(line[0]))
                         finishAddingFirst = True
            if line[1] == 'TYPE_ACCELEROMETER':
                if finishAddingFirst == True:
                    acceleration[int(line[2])].append(float(line[3]))
                    if int(line[2]) == len(acceleration)-4:
                        finishAddingFirst = False
			acceleration[4].append(activity)
			acceleration[5].append(uid)
                else:
                    if len(acceleration[3]) >0 and int(line[0]) == acceleration[3][-1]:
                        continue
                    else:
                        acceleration[int(line[2])].append(float(line[3]))
                        acceleration[3].append(int(line[0]))
                        finishAddingFirst = True
            if line[1] == 'TYPE_LINEAR_ACCELERATION':
                if finishAddingFirst == True:
                    linear_acceleration[int(line[2])].append(float(line[3]))
                    if int(line[2]) == len(linear_acceleration) - 4:
                        finishAddingFirst = False
			linear_acceleration[4].append(activity)
			linear_acceleration[5].append(uid)
                else:
                    if len(linear_acceleration[3]) >0 and int(line[0]) == linear_acceleration[3][-1]:
                        continue
                    else:
                         linear_acceleration[int(line[2])].append(float(line[3]))
                         linear_acceleration[3].append(int(line[0]))
                         finishAddingFirst = True
                         #print "linear acc set to true:", finishAddingFirst
	else:
		break
    print len(gravity[0]),len(rotation[1]),len(game_rotation[1]),len(uncalibrated_gyroscope[1]),len(magnetic[1]),len(light[1]),len(gyroscope[1]),len(pressure[1]),len(magnetic_uncalibrated[1]),len(linear_acceleration[1])
    sourceArray = [gravity,acceleration, linear_acceleration,magnetic, magnetic_uncalibrated, pressure,  rotation, game_rotation, uncalibrated_gyroscope, light, gyroscope]
    return sourceArray

######The difference between processSampling, reSample and sampleNewData:
###### processSampling deals with the original source Data,
###### the input dataArray is of shape [[timeStamp],[x],[y],[z],[label]]
###### It removes the data that is readin with less than 10 milliseconds gap.

###### Resample dataArray, is that in order to get data of same length, we first choose
###### the minumum length min_l among all sensors' readings. Then, make the minumum length the
###### standard length for every sensor, and randomly choose min_l data from l_i data.
######
###### sampleNewData: get sparse data from dense data. the step needs to be calculated ahead
###### of time. If we take 200 milliseconds as the gap, and the original data reading gap is,
###### say, 10, then we will use 200/10 = 10 as the step.

###### Important: processSampling and sampleNewData: they deal with the original data shape
###### [[timeStamps],[X],[Y],[Z],[Label]]
###### reSample new data, deal with the new shape: [[timeStamp_i,x_i,y_i,z_i,label_i],.....]

## remove data that occured within A ms since last timestamp, input is the wide array: [[x],[y],[z],[timestamps],[activity],[uid]].
## output is also the wide array[[x],[y],[z],[activity],[uid]]
def processSampling(dataArray, A=10):
    print("\033[1;35;40m Before Process Sampling, the size of data is: %d \033[0;32;40m" % (len(dataArray[0])))
    timeList = dataArray[-3]
    periodList = []
    #newPList = []
    newDataArray = [[] for i in range(len(dataArray))]
    ## add the first reading in newDataArray
    for k in range(len(dataArray)):
        newDataArray[k].append(dataArray[k][0])
    ## get the list of time gap from two adjacient reading:
    for i in range(len(timeList)-1):
        periodList.append(timeList[i+1] - timeList[i])
    #process the occurence of 1 or less than 10:
    for i in range(len(periodList)):
        if periodList[i] >= A:
            #newPList.append(periodList[i])
            for k in range(len(dataArray)):
                newDataArray[k].append(dataArray[k][i+1])
    print("\033[1;35;40m after Process Sampling, the size of data is: %d \033[0;32;40m" % (len(newDataArray[0])))
    return newDataArray

def winsorize_and_scale(dataArray):
	new_arr = []
	for i,value_column in enumerate(dataArray[:-3]):
		w_value_column = mstats.winsorize(value_column,[0.05,0.05])
		print("finish winsorization for column #%d" % (i))
 		s_w_value_column = preprocessing.scale(w_value_column)
		print("finish scaling for column #%d" %(i))
		new_arr.append(s_w_value_column)
	new_arr += dataArray[-3:]
	return new_arr


# reshape to long.
def reshape(dataArr):
    ## put the timeArray in the first column
    dataArr = [dataArr[-3]] + dataArr[:-3] + dataArr[-2:]
    print("\033[1;31;40m The sensor has %d records\033[0;32;40m"% len(dataArr[0]))
    newShape = [[dataArr[i][j] for i in range(len(dataArr))] for j in range(len(dataArr[0]))]
    print("\033[1;31;40m Finish reshape, inside the function. The reshaped data length is %d \033[0;32;40m"% len(newShape))
    return newShape

## get a subset(length is predefined) from the dataArray.
## Scenario: If we have 3 sensors, with similar length, we need to synchronize them into same length, then use reSample to get a subset of each dataArray.
## It deals with the new shape: [[x,y,z,timestamp,activity, uid],...]
def reSample(dataArray,length):
    newIndex = sorted(random.sample(range(len(dataArray)),length))
    newArray = []
    for i in newIndex:
        newArray.append(dataArray[i])
    return newArray



### make the data more sparse
def sampleNewData(dataArr, step):
    newDataArr = [[] for i in range(len(dataArr))]
    i = 0
    while i < len(dataArr[0]) - step - 1:
        for k in range(len(dataArr)):
            newDataArr[k].append(dataArr[k][i])
        i += step
    return newDataArr

### extract_key_sensors_data extract the key sensors data, synchronize the timeline and return it.
### Parameter @sensorData: the output from function getSourceData. is a list of sensors' reading.
### Parameter @key_sensor_list: the name list of the sensors.
### Full name list of sensors:
### sensors = ['gravity','acceleration', 'linear_acceleration','magnetic', 'magnetic_uncalibrated', 'pressure',  'rotation', 'game_rotation', 'uncalibrated_gyroscope', 'light', 'gyroscope']
### It returns the selected sensor data that is synchrozied on timeline.
### the return value in the format of: [time, sensor_a, sensor_b, sensor_c, ....., label, uid] of which, sensor_a = [x,y,z] or sensor_a = [v], etc....
### The steps for doing that is:
### 1. reshape the key sensor's data into wide: from [[x],[y],[z],[timestamp],[label],[uid]] to: [[t,x,y,z,label,uid],...]
### 2. sychronize the timeline of the key sensors:
###    --- how to sychronize: 1) check the time laps between two consecutive readings of each sensor (in the key sensors list)
###    --- 		      2) decide what time lap is minumum
###    ---		      3) Clip the data of each sensor with the minumum lap
###    ---		      4) Rotate the acc (using gravity and uncalibrated magnetic field)
###    --- 		      5) Compose the synchronized data. and return.
###   default key_sensor_list:["linear_acceleration","gravity","magnetic_uncalibrated","magnetic","pressure","light"]
def extract_key_sensors_data(sensorData,key_sensor_list):
 	print "hello"

### from getSourceData() we read raw data from files, here we create functions to merge the data:
def mergeData(fileArray,sensorNumber = 11):
    source = [[] for i in range(sensorNumber)]
    uid_dict = {'hernan':'001','he':'002','ni':'003','zhenhua':'004'}
    ## Note that 13 is the number of sensors, result from getSourceData
    for f in fileArray:
		print 'Currently deal with file:',f
		name_key = f[:f.index('_')]
		uid = uid_dict[name_key]
		print uid
		f = path + '/'+ f
		sourceData = getSourceData(f,-1,12213333432,uid)
		print "new sourceData"
		for i,s in enumerate(sourceData):
			print("Process sampling for sensor #%d" % (i))
			if i == 1:
				print len(s),len(s[0]),len(s[1]),len(s[2]),len(s[3]),len(s[4])
			if i==5 and len(s[0]) == 0:
				print("\033[1;33;40m No Pressure data. skipped!\033[0;32;40m")
				continue
			s = processSampling(s)
			print("Reshape the data.")
			processed_s = winsorize_and_scale(s)
			long_s = reshape(processed_s)
			print('The size after reshape for sensor #%d is %d'%(i,len(long_s)))
			if not len(long_s) == len(s[0]):
				print "Error in data length. program halt!"
				sys.exit()
			source[i] += long_s
    return source


if __name__ == '__main__':
	path = './winter'
	file_list = os.listdir(path)
	sensors = ['gravity','acceleration', 'linear_acceleration','magnetic', 'magnetic_uncalibrated', 'pressure',  'rotation', 'game_rotation', 'uncalibrated_gyroscope', 'light', 'gyroscope']
	uid_dict = {'hernan':'001','he':'002','ni':'003','zhenhua':'004'}
	'''
	key_sensor_list= ["linear_acceleration","gravity","magnetic_uncalibrated","magnetic","pressure","light"]
	if 'pressure' in key_sensor_list:
		file_list = [x for x in file_list if 'zhenhua' not in x]
	for fileName in file_list:
		print fileName
		name_key = fileName[:fileName.index('_')]
		uid = uid_dict[name_key]
		print 'name:',name_key,'uid:',uid
		fileName = path + '/'+fileName
		df = getSourceData(fileName,-1,12213333432,uid)
 		print 'Number of sensors:',len(df)
		for i in range(len(df)):
			cur = df[i]
			if len(cur[-3]) > 0:
				print("\033[1;33;40m %s : %d\033[0;32;40m" %(sensors[i], len(cur[-3])))
			else:
				print("\033[1;31;40m The #%d sensor doesn't exist in file: %s\033[0;32;40m" % (i,fileName))
	'''
	merged = mergeData(file_list)
	print("\033[1;34;40m Merge finished. The total length is: %d\033[0;32;40m"%(len(merged)))
	for i,x in enumerate(merged):
		file_to_write = sensors[i]+'_winter_all_scaled.csv'
		print("\033[1;33;40m file to write: %s\033[0;32;40m" %(file_to_write))
		np.savetxt(file_to_write,x,fmt='%s',delimiter = ',')
	print("Finish writing!")

'''
    for fileName in file_list:
        print fileName
	name_key = fileName[:fileName.index('_')]
	uid = uid_dict[name_key]
	fileName = path + '/'+fileName
	source = getSourceData(fileName, -1,234897842903,uid)
	for i,s in enumerate(source):
	    print('now process sampling for sensor #%d' % (i))
            if i == 1:
                print len(s), len(s[0]),len(s[1]),len(s[2]),len(s[3]),len(s[4]),len(s[5])
	    if i==5 and len(s[0]) == 0:
		print("\033[1;33;40m No Pressure data. skipped!\033[0;32;40m")
		continue
	    x = processSampling(s)
'''
