# This file do three things:
## 1. Synchronize the timestamps of Acc, Mag, Gravity, Pressure
## 2. Rotate the acc with mag and gravity
## 3. Add label to acc/mag/pressure according to the Data Collection Records from UB and the acc plot
## 4. Save all data in format of csv as well as pickle data

import rotationUtility as ru
import numpy as np
from matplotlib import pyplot as plt
# common paramters
path = "/Users/Xing/Dropbox/TravelData/DataCleaning/"
# Nexus 1:
nexus1_acc = 'Nexus_20140703154924_acc.csv'
nexus1_m = 'Nexus_20140703154924_magCalib.csv'
nexus1_g = 'Nexus_20140703154924_gravity.csv'
nexus1_p = 'Nexus_20140703154924_pressure.csv'

# Nexus 2: ## car

nexus2_acc = 'Nexus_20140703170114_acc.csv'
nexus2_m = 'Nexus_20140703170114_magCalib.csv'
nexus2_g = 'Nexus_20140703170114_gravity.csv'
nexus2_p = 'Nexus_20140703170114_pressure.csv'

## Prof. He's file:
n900a_1 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703154548.csv'  #walk,jog
n900a_2 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703160320.csv'  #walk
n900a_3 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703160536.csv'  #?? walk, subway, jog, car
### probably begin to drive from here (16:49 )
n900a_4 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703164946.csv'  #car
## pickup julia
n900a_5 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703170213.csv'  #car
n900a_6 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703172034.csv'  #car
n900a_7 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703172217.csv'  #car
n900a_8 = '/Users/Xing/Dropbox/Field test 2014-07-03 (1)/data SAMSUNG-SM-N900A 20140703173111.csv'  #car



## Synchronize the timeStamps

# @function processSampling
# @param description:
# @dataArray the array needs to be processed. Each element in dataArry is [timeStamp, x[optional:, y, z], [optional:'dummyTag']]
# @A the minumum time gap. If the time gaps between two consecutive records is less than A, the second recod will be removed.
# return newDataArray: array with time gap at least A. Each element in newDataArray is [timeStamp, x[optional:, y, z], [optional:'dummyTag']]



def processSampling(dataArray, A=10):
    timeList = [x[0] for x in dataArray]
    periodList = []
    ## get the list of time gap from two adjacient reading:
    for i in range(len(timeList)-1):
        periodList.append(timeList[i+1] - timeList[i])
    #process the occurence of 1 or less than 10:
    newDataArray = []
    newDataArray.append(dataArray[0])
    for i in range(len(periodList)):
        if periodList[i] >= A:
            newDataArray.append(dataArray[i+1])
    return newDataArray


# @function indexOfMostCloseNumber: return the index of the value (in the array) which is the closest with the given value
# @param description:
# @array contains values that are compared with @currentValue
# @currentValue the value to compare to
# @currentIndex beginning of the index of the value in array that is
def indexOfMostCloseNumber(array,currentValue, currentIndex):
    b = min(range(currentIndex, len(array)), key=lambda i: (array[i]-currentValue)**2)
    return b


# @function syncArrays
# @param description:
# @arrA, arrB, arrC, arrD are arrays to synchronize, arrA is the base array(acc), arrB, arrC, arrD in the travelData are m,g,p.
# Each array is the list of element: [timestamps,....]
# @maxGap the maximum time gap that is allowed
# return two arrays that is synchronized most
def syncArrays(arrA, arrB,arrC,arrD,maxGap):
    new_a = []
    new_b = []
    new_c = []
    new_d = []
    currentIndex_b,currentIndex_c,currentIndex_d = 0,0,0
    timeStamps_b = [x[0] for x in arrB]
    timeStamps_c = [x[0] for x in arrC]
    timeStamps_d = [x[0] for x in arrD]
    for i in range(len(arrA)):
        currentValue = arrA[i][0]
        b_index = indexOfMostCloseNumber(timeStamps_b,currentValue,currentIndex_b)
        c_index = indexOfMostCloseNumber(timeStamps_c,currentValue,currentIndex_c)
        d_index = indexOfMostCloseNumber(timeStamps_d,currentValue,currentIndex_d)
        if abs(arrA[i][0] - arrB[b_index][0]) > maxGap:
            currentIndex_b = b_index
        elif abs(arrA[i][0] - arrC[c_index][0]) > maxGap:
            currentIndex_c = c_index
        elif abs(arrA[i][0] - arrD[d_index][0]) > maxGap:
            currentIndex_d = d_index
        else:
            new_a.append(arrA[i])
            new_b.append(arrB[b_index])
            currentIndex_b = b_index + 1 ## current value is accepted, the search should start from the next element
            new_c.append(arrC[c_index])
            currentIndex_c = c_index + 1 ## current value is accepted, the search should start from the next element
            new_d.append(arrD[d_index])
            currentIndex_d = d_index + 1

        if currentIndex_b == len(timeStamps_b) - 1 or currentIndex_c == len(timeStamps_c) - 1 or currentIndex_d == len(timeStamps_d):
            break
    return new_a,new_b,new_c,new_d


# @function syncTime
# param description:
# @acc list of acc records, each records is [timestamp, x, y, z, 'dummytag']
# @g list of gravity records, each record is [timestamp, x, y, z]
# @m list of magnetic field records, each record is [timestamp, z, y, z]
# @p list of pressure records, each record is [timestamp, p]
# returned value description:
# @acc_new, @m_new, @g_new, @p_new

def syncTime(acc,g,m,p):
    # first procesedfile is Nexus_20140703154924_acc.csv. In this file acc and gravity has similar length, magneticField and pressure have much more data records. Almost 3 times as much as the acc and gravity records.
    #setp 1: Remove the data with smaller time gap
    acc_processed = processSampling(acc)
    g_processed = processSampling(g)
    m_processed = processSampling(m)
    p_processed = processSampling(p)
    new_acc,new_m, new_g, new_p= syncArrays(acc_processed,m_processed,g_processed,p_processed,100)
    return new_acc, new_m, new_g, new_p



def rotateData(acc,g,m):
    acc_rotated = ru.rotate_list(acc,g,m)
    return acc_rotated


def dataRotationRoutine(path,acc_file_name,m_file_name,g_file_name,p_file_name):
    print 'begin loading data'
    acc = np.loadtxt(path + acc_file_name,skiprows = 1,delimiter = ',',dtype = {'names':('timeStamp','x','y','z','label'), 'formats':('i8','f8','f8','f8','S6')})
    m = np.loadtxt(path + m_file_name, skiprows = 1, delimiter = ',',dtype = {'names':('timeStamp','x','y','z'),'formats':('i8','f8','f8','f8')})
    p = np.loadtxt(path + p_file_name, skiprows = 1, delimiter = ',',dtype = {'names':('timeStamp','value'),'formats':('i8','f8')})
    g = np.loadtxt(path + g_file_name, skiprows = 1, delimiter = ',',dtype = {'names':('timeStamp','x','y','z'),'formats':('i8','f8','f8','f8')})
    print 'finish loading data'
    new_acc,new_m,new_g,new_p = syncTime(acc,g,m,p)
    print 'Data time is synchronized'
    acc_rotated = rotateData(new_acc,new_g,new_m)
    print 'Acceleration is rotated'
    return acc_rotated, new_m, new_p

#### Function to extract information from Nexus 1 data. This function only works for Nexus 20140703154924 data file.
## Plot is done in R with: rotated_acc on x, y, z, and slide-window autocorrelation
## the conclusion are drawn from 1) the volunteer's description 2) the ACT timeline in data records 3)the plot showing the changes of the motion
##
## Conclusion: Useful data are:
#   1) bus data, between timeStamp: [312346,1440866]
#   2) subway for a short period:[3207794,3699709]
#  return data format: list of [acc_x,acc_y,acc_z,mag_x,mag_y,mag_z,pressure,label]
def nexus1DataLabeling():
    path = "/Users/Xing/Dropbox/TravelData/DataCleaning/"
    nexus1_acc = 'Nexus_20140703154924_acc.csv'
    nexus1_m = 'Nexus_20140703154924_magCalib.csv'
    nexus1_g = 'Nexus_20140703154924_gravity.csv'
    nexus1_p = 'Nexus_20140703154924_pressure.csv'

    acc_rotated, new_m, new_p = dataRotationRoutine(path,nexus1_acc,nexus1_m,nexus1_g,nexus1_p)
    bus_acc = [x for x in acc_rotated if x[0]>= 312346 and x[0] <= 1440866]
    bus_m = [x for x in new_m if x[0]>= 312346 and x[0] <= 1440866]
    bus_p = [x for x in new_p if x[0]>= 312346 and x[0] <= 1440866]
    labeled_data = []
    min_bus_len = min(len(bus_acc),len(bus_m),len(bus_p))
    for i in range(min_bus_len):
        labeled_data.append([bus_acc[i][1],bus_acc[i][2],bus_acc[i][3],bus_m[i][1],bus_m[i][2],bus_m[i][3],bus_p[i][1],'bus'])
    print 'Nexus file 1 bus data finished'
    train_acc = [x for x in acc_rotated if x[0]>= 3207794 and x[0] <= 3699709]
    train_m = [x for x in new_m if x[0]>= 3207794 and x[0] <= 3699709]
    train_p = [x for x in new_p if x[0]>= 3207794 and x[0] <= 3699709]
    min_train_len = min(len(train_acc),len(train_m),len(train_p))
    for i in range(min_train_len):
        labeled_data.append([train_acc[i][1],train_acc[i][2],train_acc[i][3],train_m[i][1],train_m[i][2],train_m[i][3],train_p[i][1],'subway'])
    print 'Nexus file 1 train data finished'

    return labeled_data


### Function to extract information from Nexus 2 data. This function only works for file "data Nexus 5 20140703170114.csv"
### Data description from the person who collect the data: "Dr He take me from Erie Canal Harbor station to UB North Campus by car."
def nexus2DataLabeling():
    path = "/Users/Xing/Dropbox/TravelData/DataCleaning/"
    nexus2_acc = 'Nexus_20140703170114_acc.csv'
    nexus2_m = 'Nexus_20140703170114_magCalib.csv'
    nexus2_g = 'Nexus_20140703170114_gravity.csv'
    nexus2_p = 'Nexus_20140703170114_pressure.csv'

    acc_rotated,new_m,new_p = dataRotationRoutine(path,nexus2_acc,nexus2_m,nexus2_g,nexus2_p)
    return acc_rotated, new_m, new_p


### Functions to extract information from N900A data (Prof.He's activities)
### N900A 20140703154548.csv'
def n900a1DataLabeling():
    n900a1_acc = 'N900a1_20140703154548_acc.csv'
    n900a1_g   = 'N900a1_20140703154548_gravity.csv'
    n900a1_m   = 'N900a1_20140703154548_magCalib.csv'
    n900a1_p   = 'N900a1_20140703154548_pressure.csv'

    acc_rotated, new_m, new_p = dataRotationRoutine(path,n900a1_acc,n900a1_m,n900a1_g, n900a1_p)
    return acc_rotated, new_m, new_p
##

##

##

### test and other stuff. easy code, can delete
def scratchBoard():
    #labeled_data = nexus1DataLabeling()
    acc_rotated,new_m,new_p = n900a1DataLabeling()
    np.savetxt('a900a1_rotated_acc.csv',acc_rotated,delimiter = ',',fmt='%s')
    np.savetxt('a900a2_processed_m.csv',new_m,delimiter = ',',fmt = '%s')
    np.savetxt('a900a3_processed_p.csv',new_p,delimiter = ',',fmt = '%s')

    #np.savetxt('nexus1_labeled_data.csv', labeled_data, delimiter = ',',fmt = '%s')
    #print 'nexus 1 labeled data is saved'


scratchBoard()
