import numpy as np
import os
import sys
import rotationUtility
from sklearn import preprocessing





### note: to print out debug infomation, set log_level>0
def log(s):
	if log_level > 0:
		print s

def isIncrement(t):
	return all(x<y for x,y in zip(t,t[1:]))

def isUnique(a):
	return len(set(a)) == 1

## Function: normalize
## It takes one parameter, 1-d array, and output the normalized array
## Calculation: e = (e-min_arr)/(max_arr-min_arr)
## if the arr contains elements with same value(i.e. max_arr == min_arr), then return 0.5
def normalize(arr):
    max_e = max(arr)
    min_e = min(arr)
    if len(arr) == 1:
        return [0.5]
    else:
        if max_e == min_e:
            return [0.5 for e in arr]
        else:
            return [1.0 * (e-min_e)/(max_e-min_e) for e in arr]


### read and process single data segment.
### Input data format: [timestamp, value, label]
### output [seg1,seg2,...],[label1,label2,...], of which seg1 = [v1,v2,...vn], n is the seg length
def data_segmentation(data, segLen, lag):
	result = []
	activity_label = []
	activity = ''
	window_index = -1
	i = 0
	while i < len(data)-segLen + 1:
		time_seq = [x[0] for x in data[i:i+segLen]]
		data_seq = [x[1] for x in data[i:i+segLen]]
		activity_seq = [x[2] for x in data[i:i+segLen]]
		# to avoid putting data collected at different event together.
		if isIncrement(time_seq) and isUnique(activity_seq):
			result.append(data_seq)
			activity_label.append(activity_seq[0])
			i = i + lag
		else:
			i += 1
	return result, activity_label

## This function generate fv for pressure data.
def generate_pressure_fv(file_name,labels,segLen,lag):
	orig_data = np.loadtxt(file_name,dtype = 'str',delimiter = ',')
	#pressure data: [time, value, label,uid]
	data = [x[:-1] for x in orig_data]
	segs, labels = data_segmentation(data,segLen,lag)






if __name__ == '__main__':
    global log_level
    log_level = int(args[1])
    data = [[1,33,'a'],[2,34,'a'],[3,35,'a'],[4,36,'a'],[5,37,'a'],[6,38,'a'],[7,39,'a'],[8,40,'a'],[9,41,'a'],[1,42,'a'],[2,43,'a'],[3,43,'a'],[4,44,'a'],[5,45,'a'],[6,46,'a'],[7,47,'b'],[8,48,'b'],[9,49,'b'],[10,50,'b'],[1,51,'c'],[2,52,'c'],[3,53,'c'],[4,54,'c'],[2,55,'d'],[3,56,'d']]
    orig_data = np.loadtxt('/home/suzy/TravelData/pressure_winter_all.csv',dtype = 'str',delimiter = ',')
    #pressure data: [time, value, label,uid]
    data = [x[:-1] for x in orig_data]
    segLen = 4
    lag = 2
    segs, labels = data_segmentation(data,segLen,lag)
    print 'original data = ', data
    print "segment = ",segs
    print "labels = ",labels
	main(sys.argv)
