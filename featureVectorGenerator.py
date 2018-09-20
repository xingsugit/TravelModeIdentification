import numpy as np
import os
import sys
from sklearn import preprocessing
import pandas as pd
from datetime import date
import json
### note: to print out debug infomation, set log_level>0

def log(s):
	if log_level > 0:
		print(s)

## red:31,green:32,yellow:33,blue:34,purple:35,cyan:36,white:37
## bold:1,normal:0
## background: black: 40
def log_red(s):
    if log_level >0:
        print('\033[1;31;40m'+ str(s) + '\033[0;32;40m')

def log_yellow(s):
    if log_level > 0:
        print('\033[1;33;40m' + str(s) + '\033[0;32;40m')

def log_blue(s):
    if log_level > 0:
        print('\033[1;34;40m' + str(s) + '\033[0;32;40m')

def log_purple(s):
    if log_level > 0:
        print('\033[1;35;40m' + str(s) + '\033[0;32;40m')

def log_cyan(s):
    if log_level > 0:
        print('\033[1;36;40m' + str(s) + '\033[0;32;40m')


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


### read and process single valued data segment. (e.g. acc_x)
### Input data format: [timestamp, value, label,uid]
### output [seg1,seg2,...],[label1,label2,...], [uid1,uid2,....] of which seg1 = [v1,v2,...vn], n is the seg length
def data_segmentation(data, segLen, lag):
    result = []
    activity_label = []
    uid = []
    current_uid = ''
    activity = ''
    window_index = -1
    i = 0
    while i < len(data)-segLen + 1:
        time_seq = [x[0] for x in data[i:i+segLen]]
        data_seq = [x[1] for x in data[i:i+segLen]]
        activity_seq = [x[2] for x in data[i:i+segLen]]
        uid_seq = [x[3] for x in data[i:i+segLen]]
        if isIncrement(time_seq) and isUnique(activity_seq) and isUnique(uid_seq):
            result.append(data_seq)
            activity_label.append(activity_seq[0])
            uid.append(uid_seq[0])
            i = i + lag
        else:
            i += 1
    return result, activity_label,uid

## frequency domain
def calc_fft(sample):
    fft_real = list(map(lambda x:np.real(x), np.fft.fft(sample)))
    fft_image = list(map(lambda x:np.imag(x), np.fft.fft(sample)))
    offset = fft_real[0]
    #print('len(sample) = ',len(sample))
    if len(fft_real) > 3:
        fft_real = fft_real[1:int(len(sample)/2)]
        fft_image = fft_image[1:int(len(sample)/2)]
    elif len(fft_real) == 1:
        fft_real = [0]
        fft_image = [0]
    else:
        fft_real = [fft_real[1]]
        fft_image = [fft_image[1]]
    energy = [np.sqrt(fft_real[i]**2 + fft_image[i] ** 2) for i in range(len(fft_real)) ]
    return energy, offset

def generate_quantile(sample):
    return np.percentile(sample,[20,40,60,80]).tolist()

## frequncy domain: fft histogram
def generate_fft_histogram(energy):
    hist_fft = []
    if len(energy) != 0:
        norm_a = normalize(energy)
        for i in np.arange(0,1,0.1):
            hist_fft.append(len([a for a in norm_a if (a>=i and a<=i+0.1)]))
    return hist_fft


def generate_stat(sample):
    return [max(sample), min(sample), np.std(sample), np.average(sample)]


def generate_singleaxis_fv(segment):
    stats_fs = generate_stat(segment)
    energy,offset = calc_fft(segment)
    freq = max(energy)
    energy_std = np.std(energy)
    freq_hist = generate_fft_histogram(energy)
    quantile = generate_quantile(segment)
    x = stats_fs + [offset.item(0) , freq, energy_std] + freq_hist + quantile
    return x



def generate_screen_brightness_fv(screen_brightness_segment):
    screen_brightness_stats_fs = generate_stat(screen_brightness_segment)
    return screen_brightness_stats_fs

def generate_pressure_fv(pressure_segment):
    pressure_stats_fs = generate_stat(pressure_segment)
    return pressure_stats_fs
def generate_magneticfield_magnitude_fv(mag_magnitude_segment):
    mag_magnitude_fs = generate_stat(mag_magnitude_segment)
    mag_magnitude_energy, mag_magnitude_offset = calc_fft(mag_magnitude_segment)
    mag_magnitude_freq = max(mag_magnitude_energy)
    return mag_magnitude_fs + [mag_magnitude_freq, mag_magnitude_offset.item(0)]

def generate_acc_magnitude_fv(acc_magnitude_segment):
    acc_magnitude_stats_fs = generate_stat(acc_magnitude_segment)
    acc_magnitude_energy,acc_magnitude_offset = calc_fft(acc_magnitude_segment)
    acc_magnitude_freq = max(acc_magnitude_energy)
    acc_magnitude_energy_std = np.std(acc_magnitude_energy)
    acc_magnitude_freq_hist = generate_fft_histogram(acc_magnitude_energy)
    acc_magnitude_quantile = generate_quantile(acc_magnitude_segment)
    return acc_magnitude_stats_fs + [acc_magnitude_offset.item(0),acc_magnitude_freq,acc_magnitude_energy_std] + acc_magnitude_freq_hist + acc_magnitude_quantile

def from_orig_to_segments(orig_file,segLen,lag):
    df = pd.read_csv(orig_file)
    acc_x = df[['timestamp','acc_x','label','uid']].as_matrix()
    acc_y = df[['timestamp','acc_y','label','uid']].as_matrix()
    acc_z = df[['timestamp','acc_z','label','uid']].as_matrix()
    rt_x = df[['timestamp','rotation_rate_x','label','uid']].as_matrix()
    rt_y = df[['timestamp','rotation_rate_y','label','uid']].as_matrix()
    rt_z = df[['timestamp','rotation_rate_z','label','uid']].as_matrix()
    acc_magnitude = df[['timestamp','magnitude_acc','label','uid']].as_matrix()
    mag_magnitude = df[['timestamp','magnitude_mag','label','uid']].as_matrix()
    pressure = df[['timestamp','pressure','label','uid']].as_matrix()
    screen_brightness = df[['timestamp','screen_brightness','label','uid']].as_matrix()
    acc_x_segment,acc_x_label,acc_x_uid = data_segmentation(acc_x,segLen,lag)
    acc_y_segment,acc_y_label,acc_y_uid = data_segmentation(acc_y,segLen,lag)
    acc_z_segment,acc_z_label,acc_z_uid = data_segmentation(acc_z,segLen,lag)
    rt_x_segment,rt_x_label,rt_x_uid = data_segmentation(rt_x,segLen,lag)
    rt_y_segment,rt_y_label,rt_y_uid = data_segmentation(rt_y,segLen,lag)
    rt_z_segment,rt_z_label,rt_z_uid = data_segmentation(rt_z,segLen,lag)
    acc_magnitude_segment,acc_magnitude_label,acc_magnitude_uid = data_segmentation(acc_magnitude,segLen,lag)
    mag_magnitude_segment,mag_magnitude_label,mag_magnitude_uid = data_segmentation(mag_magnitude,segLen,lag)
    pressure_segment,pressure_label,pressure_uid = data_segmentation(pressure,segLen,lag)
    screen_brightness_segment,screen_brightness_label,screen_brightness_uid = data_segmentation(screen_brightness,segLen,lag)
    if (not acc_x_label==acc_y_label==acc_z_label==rt_x_label == rt_y_label == rt_z_label==acc_magnitude_label==mag_magnitude_label==pressure_label == screen_brightness_label):
        raise AssertionError("labels are not consistent!")
    if (not acc_x_uid==acc_y_uid==acc_z_uid==rt_x_uid==rt_y_uid==rt_z_uid==acc_magnitude_uid==mag_magnitude_uid==pressure_uid==screen_brightness_uid):
        raise AssertionError("uids are not consistent!")
    return acc_x_segment, acc_y_segment,acc_z_segment,rt_x_segment,rt_y_segment,rt_z_segment,acc_magnitude_segment,mag_magnitude_segment,pressure_segment,screen_brightness_segment,acc_x_label,acc_x_uid

def generate_feature_vector(orig_file,segLen,lag,isAcc,isRotationRate,isBrightness,isPressure,isMagnetic,isAddUID):
    log("Begin generating feature vector")
    log_red("file to read in is: " + orig_file)
    acc_x_segment, acc_y_segment,acc_z_segment,rt_x_segment,rt_y_segment,rt_z_segment,acc_magnitude_segment,mag_magnitude_segment,pressure_segment,screen_brightness_segment,label_list,uid_list = from_orig_to_segments(orig_file,segLen,lag)
    fv_list = []
    for i in range(len(acc_x_segment)):
        fv = []
        if isAcc:
            fv += generate_singleaxis_fv(acc_x_segment[i])
            fv += generate_singleaxis_fv(acc_y_segment[i])
            fv += generate_singleaxis_fv(acc_z_segment[i])
            fv += generate_acc_magnitude_fv(acc_magnitude_segment[i])
        if isRotationRate:
            fv += generate_singleaxis_fv(rt_x_segment[i])
            fv += generate_singleaxis_fv(rt_y_segment[i])
            fv += generate_singleaxis_fv(rt_z_segment[i])
        if isBrightness:
            fv += generate_screen_brightness_fv(screen_brightness_segment[i])
        if isPressure:
            fv += generate_pressure_fv(pressure_segment[i])
        if isMagnetic:
            fv += generate_magneticfield_magnitude_fv(mag_magnitude_segment[i])
        if isAddUID:
            fv.append(uid_list[i])
        fv.append(label_list[i])
        fv_list.append(fv)
    return fv_list

def generate_arff_header(filestream,travel_mode_list,isAcc,isRotationRate, isBrightness, isPressure, isMagnetic,isUID):
    log_purple("begin write header")
    filestream.write('@relation travel_mode_detection_labeled\n\n')
    class_string = ', '.join('"{0}"'.format(w) for w in travel_mode_list)
    if isAcc:
        for i in ['X','Y','Z','ACC_MAGNITUDE_']:
            filestream.write('@attribute "'+i+'MAX" numeric\n')
            filestream.write('@attribute "'+i+'MIN" numeric\n')
            filestream.write('@attribute "'+i+'STND" numeric\n')
            filestream.write('@attribute "'+i+'AVG" numeric\n')
            filestream.write('@attribute "'+i+'OFFSET" numeric\n')
            filestream.write('@attribute "'+i+'FRQ" numeric\n')
            filestream.write('@attribute "'+i+'ENERGYSTND" numeric\n')
            for j in range(10):
                    filestream.write('@attribute "'+i+str(j)+'" numeric\n')
            for k in [20,40,60,80]:
                filestream.write('@attribute "'+i + 'QUANTILE' +str(k)+'" numeric\n')
    if isRotationRate:
        for i in ['X_Rotation_','Y_Rotation','Z_Rotation']:
            filestream.write('@attribute "'+i+'MAX" numeric\n')
            filestream.write('@attribute "'+i+'MIN" numeric\n')
            filestream.write('@attribute "'+i+'STND" numeric\n')
            filestream.write('@attribute "'+i+'AVG" numeric\n')
            filestream.write('@attribute "'+i+'OFFSET" numeric\n')
            filestream.write('@attribute "'+i+'FRQ" numeric\n')
            filestream.write('@attribute "'+i+'ENERGYSTND" numeric\n')
            for j in range(10):
                    filestream.write('@attribute "'+i+str(j)+'" numeric\n')
            for k in [20,40,60,80]:
                filestream.write('@attribute "'+i + 'QUANTILE' +str(k)+'" numeric\n')
    if isBrightness:
        filestream.write('@attribute "Screen_brightness_MAX" numeric\n')
        filestream.write('@attribute "Screen_brightness_MIN" numeric\n')
        filestream.write('@attribute "Screen_brightness_STND" numeric\n')
        filestream.write('@attribute "Screen_brightness_AVG" numeric\n')
    if isPressure:
        filestream.write('@attribute "pressure_MAX" numeric\n')
        filestream.write('@attribute "pressure_MIN" numeric\n')
        filestream.write('@attribute "pressure_STND" numeric\n')
        filestream.write('@attribute "pressure_AVG" numeric\n')
    if isMagnetic:
        filestream.write('@attribute "magEnergy_MAX" numeric\n')
        filestream.write('@attribute "magEnergy_MIN" numeric\n')
        filestream.write('@attribute "magEnergy_STND" numeric\n')
        filestream.write('@attribute "magEnergy_AVG" numeric\n')
        filestream.write('@attribute "magEnergy_FREQ" numeric\n')
        filestream.write('@attribute "magEnergy_OFFSET" numeric\n')
    if isUID:
        filestream.write('@attribute "UID" numeric\n')
    filestream.write('@attribute class{ ' + class_string + '}\n\n')
    filestream.write('@data\n')

def write_data(filestream,fvlist):
    for fv in fvlist:
        filestream.write(','.join([str(e) for e in fv])+'\n')

def generate_data_file(orig_file,segLen,lag,isAcc,isRotationRate,isBrightness,isPressure,isMagnetic,isAddUID,isNormalize,travel_mode_list,isAndroid,isArff,season):
    fv_list = generate_feature_vector(orig_file,segLen,lag,isAcc,isRotationRate,isBrightness,isPressure,isMagnetic,isAddUID)
    if len(travel_mode_list) == 2:
        mode = 'unwheeled'
    elif len(travel_mode_list) == 4:
        mode = 'wheeled'
    else:
        mode = 'allmode'
    if lag == segLen:
        slide = 'noSlide'
    else:
        slide = 'lag' + str(lag)
    sensors = ''
    if isAcc:
        sensors += 'a'
    if isRotationRate:
        sensors += 'r'
    if isBrightness:
        sensors += 'b'
    if isPressure:
        sensors += 'p'
    if isMagnetic:
        sensors += 'm'
    if isNormalize:
        normalize = 'normalize'
    else:
        normalize = 'raw'
    if isAndroid:
        phone_type = 'Android'
    else:
        phone_type = 'iPhone'
    if isAddUID:
        write_to_file = phone_type + '_' + season + '_' + mode + '_' + 'segLen'+str(segLen) + '_' + slide + '_' + sensors + '_' + normalize + '_uid_' + date.today().isoformat()

    else:
        write_to_file = phone_type + '_' + season + '_' + mode + '_' + 'segLen'+str(segLen) + '_' + slide + '_' + sensors + '_' + normalize + '_' + date.today().isoformat()
    if isArff:
        write_to_file = write_to_file + '.arff'
    else:
        write_to_file = write_to_file + '.csv'
    log_blue("\nTarget file: " + write_to_file)
    fs = open('../data/'+write_to_file,'w')
    if isArff:
        generate_arff_header(fs,travel_mode_list,isAcc,isRotationRate,isBrightness,isPressure,isMagnetic,isAddUID)
    write_data(fs,fv_list)
    log_yellow('File finished writing!')
    fs.close()

def main(args):
    global log_level
    config_file = args[2]
    with open(config_file,'r') as configF:
        config = json.load(configF)
    log_level = config['log_level']
    #segLen = config['segmentation']['segLen']
    lag = config['segmentation']['lag']
    setting = config['setting']
    segLen = 8
    #for lag in range(1,segLen+1):
    isAcc = config['sensors']['isAcc']
    isRotationRate = config['sensors']['isRotationRate']
    isBrightness = config['sensors']['isBrightness']
    isPressure = config['sensors']['isPressure']
    isMagnetic = config['sensors']['isMagnetic']
    isNormalize = config['segmentation']['isNormalization']
    isAddUID = config['segmentation']['isAddUID']
    isAndroid = (config[setting]['os'] == 'android')
    orig_file = config[setting]['file_name']
    isArff = config['file']['isArff']
    season = config[setting]['season']
    mode_list = ['car','bus','bike','subway','walk','jog']
    generate_data_file(orig_file,segLen,lag,isAcc,isRotationRate,isBrightness,isPressure,isMagnetic,isAddUID,isNormalize,mode_list,isAndroid,isArff,season)



if __name__ == '__main__':
    print("inside featureVectorGenerator.py")
    main(sys.argv)
