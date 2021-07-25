# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:37:08 2019

@author: Yu-Cheng, Liang 490858
"""
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io.wavfile
import librosa
import librosa.display
from pydub.utils import make_chunks
import csv

bird_list_path = 'P:\\kit306\\Assignment1\\bird\\bird' 
bird_lists = os.listdir(bird_list_path)

file_name = 'bird_sound.csv'
#---------------check whether csv file is exist or not-------------------------
bird_csv_file_path = 'P:\\kit306\\Assignment1\\bird\\'+file_name
try:
    os.remove(bird_csv_file_path)
    print("Removed bird .csv file!")
except FileNotFoundError:
    pass

#---------------------Making audion time length are same----------------------- 
def split_time_len(target_time,data_sound):
    duration = librosa.get_duration(data_sound)
    #print("bird sound duration: ",duration)
    split_gap = int((target_time/duration)*len(data_sound))
    chunks = make_chunks(data_sound,split_gap)
    #print("first one chunk len: ",librosa.get_duration(chunks[0]))
    #print("last one chunk len: ",librosa.get_duration(chunks[len(chunks)-1]))
    return chunks[0:len(chunks)-1] #remove last one file

def extend_time_len(target_time,data_sound):
    duration = librosa.get_duration(data_sound)
    extend_times = int(target_time/duration)
    new_data_sound = data_sound.tolist()
    for i in range(0,extend_times-1): #if it didn't -1, it will over target time
        new_data_sound.extend(data_sound.tolist())
    n_duration = librosa.get_duration(new_data_sound) #len(new_data_sound)/sample_rate
    #calculate rest of time 
    rest_need_time = target_time - n_duration
    #grab specific range of array in rest_need_time
    rest_need_points = int((rest_need_time/duration)*len(data_sound))
    new_data_sound.extend(data_sound[0:rest_need_points].tolist())
    #print(librosa.get_duration(new_data_sound))
    return np.array(new_data_sound) #return extend bird sound
#-------------------------------------------------------------------------------

#------------------------filter specific range of freq-------------------------
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    #print(low)
    #print(high)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

#filter range of frequency
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

#------------------------------------------------------------------------------

#-----------------------calculate mfcc and other features----------------------
'''
simply checking whether max in the meddle or not
if max in the middle, it is normal distribution
otherwise, it is skewed curve
'''
def is_mean(arr):
    arr_max = np.max(arr).item()
    key = list(arr).index(arr_max)
    #print('key: ',key)
    lower = len(arr)/3
    #print('low: ', lower)
    upper = len(arr)*2/3
    #print('up: ',upper)
    if key<upper and key>lower:
        return True
    else:
        return False

#cal value root-square-average 
def rms_mfcc(mfcc):
    return np.sqrt(np.mean(np.asarray(mfcc)**2))

#mfcc 
def cal_features(mfcc):
    d_mfcc = []
    for m in mfcc:
        if is_mean(m):
            value = np.mean(m)
        else:
            value = np.median(m)
        d_mfcc.append(value)
    #plt.figure(figsize=(14,5))
    #plt.plot(d_mfcc)
    return d_mfcc

def zcr_median(y):
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return np.median(zcr)

def spectral_centroid_median(y,sr):
    cent = librosa.feature.spectral_centroid(y,sr=sr)[0]
    return np.median(cent)

def mfcc_add_delta(mfcc,delta):
    d_mfcc = []
    for m,d in zip(mfcc,delta): #zip is used for parallel iteration
        temp = []
        for x, y in zip(m,d):
            temp.append(x+y)
        d_mfcc.append(temp)
    return np.asarray(d_mfcc)

def delta_add_mfcc_compute(y,sr):
    mfcc = librosa.feature.mfcc(y,sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta_mfcc = mfcc_add_delta(mfcc,delta) 
    return cal_features(delta_mfcc)

def mfcc_compute(y,sr):
    mfcc = librosa.feature.mfcc(y,sr=sr, n_mfcc=13) #get mfcc
    return cal_features(mfcc) #cal average each n_mfcc
#------------------------------------------------------------------------------

#-----------------------------------main function------------------------------
#calculate each bird median time
each_bird_time = {}
for bird in bird_lists:
    print('Calculating :',bird,' median time')
    b_audio_files = os.listdir(bird_list_path+'\\'+bird)
    bird_time = []
    for audio in b_audio_files:
        rate, data = scipy.io.wavfile.read(bird_list_path+'\\'+bird+'\\'+audio)
        duration = len(data)/rate
        bird_time.append(duration)
    median_time = round(np.median(bird_time),1)
    each_bird_time.update({bird:median_time})
    #break

#------------------------------------write csv file----------------------------
# Write csv file - Opens a file for writing, creates the file if it does not exist
with open(bird_csv_file_path,'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Bird_Name','File Name','Splitted File Index','Sample Rate','Zero Crossing Rate median','Spectral Centroid median','mfcc mean','mfcc median','mfcc RMS','mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10','mfcc_11','mfcc_12','mfcc_13','Delta-mfcc mean','Delta-mfcc median','Delta-mfcc rms','Delta-mfcc_1','Delta-mfcc_2','Delta-mfcc_3','Delta-mfcc_4','Delta-mfcc_5','Delta-mfcc_6','Delta-mfcc_7','Delta-mfcc_8','Delta-mfcc_9','Delta-mfcc_10','Delta-mfcc_11','Delta-mfcc_12','Delta-mfcc_13','Duration'])
    print("Created bird csv file!!")
    #print(bird_types) #debug
    for bird in bird_lists:
        print('--------Start writing ',bird,' all data-----------')
        b_audio_files = os.listdir(bird_list_path+'\\'+bird)
        for audio in b_audio_files:
            data , sr = librosa.load(bird_list_path+'\\'+bird+'\\'+audio)
            median_time = each_bird_time.get(bird)
            duration = round(librosa.get_duration(data),1)
            if duration > median_time:
                for i,y in enumerate(split_time_len(median_time,data)):
                    #------------------Main------------------------------------
                    filter_y = butter_bandpass_filter(y,1000,7000,sr) #filter under 1000hz and above 7000hz
                    d_mfcc = mfcc_compute(filter_y,sr)
                    zcr = zcr_median(filter_y)
                    cent = spectral_centroid_median(filter_y,sr)
                    delta_mfcc = delta_add_mfcc_compute(filter_y,sr)
                    filewriter.writerow([bird,audio,i,sr,zcr,cent,np.mean(d_mfcc),np.median(d_mfcc),rms_mfcc(d_mfcc),d_mfcc[0],d_mfcc[1],d_mfcc[2],d_mfcc[3],d_mfcc[4],d_mfcc[5],d_mfcc[6],d_mfcc[7],d_mfcc[8],d_mfcc[9],d_mfcc[10],d_mfcc[11],d_mfcc[12],np.mean(delta_mfcc),np.median(delta_mfcc),rms_mfcc(delta_mfcc),delta_mfcc[0],delta_mfcc[1],delta_mfcc[2],delta_mfcc[3],delta_mfcc[4],delta_mfcc[5],delta_mfcc[6],delta_mfcc[7],delta_mfcc[8],delta_mfcc[9],delta_mfcc[10],delta_mfcc[11],delta_mfcc[12],librosa.get_duration(y)])
                    #----------------------------------------------------------
            elif duration < median_time:
                #------------------Main----------------------------------------
                extend_y = extend_time_len(median_time,data)
                filter_y = butter_bandpass_filter(extend_y,1000,7000,sr) #filter under 1000hz and above 7000hz
                d_mfcc = mfcc_compute(filter_y,sr)
                zcr = zcr_median(filter_y)
                cent = spectral_centroid_median(filter_y,sr)
                delta_mfcc = delta_add_mfcc_compute(filter_y,sr)
                filewriter.writerow([bird,audio,'',sr,zcr,cent,np.mean(d_mfcc),np.median(d_mfcc),rms_mfcc(d_mfcc),d_mfcc[0],d_mfcc[1],d_mfcc[2],d_mfcc[3],d_mfcc[4],d_mfcc[5],d_mfcc[6],d_mfcc[7],d_mfcc[8],d_mfcc[9],d_mfcc[10],d_mfcc[11],d_mfcc[12],np.mean(delta_mfcc),np.median(delta_mfcc),rms_mfcc(delta_mfcc),delta_mfcc[0],delta_mfcc[1],delta_mfcc[2],delta_mfcc[3],delta_mfcc[4],delta_mfcc[5],delta_mfcc[6],delta_mfcc[7],delta_mfcc[8],delta_mfcc[9],delta_mfcc[10],delta_mfcc[11],delta_mfcc[12],librosa.get_duration(extend_y)])
                #----------------------------------------------------------
            else:
                #------------------Main----------------------------------------
                filter_y = butter_bandpass_filter(data,1000,7000,sr) #filter under 1000hz and above 7000hz
                d_mfcc = mfcc_compute(filter_y,sr)
                zcr = zcr_median(filter_y)
                cent = spectral_centroid_median(filter_y,sr)
                delta_mfcc = delta_add_mfcc_compute(filter_y,sr)
                filewriter.writerow([bird,audio,'',sr,zcr,cent,np.mean(d_mfcc),np.median(d_mfcc),rms_mfcc(d_mfcc),d_mfcc[0],d_mfcc[1],d_mfcc[2],d_mfcc[3],d_mfcc[4],d_mfcc[5],d_mfcc[6],d_mfcc[7],d_mfcc[8],d_mfcc[9],d_mfcc[10],d_mfcc[11],d_mfcc[12],np.mean(delta_mfcc),np.median(delta_mfcc),rms_mfcc(delta_mfcc),delta_mfcc[0],delta_mfcc[1],delta_mfcc[2],delta_mfcc[3],delta_mfcc[4],delta_mfcc[5],delta_mfcc[6],delta_mfcc[7],delta_mfcc[8],delta_mfcc[9],delta_mfcc[10],delta_mfcc[11],delta_mfcc[12],duration])
                #--------------------------------------------------------------
        print('----------------',bird,' Finished!!---------------')
        #break


