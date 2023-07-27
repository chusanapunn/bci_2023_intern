import pylsl
import math as math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from typing import List
import scipy.signal as sig
import scipy.fftpack as fft
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pywt as pywt

import rtfs_MarkerUI as mui
# import new_marker_pg as mkui

import sys
import os as os

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from mne.decoding import ( Scaler, cross_val_multiscore, Vectorizer)     
from sklearn.metrics import classification_report, confusion_matrix

plot_duration = 5 # how many seconds of data to show
update_interval = 20  # ms between screen updates
pull_interval = 60 # ms between each pull operation
fft_interval = 500 # ms between each FFT calculation and triggering
global rawSignalArray,unicorn,unitz,triggerCode,marker_ui,channel_count,fftSignalArray,analyzeChannel,raw_info,epochs,raw,saveName,fn
triggerCode=0
markerSignalArray = []
codearray=[]
inputSignalCounter=0
record=False
att_count=0
rlx_count=0
signalDataCount=0
loadraw=False
unicorn = False
unitz = 1000000
channel_count=0
analyzeChannel=0


app=QtWidgets.QApplication(sys.argv)

def display_confusionMatrix(clf):
    global y_train,epochs
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    labels = y_train
    # Do cross-validation
    labels = np.where(labels>1500,0,.99)
    
    
    preds = np.empty(len(labels))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    
    for train, test in cv.split(epochs, labels):

        try:
            clf.fit(epochs.get_data()[train], labels[train])
        except AttributeError:
            clf.fit(epochs[train],labels[train])
        try:
            preds[test] = clf.predict(epochs.get_data()[test])
        except ValueError:
            preds = preds.reshape(-1,1)
            preds[test] = clf.predict(epochs.get_data()[test])
        except AttributeError:
            preds = preds.reshape(-1,1)
            preds[test] = clf.predict(epochs[test])
    # print("Preds",preds)
    preds[test]=preds[test].round(2)
    # print("Preds",np.ravel(preds))
    preds = np.where(preds>=np.median(preds),.99,0)
    preds = label_encoder.fit_transform(np.ravel(preds))
    # print("Preds",preds)
    # Classification report
    try:
        target_names = epochs.ch_names
    except AttributeError:
        target_names = ["Name"]*epochs.shape[0]

    report = classification_report(labels, preds, target_names=target_names)
    print(report)

    # Normalized confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig, ax = plt.subplots(1)
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(title="Normalized Confusion matrix")
    fig.colorbar(im)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    fig.tight_layout()
    ax.set(ylabel="True label", xlabel="Predicted label")
    plt.show()

def printTrainshape(x_train,y_train,epochs):
    print("___________")
    print("xtrainShape: ",x_train.shape)
    # print("Xtrain:", x_train)
    print("ytrainshape: ",y_train.shape)
    # print("Ytrain:", y_train)
    try:
        print("Epoch Info: ",epochs.info)
    except AttributeError: 
        print("Fake Epoch Shape",epochs.shape)
    print("___________")
    
def set_buffer(info,dtypes,bufferText):
    global channel_count
    bufsize = (math.ceil(1.5*math.ceil(info.nominal_srate()*plot_duration)), channel_count+1)
    buffer = np.empty(bufsize, dtype=dtypes[info.channel_format()])
    empty = np.array([])
    bufferText.setText("bufferSize is: "+str(bufsize[0])+". Stream Channel Count:"+str(bufsize[1]))

    # rawSignalArray=np.empty(bufsize, dtype=dtypes[info.channel_format()])
    # print("RS",rawSignalArray.transpose())
    return bufsize,buffer,empty

def butter_bandstop_filter(data, fs, order, a, b):
        # Get the filter coefficients  
        b, a = sig.butter(order, [a,b], 'bandstop', fs=fs, output='ba')
        y = sig.filtfilt(b, a, data, padlen= len(data)-1)
        return y

def butter_bandpass_filter(data, fs, order, a, b):
        # Get the filter coefficients  
        b, a = sig.butter(order, [a,b], 'bandpass', fs=fs, output='ba')
        y = sig.filtfilt(b, a, data, padlen= len(data)-1)
        return y

def butter_lowpass_filter(data, cutOff, fs, order):
        
        # Get the filter coefficients 
        b_lp, a_lp = sig.butter(order, cutOff, 'lowpass', fs=fs, output='ba')
        y = sig.filtfilt(b_lp, a_lp, data, padlen= len(data)-1)
        return y


def butter_highpass_filter(data, cutOff, fs, order):
        nyq=0.5*fs
        normcutoff = cutOff/nyq
        # Get the filter coefficients 
        b_hp, a_hp = sig.butter(order, normcutoff, 'highpass', fs=fs, output='ba')
        y = sig.filtfilt(b_hp, a_hp, data, padlen= len(data)-1)
        return y

    # tmarray = []

def changeCode(code,tch:pg.PlotItem,fixationText,fixationText2):
    global markerSignalArray,record,att_count,codearray,rlx_count
    labela=["Up","Down",'Right',"Left"]
    label=None
    label2=None

    if (code==0):
        # label=random.choice(labela)
        label=None
    elif(code==96):
        # if(markerSignalArray):
        #     print(markerSignalArray.pop())
        label="Start Recording"
        record=True
        codearray=[]
        
    elif(code==97 and record):
        if(markerSignalArray):
            markerSignalArray.pop()
        label="Stop Recording"
        record=False
        
    elif (code==1000):
        label="Attention :" +str((att_count+1))
        att_count+=1
        print(label)
        if(markerSignalArray):
            markerSignalArray.pop()  
        
    elif (code == 2000):
        label2="Relax :" +str((rlx_count+1))
        rlx_count+=1
        print(label2)
        if(markerSignalArray):
            markerSignalArray.pop()
        
    else:
        label=None
        label2=None
        code=0
    time_now=pylsl.local_clock()
    
    if (label and label not in labela):
        dispLabel=None
        
        if (code==1000):
            fixationText.setText(label)
            dispLabel = label

        elif (code==2000):
            fixationText2.setText(label2)
            dispLabel = label2
        else:
            fixationText2.setText(label2)
            dispLabel =label2

        mcurve=pg.InfiniteLine(time_now, angle=90, movable=False, label=dispLabel,pen="Red")
        tch.addItem(mcurve)

    if (record):
        markerSignalArray.append(code)
        # print("Marker:",markerSignalArray)
        if (code!=0):
            codearray.append(code)
            # print("Code:",codearray)
            # if (markerSignalArray>rawSignalArray):
                # markerSignalArray.pop()
        # if (code==96):
        #     if(markerSignalArray):
        #         print(markerSignalArray.pop())
    # codearray=markerSignalArray[np.where(markerSignalArray!=0)]
    # print(codearray)


class Inlet:
    def __init__(self,info:pylsl.StreamInfo):
        global fs,nyq, channel_count, rawSignalArray, fftSignalArray
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        
        self.name = info.name()
        channel_count = info.channel_count()-1

        rawSignalArray=[[]for _ in range(channel_count)]
        fftSignalArray = [[]for _ in range(channel_count)]
        
        fs = info.nominal_srate()
        nyq=1/2*fs   
        
    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        pass
    def plot_fft(self,ch):
        pass


class DataInlet(Inlet):  
    def __init__(self,info:pylsl.StreamInfo,plt:pg.PlotItem,fftplt:pg.PlotItem,filtplt:pg.PlotItem,bufferText):  #,filtfftplt:pg.PlotItem
        global rawSignalArray,fftSignalArray,raw
        super().__init__(info)
        dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
        self.bufsize,self.buffer,self.empty=set_buffer(info,dtypes,bufferText)
        
        # signal Curves
        self.curves = [pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(channel_count)]
        # self.fftcurves= [pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtcurves= [pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(channel_count)]
        # self.filtfftcurves=[pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(self.channel_count)]

        # triggerValA=1.3
        # triggerValB=0.5
        # triggerLine=pg.InfiniteLine(label="SimpleTriggerLine",pos=triggerValA,movable=True,angle=0)
        
        for curve in self.curves:
            plt.addItem(curve)
            # plt.addItem(triggerLine)
        # for fftcurve in self.fftcurves:
        #     fftplt.addItem(fftcurve)
        for filtcurve in self.filtcurves:
            filtplt.addItem(filtcurve)
            # filtplt.addItem(triggerLine)
        # for filtfftcurve in self.filtfftcurves:
        #     filtfftplt.addItem(filtfftcurve)

    def pull_and_plot(self, plot_time, ch,tch,DB_text,DB_markertext,fixationText,fixationText2,dataAmountText,unicorn):
        global signalDataCount,unitz,fftSignalArray
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        this_x=None
        global inputSignalCounter,rawSignalArray,this_y
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0:ts.size, :]
            this_x = np.array([])
            old_offset = 0
            new_offset = 0
            old_x, old_y = self.curves[ch].getData()
            old_offset = old_x.searchsorted(plot_time)
            new_offset = ts.searchsorted(plot_time)
            


            # print("bef",y.max())
            if unicorn:
                # this_y = np.asarray([a/unitz for a in this_y])
                # this_y = this_y/unitz
                # filtered_thisY = np.asarray([b/unitz for b in filtered_thisY])
                # filtered_thisY=filtered_thisY/unitz
                y =  np.asarray([c/unitz for c in y])
                # y=y/unitz
            

            this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
            this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch] - ch))

            # print("af",y.max())
            


             # Filter and plot signal
            if (ch==0):     # channel 1 filter
                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
                # filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)
                # filtered_thisY=butter_lowpass_filter(this_y,5,fs,2)
                filtered_thisY=butter_bandstop_filter(this_y,fs,6,49,51)
                # print("skip filter")
                # filtered_thisY = this_y
                
            elif (ch==1):     # channel 1 filter
                filtered_thisY=butter_bandstop_filter(this_y,fs,6,47,53)
                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
                # filtered_thisY = this_y

            elif (ch==2):     # channel 1 filter
                # filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)
                filtered_thisY = this_y
            else:
                filtered_thisY=this_y
            
            

            self.curves[ch].setData(this_x, this_y)
            self.filtcurves[ch].setData(this_x,filtered_thisY)
            # print("thisY",this_y.size)
            # print("ts",ts.size)
            # print("thisY",len(this_y))
            # print("Filter Y", len(filtered_thisY))
            # rawSignalArray=np.hstack((rawSignalArray[old_offset:],filtered_thisY[new_offset:]))
            for i in range(ts.size):
                fftSignalArray[ch].append(y[i,ch])
                # print("Assign fftSignalArray:",fftSignalArray)
                signalDataCount+=1
                if (record):
                    # for ii in range(ts.size):   
                    # rawSignalArray = np.array(rawSignalArray)
                    # y = np.array(y)
                    # print("rawSignalArray:",rawSignalArray[ch].shape,"Y:",y[i,ch].shape)
                    rawSignalArray[ch].append(y[i,ch])
                    # print(rawSignalArray)
                    # print("After : ",rawSignalArray)
                    if (ch==0):
                        inputSignalCounter+=1
                        changeCode(0,tch,fixationText,fixationText2)  # To stimulus channels
                
            dataAmountText.setText("Data Input Amount: "+str(int(inputSignalCounter)))
            try :
                DB_text.setText("Raw Data Count: " + str(len(rawSignalArray[ch])))
            except IndexError:
                DB_text.setText("Raw Data Count: " + str(len(rawSignalArray[ch])))
            DB_markertext.setText("Marker Data Count: " + str(len(markerSignalArray)))
        
    def convertToRaw(self,unicorn):
        global rawSignalArray,markerSignalArray,channel_count,fs,filtraw,ch1_pick,ch2_pick,raw,raw_info
        ch_names=[f"eeg{n:02}" for n in range(1,channel_count+1)]
        ch_types=["eeg"]*(channel_count)
        stim_ch_names= ["Stim"]     
        info = mne.create_info(ch_names, sfreq=fs,ch_types=ch_types)
        stim_info = mne.create_info(stim_ch_names, sfreq=fs,ch_types="stim")
        
        print("Try converting to Raw")
        # rawarray=np.array(rawSignalArray)
        
        markarray=np.array(markerSignalArray)
        markarray=markarray.reshape(1,len(markerSignalArray))
        print(info)

        rawSignalArrayN = np.linalg.norm(rawSignalArray)
        rawSignalArray = rawSignalArray/rawSignalArrayN

        raw = mne.io.RawArray(rawSignalArray, info)
        stim_raw = mne.io.RawArray(markarray,stim_info)
        
        ch1_pick=mne.pick_channels(ch_names=ch_names,include=["eeg01"])
        ch2_pick=mne.pick_channels(ch_names=ch_names,include=["eeg02"])
        # filter
        # filtraw=raw.copy().notch_filter(11,picks=ch1_pick,n_jobs=1)
        if (unicorn):
            powerline = (50,100)
            ori_raw=raw.copy()
            filtraw=ori_raw.notch_filter(powerline,n_jobs=1)
            filtraw=filtraw.filter(0.1,50)
        else:
            filtraw=raw.copy()
            filtraw=filtraw.notch_filter(20,n_jobs=1)
        # ica = mne.preprocessing.ICA(n_components=2,random_state=9030,max_iter=100)

        filtraw.add_channels([stim_raw], force_update_info=True)
        raw.add_channels([stim_raw], force_update_info=True)

        raw_info =raw.info

        print(raw.info)
        if unicorn:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title="Raw Signal-non scale")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title = "Filtered Signal-non scale")
        else:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,scalings="auto",title="Raw Signal")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,scalings="auto",title = "Filtered Signal")

    def loadraw(self,fn):
        global raw,unicorn,filtraw,channel_count,fs,loadraw,raw_info
        loadraw=True
        raw = mne.io.read_raw_fif(fn,preload=True)
        raw_info =raw.info
        # raw.drop_channels(['l'])
        # raw.drop_channels(["r"])
        print(raw_info)
        filtraw = raw.copy()
        powerline = (50,100)

        filtraw=filtraw.notch_filter(powerline,n_jobs=1)
        filtraw=filtraw.filter(0.1,50)
        # print("the raw",raw)
        # print("the filtraw",filtraw)
        
        # raw = mne.io.RawArray(raw, raw_info)
        # filtraw = mne.io.RawArray(filtraw,raw_info)

        if unicorn:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title="Raw Signal-non scale")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title = "Filtered Signal-non scale")
            # channel_count = raw.info["nchan"]
        else:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,scalings="auto",title="Raw Signal")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,scalings="auto",title = "Filtered Signal")

    def spect_plot(self):
        global filtraw,ch1_pick,ch2_pick,raw
        specraw=raw.compute_psd()
        spectrum = filtraw.compute_psd()
        specraw.plot(average=True,exclude="bads")
        spectrum.plot(average=True,exclude="bads")
        # spectrum.plot(average=True, picks=ch1_pick, exclude='bads')
        # spectrum.plot(average=True, picks=ch2_pick, exclude='bads')

    def find_events(self,unicorn):
        global filtraw,fs,events_dict,events
        events = mne.find_events(filtraw,stim_channel = "Stim")
        print(events[:10])
        print("------Event Done-----")
        # Trial Descriptor
        events_dict={
            "attention":1000,
            "relax":2000
        }

        # Plot Events

        fig = mne.viz.plot_events(
            events=events, sfreq=filtraw.info["sfreq"], first_samp=filtraw.first_samp, event_id=events_dict
        )

        # fig.subplots_adjust(right=0.7)  # make room for legend
        # if (unicorn):
        #     filtraw.plot(events,event_color={1000:"r",2000:"g",100:"b",96:"y"},title="Filtered Events-non scale")

        # else:
        #     filtraw.plot(events,event_color={1000:"r",2000:"g",100:"b",96:"y"},scalings="auto",title="Filtered Events")

    def epoch_plot(self):
        global filtraw,events,events_dict,unicorn,att_epo,rel_epo,x_train,y_trainAtt,y_trainRlx,epochs,fs,att_data,rel_data,raw_info,y_train
        import scipy.stats as stats
        from matplotlib.pyplot import specgram
        epochs = mne.Epochs(filtraw, events, event_id=events_dict, preload=True)

        # epochs.plot(
        #     events=events,event_id=events_dict,title= "Epochs",scalings="auto"
        # )
        # if unicorn:
        #     # epochs.plot(
        #     #     events=events,event_id=events_dict,title= "Sum Epochs-nonscale"
        #     # )
        #     epochs["attention"].plot(
        #         events=events,event_id=events_dict,title= "Attention Epochs-nonscale"
        #     )
        #     epochs["relax"].plot(
        #         events=events,event_id=events_dict,title= "Relax Epochs-nonscale"
        #     )

        
            
        # else:
        #     # epochs.plot(
        #     #     events=events,event_id=events_dict,title= "Sum Epochs",scalings="auto"
        #     # )
        #     epochs["attention"].plot(
        #         events=events,event_id=events_dict,title= "Attention Epochs",scalings="auto"
        #     )
        #     epochs["relax"].plot(
        #         events=events,event_id=events_dict,title= "Relax Epochs",scalings="auto"
        #     )
        epochs.pick_types(eeg=True,exclude="bads")
        # fepochs.pick_types(eeg=True,exclude="bads")
        att_data = epochs["attention"].get_data()
        rel_data = epochs["relax"].get_data()

        att_epo = epochs["attention"]
        rel_epo = epochs["relax"]

        x_train = epochs.get_data()
        y_trainAtt = epochs["attention"].events[:,2]
        y_trainRlx = epochs["relax"].events[:,2]

        y_train = np.append(y_trainAtt,y_trainRlx)

        print("--------Epoch processing done----------")

        NFFT = fs 
        
        # plt.specgram(,NFFT = NFFT, Fs=fs,noverlap=0)
     
        fig, (ax1,ax2) = plt.subplots(2)
        a_data = att_data.flatten()
        r_data = rel_data.flatten()
        # Pxx, freqs, bins, im = specgram(x_train, NFFT=int(NFFT), Fs = int(fs), 
        #                                     cmap='seismic', noverlap=int(NFFT/2), vmin = -10, vmax = -10)
        # plt.subplots(1)
        Pxxa, freqsa, binsa, ima = ax1.specgram(a_data, NFFT=int(NFFT), Fs = int(fs), 
                                            cmap='seismic', noverlap=int(NFFT/2),vmin = -200, vmax = -160)
        fig.colorbar(ima, fraction=0.046, pad=0.04).set_label('Intensity [dB]')
        ax1.set_ylim([0, 50])
                                            
        Pxxr, freqsr, binsr, imr = ax2.specgram(r_data, NFFT=int(NFFT), Fs = int(fs), 
                                            cmap='seismic', noverlap=int(NFFT/2),vmin = -200, vmax = -160)
        fig.colorbar(imr, fraction=0.046, pad=0.04).set_label('Intensity [dB]')
        ax2.set_ylim([0, 50])
        plt.title("Spectrogram")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.show()


        print(f'Shape data for Spectrogram is {a_data.shape}')
        print(f'Pxx shape is {Pxxa.shape}')
        a_data_z = np.zeros((Pxxa.shape[0], Pxxa.shape[1]))
        for i in range(Pxxa.shape[0]):
            a_data_z[i] = stats.zscore(Pxxa[i][:])
        print(f'Z-Score Pxx shape is {np.min(a_data_z)}')

        print(f'Shape data for Spectrogram is {r_data.shape}')
        print(f'Pxx shape is {Pxxr.shape}')
        r_data_z = np.zeros((Pxxr.shape[0], Pxxr.shape[1]))
        for i in range(Pxxr.shape[0]):
            r_data_z[i] = stats.zscore(Pxxr[i][:])
        print(f'Z-Score Pxx shape is {np.min(r_data_z)}')

        fig, (ax3,ax4) = plt.subplots(2)

        ax3.imshow(a_data_z, origin='lower', cmap="seismic",vmin = -10, vmax = 10)
        fig.colorbar(ima).set_label('Intensity [dB]')

        ax3.set_ylim([0, 50])
        plt.title("Z Score Spectrogram Attention")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')

        ax4.imshow(r_data_z, origin='lower', cmap="seismic",vmin = -10, vmax = 10)
        fig.colorbar(imr).set_label('Intensity [dB]')

        ax4.set_ylim([0, 50])
        plt.title("Z Score Spectrogram Relaxation")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')

        plt.show()  


        # freqs = np.logspace(*np.log10([5, 50]), num=16)
        # n_cycles = 2  # different number of cycle per frequency
        # power, itc =  mne.time_frequency.tfr_morlet(
        #     epochs,
        #     freqs=freqs,
        #     n_cycles=n_cycles,
        #     use_fft=True,
        #     return_itc=True,
        #     decim=3,
        #     n_jobs=None,
        #     average = True,
        # )
        # for i in range(x_train.shape[1]):
        #     power.plot([i],baseline=(-0.5,0),mode="logratio",title="Class:"+str(i+1),vmin=-.6,vmax=.6)

        # epochs["attention"].plot_image(
        #     picks="eeg" #,combine="mean"
        #     ,title="Attention Epoch"
        # )
        # epochs["relax"].plot_image(
        #     picks="eeg" #,combine="mean"
        #     ,title="Relax Epoch"
        # )

        # epochs["attention"].compute_psd().plot(average=False,picks='data', exclude='bads')
        # epochs["relax"].compute_psd().plot(average=False,picks='data', exclude='bads')
        
        
        # # ____PLOT EACH EPOCH PSD________________

        # att_shape = epochs["attention"].get_data().shape
        # rlx_shape = epochs["relax"].get_data().shape
        # attaxisrow = math.ceil(att_shape[0]/2)
        # fig, ax = plt.subplots(attaxisrow,2)
        # fig.set_size_inches(8.5, 12.5, forward=True)
        # for j,i in enumerate (epochs["attention"]):
        #     p_stim = np.zeros((1,len(i[1])))
        #     # print(p_stim.shape)
        #     i=np.append(i,p_stim,axis=0)
        #     # i=np.r_[i,[p_stim]]
        #     # print("i new",i.shape)
        #     # np.append(i,[np.zeros((1,len(filtraw.times)))],axis=0)
        #     # print("i",i.shape)
        #     i_epo = mne.io.RawArray(i,raw_info)
        #     i_epo.drop_channels("Stim")
        #     if (j<attaxisrow):
        #         i_epo.compute_psd().plot(axes=ax[j,0],show=False,average=False,picks='data', exclude='bads')
        #         ax[j,0].set_title("Attention PSD of Epoch:{}".format(j+1))
        #     elif (j>=attaxisrow):
        #         i_epo.compute_psd().plot(axes=ax[j-attaxisrow,1],show=False,average=False,picks='data', exclude='bads')
        #         ax[j-attaxisrow,1].set_title("Attention PSD of Epoch:{}".format(j+1))
        #     fig.set_tight_layout(True)

        # rlxaxisrow = math.ceil(rlx_shape[0]/2)
        # fig2, ax2 = plt.subplots(rlxaxisrow,2)
        # fig2.set_size_inches(8.5, 12.5, forward=True)
        # for jj,ii in enumerate (epochs["relax"]):
        #     p_stim = np.zeros((1,len(ii[1])))
        #     # print(p_stim.shape)
        #     ii=np.append(ii,p_stim,axis=0)
        #     # np.append(i,[np.zeros((1,len(filtraw.times)))],axis=0)
        #     # print("i",i.shape)
        #     i_epo = mne.io.RawArray(ii,raw_info)
        #     i_epo.drop_channels("Stim")
        #     if (jj<rlxaxisrow):
        #         i_epo.compute_psd().plot(axes=ax2[jj,0],show=False,average=False,picks='data', exclude='bads')
        #         ax2[jj,0].set_title("Relax PSD of Epoch :{} ".format(jj+1))
        #     elif (jj>=rlxaxisrow):
        #         i_epo.compute_psd().plot(axes=ax2[jj-rlxaxisrow,1],show=False,average=False,picks='data', exclude='bads')
        #         ax2[jj-rlxaxisrow,1].set_title("Relax PSD of Epoch :{}".format(jj+1))
        #     fig2.set_tight_layout(True)

        # plt.show()
    
    def plot_pca_tsne(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        import pandas as pd
        import seaborn as sns

        global x_train, y_train,epochs
        n_cluster =2
        scaler = StandardScaler()
        pca = PCA(n_components= n_cluster)
        kmeans = KMeans(n_clusters= n_cluster)
        tsne = TSNE(n_components=2,verbose = 1,random_state=90)
        tsne_df = pd.DataFrame()

        printTrainshape(x_train,y_train,epochs)
        
        # pca.fit(x_scaled)
        x_train = x_train[:,0,:]

        try:
            scaler.fit(x_train)
            x_scaled = scaler.transform(x_train)

            pca_x = pca.fit_transform(x_scaled)
            tsne_x = tsne.fit_transform(x_scaled)
        except:
            pca_x = pca.fit_transform(x_train)
            tsne_x = tsne.fit_transform(x_train)

        tsne_df["y"]=y_train
        tsne_df["c1"]=tsne_x[:,0]
        tsne_df["c2"]=tsne_x[:,1]

        pca_np = pd.DataFrame(pca_x, columns=["PC1","PC2"]).to_numpy()

        x_clustered = kmeans.fit_predict(pca_np)
        LABEL_COLOR_MAP = {0:"r",1:"g"}
        label_color = [LABEL_COLOR_MAP[l]for l in x_clustered]

        tsne_np = tsne_df.to_numpy()

        # plt.figure(figsize=(7,7))
        # plt.scatter(,pca_np[:,1], c=label_color,alpha=0.5)
        # plt.legend()
        # plt.show()
        fig,(ax1,ax2) = plt.subplots(2)

        sns.scatterplot(ax = ax1 ,x=pca_np[:,0],y=pca_np[:,1],c=label_color,alpha=0.5).set(title="PCA")

        sns.scatterplot(ax = ax2, x=tsne_np[:,1],y=tsne_np[:,2],c=tsne_df.y.tolist(),alpha=0.5,
                        data=tsne_np).set(title="TSNE")
        plt.show()

        
        
        # ax.scatter(PC1_a, 
        #             PC2_a, 
        #             c="blue",
        #         label="Malignant")
        
        # ax.scatter(PC1_b, 
        #             PC2_b, 
        #             c="orange",
        #         label="Benign")
        
        # ax.legend(title="Label")
        
        # plt.title("Figure 1",
        #         fontsize=16)
        # plt.xlabel('First Principal Component',
        #         fontsize=16)
        # plt.ylabel('Second Principal Component',
        #         fontsize=16)


    def filter_bank(self):
        global fs,att_data,rel_data,analyzeChannel,channel_count,x_train,y_trainAtt,y_trainRlx,y_train,epochs
        
        bandrawAtt = att_data.copy()[:,analyzeChannel,:].flatten()
        bandrawRel = rel_data.copy()[:,analyzeChannel,:].flatten()
        
        deltaAtt = butter_bandpass_filter(bandrawAtt, fs, 6, .1, 4)
        thetaAtt = butter_bandpass_filter(bandrawAtt, fs, 6, 4 , 8)
        alphaAtt = butter_bandpass_filter(bandrawAtt, fs, 6, 8, 13)
        betaAtt = butter_bandpass_filter(bandrawAtt, fs, 6, 13, 32)
        gammaAtt = butter_bandpass_filter(bandrawAtt, fs, 6, 32, 50)

        deltaRel = butter_bandpass_filter(bandrawRel, fs, 6, .1, 4)
        thetaRel = butter_bandpass_filter(bandrawRel, fs, 6, 4 , 8)
        alphaRel = butter_bandpass_filter(bandrawRel, fs, 6, 8, 13)
        betaRel = butter_bandpass_filter(bandrawRel, fs, 6, 13, 32)
        gammaRel = butter_bandpass_filter(bandrawRel, fs, 6, 32, 50)

        bandAtt = [deltaAtt, thetaAtt, alphaAtt, betaAtt, gammaAtt]
        bandRel = [deltaRel, thetaRel, alphaRel, betaRel, gammaRel]

        n_att = len(bandrawAtt)
        n_rel = len(bandrawRel)

        timeAtt = np.arange(n_att)
        timeRel = np.arange(n_rel)

        # print("band delta:",band[0])
        bandname = ["delta","theta","alpha","beta","gamma"]
        fig, ax = plt.subplots(len(bandname))
        fig.set_size_inches(8,12)
        for i, b in enumerate (bandAtt):
            ax[i].plot(timeAtt,b,label=bandname[i])
            ax[i].set_xlabel("Time (s)")
            ax[i].set_title(bandname[i]+" band")
            ax[i].set_xlim(0,timeAtt.max())
        fig.suptitle("Attention Bandpower "+str(analyzeChannel+1))
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(len(bandname))
        fig.set_size_inches(8,12)
        for i, b in enumerate (bandRel):
            ax[i].plot(timeRel,b,label=bandname[i])
            ax[i].set_xlabel("Time (s)")
            ax[i].set_title(bandname[i]+" band")
            ax[i].set_xlim(0,timeRel.max())
        fig.suptitle("Relax Bandpower "+str(analyzeChannel+1))
        fig.tight_layout()
        plt.show() 
        # alphaAtt.reshape(x_train[0]/2,x_train[1])

        print("Attention alpha",alphaAtt.shape)
        print("Attention beta",betaAtt.shape)
        print("Relax alpha",alphaRel.shape)
        print("Relax beta",betaRel.shape)
        data_diff = alphaRel.shape[0] - alphaAtt.shape[0]
        print(att_data.shape)
        alphaRel = alphaRel[:-data_diff]
        betaRel = betaRel[:-data_diff]
        alphaAtt=alphaAtt.reshape(att_data.shape[0],att_data.shape[2])
        betaAtt=betaAtt.reshape(att_data.shape[0],att_data.shape[2])
        alphaRel=alphaRel.reshape(att_data.shape[0],att_data.shape[2])
        betaRel=betaRel.reshape(att_data.shape[0],att_data.shape[2])
        print("Attention alpha",alphaAtt.shape)
        print("Attention beta",betaAtt.shape)
        print("Relax alpha",alphaRel.shape)
        print("Relax beta",betaRel.shape)
        
        attentionband = np.dstack((alphaAtt,betaAtt))
        relaxband = np.dstack((alphaRel,betaRel))

        y_trainAtt=y_trainAtt[:attentionband.shape[0]]
        y_trainRlx=y_trainRlx[:attentionband.shape[0]]

        print("attBand: ",attentionband.shape)
        print("rlxband:" , relaxband.shape)
        x_train = np.vstack((attentionband,relaxband))
        x_train=x_train.reshape((x_train.shape[0],x_train.shape[2],x_train.shape[1]))
        epochs = x_train
        y_train = np.append(y_trainAtt,y_trainRlx)
        printTrainshape(x_train,y_train,epochs)

    def discrete_wavelet(self):
        # get separately
        global att_data,rel_data,analyzeChannel,channel_count,y_train,x_train
        dwtattdata = att_data[:,analyzeChannel,:].flatten()
        dwtreldata = rel_data[:,analyzeChannel,:].flatten()
        # print("DWavelet Attention data",dwtattdata)
        coeffs_att = pywt.wavedec(dwtattdata, 'db4', level=6)
        cA2a, cD1a, cD2a,alphaAtt,betaAtt,cD5a,cD6a = coeffs_att
        coeffs_rel = pywt.wavedec(dwtreldata, 'db4', level=6)
        cA2r, cD1r, cD2r,alphaRlx,betaRlx,cD5r,cD6r = coeffs_rel
        fig = plt.figure("Discrete Wavelet: "+str(analyzeChannel+1))

        plt.subplot(7, 2, 1)
        plt.plot(dwtattdata)
        plt.ylabel('Noisy Signal')

        plt.subplot(7, 2, 3)
        plt.plot(cD6a)
        plt.ylabel('noisy')

        plt.subplot(7,2,5)
        plt.plot(cD5a)
        plt.ylabel("gamma")

        plt.subplot(7,2,7)
        plt.plot(betaAtt)
        plt.ylabel("beta")

        plt.subplot(7,2,9)
        plt.plot(alphaAtt)
        plt.ylabel("alpha")

        plt.subplot(7,2,11)
        plt.plot(cD2a)
        plt.ylabel("theta")

        plt.subplot(7,2,13)
        plt.plot(cD1a)
        plt.ylabel("delta")

        plt.subplot(7, 2, 2)
        plt.plot(dwtreldata)
        plt.ylabel('Noisy Signal')

        plt.subplot(7,2,4)
        plt.plot(cD6r)
        plt.ylabel('noisy')

        plt.subplot(7,2,6)
        plt.plot(cD5r)
        plt.ylabel("gamma")

        plt.subplot(7,2,8)
        plt.plot(betaRlx)
        plt.ylabel("beta")

        plt.subplot(7,2,10)
        plt.plot(alphaRlx)
        plt.ylabel("alpha")

        plt.subplot(7,2,12)
        plt.plot(cD2r)
        plt.ylabel("theta")

        plt.subplot(7,2,14)
        plt.plot(cD1r)
        plt.ylabel("delta")
        plt.suptitle("Channel :"+str(analyzeChannel+1))
        plt.show()

        print("attention Beta",betaAtt.shape)
        print("attention Alpha",alphaAtt.shape)
        print("Relax Beta",betaRlx.shape)
        print("Relax Alpha",alphaRlx.shape)
        print("DWAVELET FEATURE_____")
        betaAtt.flatten()   
        betaAtt=betaAtt.reshape((att_data.shape[0],math.ceil(betaAtt.shape[0]/att_data.shape[0])))
        alphaAtt=alphaAtt.reshape((att_data.shape[0],math.ceil(alphaAtt.shape[0]/att_data.shape[0])))
        betaRlx=betaRlx.reshape((att_data.shape[0],math.ceil(betaRlx.shape[0]/att_data.shape[0])))
        alphaRlx=alphaRlx.reshape((att_data.shape[0],math.ceil(alphaRlx.shape[0]/att_data.shape[0])))
        print("attention Beta",betaAtt.shape)
        print("attention Alpha",alphaAtt.shape)
        print("Relax Beta",betaRlx.shape)
        print("Relax Alpha",alphaRlx.shape)
        
        attentionband = np.vstack((alphaAtt,alphaRlx))
        relaxband = np.vstack((betaAtt,betaRlx))
        print(attentionband.shape)
        print(relaxband.shape)
        x_train = np.hstack((attentionband),(relaxband))
        printTrainshape(x_train,y_train,epochs)


    def hilbert_plot(self):
        import emd
        global epochs,fs,att_data,rel_data,analyzeChannel,channel_count,y_train,x_train,y_trainAtt,y_trainRlx
        attx=att_data[:,analyzeChannel,:]   # this indexing
        rlxx=rel_data[:,analyzeChannel,:]

        # print(attx)
        attx=attx.flatten()
        rlxx=rlxx.flatten()

        atxn=np.linalg.norm(attx)
        rlxn=np.linalg.norm(rlxx)

        atnorm=attx/atxn
        rlnorm=rlxx/rlxn

        imfatt = emd.sift.mask_sift(atnorm, max_imfs=5)
        imfrlx = emd.sift.mask_sift(rlnorm, max_imfs=5)
        # print(imfatt.shape)
        emd.plotting.plot_imfs(imfatt)
        plt.title("Attention IMFS "+str(analyzeChannel+1))

        emd.plotting.plot_imfs(imfrlx)
        plt.title("Relax IMFS "+str(analyzeChannel+1))
        plt.show()

        alphaAtt = imfatt[:,1]
        betaAtt = imfatt[:,2]
        alphaRel = imfrlx[:,1]
        betaRel = imfrlx[:,2]

        print("Attention IMF-1", alphaAtt.shape)
        print("Attention IMF-2", betaAtt.shape)
        print("Relax IMF-1", alphaRel.shape)
        print("Relax IMF-2", betaRel.shape)

        bandDiff = alphaRel.shape[0] - alphaAtt.shape[0]
        # alphaRel = alphaRel[:]

        alphaRel = alphaRel[:-bandDiff]
        betaRel =  betaRel[:-bandDiff]

        alphaAtt = alphaAtt.reshape((att_data.shape[0],att_data.shape[2]))
        betaAtt = betaAtt.reshape((att_data.shape[0],att_data.shape[2]))
        alphaRel = alphaRel.reshape((att_data.shape[0],att_data.shape[2]))
        betaRel = betaRel.reshape((att_data.shape[0],att_data.shape[2]))

        attentionband = np.dstack((alphaAtt,betaAtt))
        relaxband = np.dstack((alphaRel,betaRel))

        print("Attention IMF-1", alphaAtt.shape)
        print("Attention IMF-2", betaAtt.shape)
        print("Relax IMF-1", alphaRel.shape)
        print("Relax IMF-2", betaRel.shape)

        print("attBand: ",attentionband.shape)
        print("rlxband:" , relaxband.shape)

        x_train = np.vstack((attentionband,relaxband))
        epochs = x_train
        x_train=x_train.reshape((x_train.shape[0],x_train.shape[2],x_train.shape[1]))

        y_trainAtt=y_trainAtt[:attentionband.shape[0]]
        y_trainRlx=y_trainRlx[:attentionband.shape[0]]

        y_train = np.append(y_trainAtt,y_trainRlx)
        printTrainshape(x_train,y_train,epochs)

    def clf_logreg(self):
        global x_train,y_train,epochs

        printTrainshape(x_train,y_train,epochs)
        clf = make_pipeline(Scaler(epochs.info),
                            Vectorizer(),
                            LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=3000))

        scores = cross_val_multiscore(clf,x_train,y_train, cv=5, n_jobs=1)
        score = np.mean(scores,axis=0) # cross validation score

        print("Logistic Regression Classify Score: %0.2f%%" % (100 * score,))
        display_confusionMatrix(clf)

    def clf_SVM(self):
        from sklearn.svm import SVC   
        global x_train,y_train,epochs

        printTrainshape(x_train,y_train,epochs)
        clf= make_pipeline(Scaler(epochs.info),
                           Vectorizer(),
                           SVC())
        
        scores = cross_val_multiscore(clf,x_train,y_train, cv=5, n_jobs=1)
        score = np.mean(scores,axis=0) # cross validation score

        print("Single Value Decomposition Classify Score: %0.2f%%" % (100*score,))
        display_confusionMatrix(clf)

    def clf_DNN(self):
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from keras.optimizers import SGD
        from keras.losses import BinaryCrossentropy
        
        global x_train, y_train, epochs

        printTrainshape(x_train,y_train,epochs)
        ydnn_train=y_train.copy()
        ydnn_train[ydnn_train==1000]=.99
        ydnn_train[ydnn_train==2000]=0
    
        xdnn_train, xdnn_test, ydnn_train, ydnn_test = train_test_split( x_train, ydnn_train, test_size = .20, random_state = 90)
        # (x_train,y_train),(x_test,y_test) =
        xdnn_train = tf.keras.utils.normalize(xdnn_train, axis=1)
        xdnn_test = tf.keras.utils.normalize(xdnn_test, axis=1)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu,input_dim=3))
        model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))                         
        # model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))

        opt = SGD(learning_rate=0.01)

        model.compile(optimizer = opt,
                      loss = BinaryCrossentropy(),  #sparse_categorical_crossentropy
                      metrics = ['accuracy'])
        model.fit(xdnn_train,ydnn_train,epochs=5)


        val_loss,val_acc = model.evaluate(xdnn_test,ydnn_test)
        print("Loss: ",val_loss,"Accuracy: ", val_acc)
        display_confusionMatrix(model)

    def save_exp(self):
        global raw,saveName
        raw.save(str(saveName)+"_raw.fif",overwrite=True)
        print("Save File as", str(saveName)+"_raw.fif")

    def plot_fft(self,ch,fftAt,allText):
        global fs,signalDataCount,fftSignalArray,channel_count,unicorn,unitz,this_y,rawSignalArray,ftch

        ftch=np.asarray(fftSignalArray[ch], dtype=np.float64)
        
        try:
            # fftSignalArray = data[np.where(data!=0)]
            normft=np.linalg.norm(ftch)
            if (normft):
                ftch = np.asarray([(x/normft) for x in ftch])
            ftch = ftch[np.where(ftch<ftch.max())]
            ftch = ftch[np.where(ftch>ftch.min())]

        except :
            print("fft buffering")
        # print(fftSignalArray[:,ch])
        
        # print("after",fftSignalArray)

        fftsize=128
        try:
            fftSignalArraysize = len(ftch[np.where(ftch!=0)])
        except:
            fftSignalArraysize=1000
        
        fftAt.setText("FFT Calc Size: "+str(fftsize))
        allText.setText("All data Incoming: "+str(signalDataCount))

        # print(len(fftSignalArray))
        # print(np.count_nonzero(data.astype(int)))
        if (fftSignalArraysize>(fftsize*5)):

            fft_result=np.fft.fft(ftch)
            ab_fft=np.abs(fft_result)

            freq=fft.fftfreq(ftch.shape[0],d=1/fs)
            ab_freq=np.abs(freq)
            # ab_fft = fft.rfft(fftSignalArray)/(fftSignalArraysize/2)
            # ab_fft = np.abs(ab_fft)
            # ab_freq = fft.rfftfreq(fftSignalArraysize,d=1/fs)

            if (ch==0):     # channel 1 filter
                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
                # filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)
                # filtered_thisY=butter_lowpass_filter(this_y,30,fs,4)
                filtered_data=butter_bandstop_filter(ftch,fs,6,47,53)
                # filtered_data = fftSignalArray
        
            elif (ch==1):     # channel 1 filter
                filtered_data=butter_bandstop_filter(ftch,fs,6,47,53)
                # filtered_data = fftSignalArray

                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
            elif (ch==2):     # channel 1 filter
                # filtered_data=butter_bandstop_filter(fftSignalArray,fs,4,28,34)
                filtered_data = ftch
            else:
                filtered_data= ftch

            filtered_fft_result = np.fft.fft(filtered_data)
            abs_filtered_fft_result=np.abs(filtered_fft_result)
            # try:
            #     self.fftcurves[ch].setData(ab_freq,ab_fft)
            #     self.filtfftcurves[ch].setData(ab_freq, abs_filtered_fft_result)
            # except:
            #     print("can't plot fft") 

    def reset_buffer(self,ch):
        print("Resetting buffer Channel:", ch+1)

        # self.fftcurves[ch].setData(self.empty,self.empty)
        # self.filtfftcurves[ch].setData(self.empty,self.empty)
        # self.buffer=np.empty(self.buffer.shape)
        # print("resest buffer", self.buffer.shape[0])
