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

import random
import socket
import time

import sys
import os as os

import rtfs_genfunc as gf

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from mne.decoding import (SlidingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer)

plot_duration = 5 # how many seconds of data to show
update_interval = 20  # ms between screen updates
pull_interval = 60 # ms between each pull operation
fft_interval = 500 # ms between each FFT calculation and triggering
global rsarray,unicorn,unitz
mmarray = []
codearray=[]
db_counter=0
record=False
att_count=0
max_epoch=5
data_count=0
unicorn = False
loadraw=False
unitz = 1000000
import functools


class Inlet:
    def __init__(self,info:pylsl.StreamInfo):
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        
        self.name = info.name()
        self.channel_count = info.channel_count()
        global fs
        global nyq
        fs = info.nominal_srate()
        nyq=1/2*fs   
        
    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        pass
    def plot_fft(self,ch):
        pass


class DataInlet(Inlet):  
    def __init__(self,info:pylsl.StreamInfo,plt:pg.PlotItem,fftplt:pg.PlotItem,filtplt:pg.PlotItem,filtfftplt:pg.PlotItem,bufferText):
        global rsarray,ftarray
        super().__init__(info)
        dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
        self.bufsize,self.buffer,self.empty=gf.set_buffer(info,dtypes,bufferText)
        
        
        # signal Curves
        self.curves = [pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.fftcurves= [pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtcurves= [pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtfftcurves=[pg.PlotCurveItem(x=self.empty, y=self.empty , autoDownsample=True) for _ in range(self.channel_count)]

        # triggerValA=1.3
        # triggerValB=0.5
        # triggerLine=pg.InfiniteLine(label="SimpleTriggerLine",pos=triggerValA,movable=True,angle=0)
        
        for curve in self.curves:
            plt.addItem(curve)
            # plt.addItem(triggerLine)
        for fftcurve in self.fftcurves:
            fftplt.addItem(fftcurve)
        for filtcurve in self.filtcurves:
            filtplt.addItem(filtcurve)
            # filtplt.addItem(triggerLine)
        for filtfftcurve in self.filtfftcurves:
            filtfftplt.addItem(filtfftcurve)

    def pull_and_plot(self, plot_time, ch,tch,DB_text,DB_markertext,fixationText,dataAmountText):
        global data_count,unitz,ftarray
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        this_x=None
        global db_counter,rsarray,unicorn,this_y
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
                filtered_thisY=gf.butter_bandstop_filter(this_y,fs,4,49,51)
                # print("skip filter")
                # filtered_thisY = this_y
                
            elif (ch==1):     # channel 1 filter
                filtered_thisY=gf.butter_bandstop_filter(this_y,fs,4,47,53)
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
            # rsarray=np.hstack((rsarray[old_offset:],filtered_thisY[new_offset:]))
            for i in range(ts.size):
                ftarray[ch].append(y[i,ch])
                # print("Assign ftarray:",ftarray)
                data_count+=1
                if (record):
                    # for ii in range(ts.size):   
                    rsarray[ch].append(y[i,ch])
                    # print(rsarray)
                    # print("After : ",rsarray)
                    if (ch==0):
                        db_counter+=1
                        gf.changeCode(0,tch,fixationText)
                
            dataAmountText.setText("Data Input Amount: "+str(int(db_counter)))
            try :
                DB_text.setText("Raw Data Count: " + str(len(rsarray[ch])))
            except IndexError:
                DB_text.setText("Raw Data Count: " + str(len(rsarray[ch])))
            DB_markertext.setText("Marker Data Count: " + str(len(mmarray)))
            # print("Y",y.shape)
            # print("RS",rsarray.shape)
            # print(mmarray)
            
            # if (DB_markertext.text()==DB_text.text()):
            #     fixationText.setText("Matched!!")
            # print("To be raw data"+str(filtered_thisY[0:10]))
            # print("To be stim channel"+str(mmarray))
        
    def convertToRaw(self):
        global rsarray,mmarray,channel_count,fs,filtraw,ch1_pick,ch2_pick,unicorn,raw
        ch_names=[f"eeg{n:02}" for n in range(1,channel_count+1)]  
        
        ch_types=["eeg"]*(channel_count)

        if (unicorn):
            refch=["l","r"]
            ch_names.pop()
            ch_names.pop()
            ch_names.extend(refch)

        stim_ch_names= ["Stim"]     
        info = mne.create_info(ch_names, sfreq=fs,ch_types=ch_types)
        stim_info = mne.create_info(stim_ch_names, sfreq=fs,ch_types="stim")
        
        print("Try converting to Raw")
        # rawarray=np.array(rsarray)
        
        markarray=np.array(mmarray)
        markarray=markarray.reshape(1,len(mmarray))
        print(info)

        rsarrayN = np.linalg.norm(rsarray)
        rsarray = rsarray/rsarrayN

        raw = mne.io.RawArray(rsarray, info)
        stim_raw = mne.io.RawArray(markarray,stim_info)
        
        ch1_pick=mne.pick_channels(ch_names=ch_names,include=["eeg01"])
        ch2_pick=mne.pick_channels(ch_names=ch_names,include=["eeg02"])
        # filter
        # filtraw=raw.copy().notch_filter(11,picks=ch1_pick,n_jobs=1)
        if (unicorn):
            raw.set_eeg_reference(ref_channels=refch)
        # print(raw.channel)
            powerline = (50,100)

            ori_raw=raw.copy()
            filtraw=ori_raw.notch_filter(powerline,n_jobs=1)
            filtraw=filtraw.filter(0.1,50)
        else:
            filtraw=raw.copy()
            filtraw=filtraw.notch_filter(20,n_jobs=1)
        # ica = mne.preprocessing.ICA(n_components=2,random_state=9030,max_iter=100)
        
        # ica.fit(filtraw)
        # ica.exclude = []
        # ica.plot_properties(filtraw) #,picks=ica.exclude


        # filtraw = raw.copy().filter(l_freq=8,h_freq=12,picks=ch1_pick)
        # # filtraw=raw.copy().filter(0,30)
        # filtraw = filtraw.filter(18,22,picks=ch2_pick)

        # filtraw = raw.copy().filter(0,38)
        # raw.filter(0,30)
        filtraw.add_channels([stim_raw], force_update_info=True)
        raw.add_channels([stim_raw], force_update_info=True)

        print(raw.info)
        if unicorn:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title="Raw Signal-non scale")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title = "Filtered Signal-non scale")
        else:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,scalings="auto",title="Raw Signal")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,scalings="auto",title = "Filtered Signal")

    def loadraw(self):
        global raw,unicorn,fn,filtraw,channel_count,fs,loadraw,raw_info
        loadraw=True
        normalize = False
        raw = mne.io.read_raw_fif(fn,preload=True)
        raw_info =raw.info
        # channel_count = info["nchan"]
        # fs = info["sfreq"]
        print(raw_info)
        filtraw = raw.copy()
        powerline = (50,100)

        filtraw=filtraw.notch_filter(powerline,n_jobs=1)
        filtraw=filtraw.filter(0.1,50)
        if normalize:
            raw = Scaler(scalings='mean').fit_transform(raw.get_data())
            filtraw = Scaler(scalings='mean').fit_transform(filtraw.get_data())
            
            raw = raw.reshape(raw.shape[0],raw.shape[1])
            filtraw = filtraw.reshape(filtraw.shape[0],filtraw.shape[1])

        
        # print("the raw",raw)
        # print("the filtraw",filtraw)
        
            raw = mne.io.RawArray(raw, raw_info)
            filtraw = mne.io.RawArray(filtraw,raw_info)

        if unicorn:
            raw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title="Raw Signal-non scale")
            filtraw.plot(n_channels=channel_count,show_scrollbars=True, show=True,title = "Filtered Signal-non scale")
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

    def find_events(self):
        global filtraw,fs,events_dict,events,unicorn
        events = mne.find_events(filtraw,stim_channel = "Stim")
        print(events[:10])

        # Trial Descriptor
        events_dict={
            "attention":1000,
            "relax":2000
        }
        # annot_events = mne.annotations_from_events(events=events,event_desc=events_dict,sfreq=fs)
        # raw.set_annotations(annot_events)
        fig = mne.viz.plot_events(
            events=events, sfreq=filtraw.info["sfreq"], first_samp=filtraw.first_samp, event_id=events_dict
        )
        fig.subplots_adjust(right=0.7)  # make room for legend
        if (unicorn):
            filtraw.plot(events,event_color={1000:"r",2000:"g",100:"b",96:"y"},title="Filtered Events-non scale")

        else:
            filtraw.plot(events,event_color={1000:"r",2000:"g",100:"b",96:"y"},scalings="auto",title="Filtered Events")

    def epoch_plot(self):
        global filtraw,events,events_dict,unicorn,att_epo,rel_epo,x_train,y_train,epochs,fs,loadraw,raw_info,att_data,rel_data
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
        att_data = epochs["attention"].get_data()[:,0,:]
        rel_data = epochs["relax"].get_data()[:,0,:]

        att_epo = epochs["attention"]
        rel_epo = epochs["relax"]

        # print("Att",att_epo.shape)
        # print("Rlx",rel_epo.shape)
        
        x_train = epochs.get_data()
        # x_train=x_train.reshape(x_train.shape[0]*x_train.shape[2],x_train.shape[1])
        # x_train = x_train.reshape(x_train.shape[0]*x_train.shape[2],x_train.shape[1])
        # print(x_train.shape)
        y_train = epochs.events[:,2]
        # print(x_train)
        # print("shapey",y_train.shape)
        # print("X:",x_train.shape)
        # print("Y:",y_train.shape)
        epochs["attention"].plot_image(
            picks="eeg" #,combine="mean"
            ,title="Attention Epoch"
        )
        epochs["relax"].plot_image(
            picks="eeg" #,combine="mean"
            ,title="Relax Epoch"
        )

        # att_db=mne.io.RawArray(att_db, info)
        # rlx_db=mne.io.RawArray(rlx_db, info)
        # epochs["attention"].compute_psd().plot(average=False,picks='data', exclude='bads')
        # epochs["relax"].compute_psd().plot(average=False,picks='data', exclude='bads')
        att_shape = epochs["attention"].get_data().shape
        rlx_shape = epochs["relax"].get_data().shape
        # print(att_shape)

        attaxisrow = math.ceil(att_shape[0]/2)
        fig, ax = plt.subplots(attaxisrow,2)
        fig.set_size_inches(8.5, 12.5, forward=True)
        for j,i in enumerate (epochs["attention"]):
            p_stim = np.zeros((1,len(i[1])))
            # print(p_stim.shape)
            i=np.append(i,p_stim,axis=0)
            # i=np.r_[i,[p_stim]]
            # print("i new",i.shape)
            # np.append(i,[np.zeros((1,len(filtraw.times)))],axis=0)
            # print("i",i.shape)
            i_epo = mne.io.RawArray(i,raw_info)
            i_epo.drop_channels("Stim")
            if (j<attaxisrow):
                i_epo.compute_psd().plot(axes=ax[j,0],show=False,average=False,picks='data', exclude='bads')
                ax[j,0].set_title("Attention PSD of Epoch:{}".format(j+1))
            elif (j>=attaxisrow):
                i_epo.compute_psd().plot(axes=ax[j-attaxisrow,1],show=False,average=False,picks='data', exclude='bads')
                ax[j-attaxisrow,1].set_title("Attention PSD of Epoch:{}".format(j+1))
            
            fig.set_tight_layout(True)

        rlxaxisrow = math.ceil(rlx_shape[0]/2)
        fig2, ax2 = plt.subplots(rlxaxisrow,2)
        fig2.set_size_inches(8.5, 12.5, forward=True)
        for jj,ii in enumerate (epochs["relax"]):
            # raw_info["ch_names"].pop()
            # raw_info["nchan"]=channel_count-1
            # print(raw_info)
            # print("counter ",j+1," round")
            # print("i",i.shape)
            p_stim = np.zeros((1,len(ii[1])))
            # print(p_stim.shape)
            ii=np.append(ii,p_stim,axis=0)
            # i=np.r_[i,[p_stim]]
            # print("i new",i.shape)
            # np.append(i,[np.zeros((1,len(filtraw.times)))],axis=0)
            # print("i",i.shape)
            i_epo = mne.io.RawArray(ii,raw_info)
            i_epo.drop_channels("Stim")
            if (jj<rlxaxisrow):
                i_epo.compute_psd().plot(axes=ax2[jj,0],show=False,average=False,picks='data', exclude='bads')
                ax2[jj,0].set_title("Relax PSD of Epoch :{} ".format(jj+1))
            elif (jj>=rlxaxisrow):
                i_epo.compute_psd().plot(axes=ax2[jj-rlxaxisrow,1],show=False,average=False,picks='data', exclude='bads')
                ax2[jj-rlxaxisrow,1].set_title("Relax PSD of Epoch :{}".format(jj+1))
            fig2.set_tight_layout(True)

        plt.show()

    def hilbert_plot(self):
        import emd
        # import sklearn
        from sklearn.preprocessing import normalize
        global epochs,fs,att_data,rel_data
        attx=att_data   # this indexing
        rlxx=rel_data

        # print(attx)
        attx=attx.flatten()
        rlxx=rlxx.flatten()

        atxn=np.linalg.norm(attx)
        rlxn=np.linalg.norm(rlxx)

        atnorm=attx/atxn
        rlnorm=rlxx/rlxn

        # print('attention Norm',atnorm.shape)
        # print('rlx Norm',rlnorm.shape)
        # filtrawn=normalize(filtraw)
        imfatt = emd.sift.mask_sift(atnorm, max_imfs=5)
        imfrlx = emd.sift.mask_sift(rlnorm, max_imfs=5)
        # print(imfatt.shape)
        emd.plotting.plot_imfs(imfatt)
        plt.title("Attention IMFS")

        emd.plotting.plot_imfs(imfrlx)
        plt.title("Relax IMFS")

        IPa, IFa, IAa = emd.spectra.frequency_transform(imfatt, fs, 'nht')
        IPr, IFr, IAr = emd.spectra.frequency_transform(imfrlx, fs, 'nht')
        # print("att",imfatt)
        # print("rlx",imfrlx)
        plt.figure(figsize=(12, 8))

        fig,axs = plt.subplots(2,2)
        # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
        axs[0,0].hist(IFa[:, 0], np.linspace(0, 20), weights=IAa[:, 0])
        axs[0,0].grid(True)
        axs[0,0].set_title('Attention IF Histogram\nweighted by IA 1')
        axs[0,0].set_xticks(np.arange(0, 20, 5))
        axs[0,0].set(xlabel='Frequency (Hz)')


        # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
        axs[0,1].hist(IFr[:, 0], np.linspace(0, 20), weights=IAr[:, 0])
        axs[0,1].grid(True)
        axs[0,1].set_title('Relax IF Histogram\nweighted by IA 1')
        axs[0,1].set_xticks(np.arange(0, 20, 5))
        axs[0,1].set(xlabel='Frequency (Hz)')

        # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
        axs[1,0].hist(IFa[:, 1], np.linspace(0, 20), weights=IAa[:, 1])
        axs[1,0].grid(True)
        axs[1,0].set_title('Attention IF Histogram\nweighted by IA 2')
        axs[1,0].set_xticks(np.arange(0, 20, 5))
        axs[1,0].set(xlabel='Frequency (Hz)')

        # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
        axs[1,1].hist(IFr[:, 1], np.linspace(0, 20), weights=IAr[:, 1])
        axs[1,1].grid(True)
        axs[1,1].set_title('Relax IF Histogram\nweighted by IA 2')
        axs[1,1].set_xticks(np.arange(0, 20, 5))
        axs[1,1].set(xlabel='Frequency (Hz)')

        fig.tight_layout()
        plt.show()

    def discrete_wavelet(self):
        # get separately
        global att_data,rel_data
        
        dwtattdata = att_data
        dwtreldata = rel_data

        dwtattdata=dwtattdata.flatten()
        dwtreldata=dwtreldata.flatten()
        
        coeffs_att = pywt.wavedec(dwtattdata, 'db4', level=6)
        cA2a, cD1a, cD2a,cD3a,cD4a,cD5a,cD6a = coeffs_att
        coeffs_rel = pywt.wavedec(dwtreldata, 'db4', level=6)
        cA2r, cD1r, cD2r,cD3r,cD4r,cD5r,cD6r = coeffs_rel
        
        plt.subplot(7, 2, 1)
        plt.plot(dwtattdata)
        plt.ylabel('Noisy Signal')
        plt.subplot(7, 2, 2)
        plt.plot(cD6a)
        plt.ylabel('noisy')
        plt.subplot(7,2,3)
        plt.plot(cD5a)
        plt.ylabel("gamma")
        plt.subplot(7,2,4)
        plt.plot(cD4a)
        plt.ylabel("beta")
        plt.subplot(7,2,5)
        plt.plot(cD3a)
        plt.ylabel("alpha")

        plt.subplot(7,2,6)
        plt.plot(cD2a)
        plt.ylabel("theta")

        plt.subplot(7,2,7)
        plt.plot(cD1a)
        plt.ylabel("delta")

        plt.subplot(7, 2, 8)
        plt.plot(dwtreldata)
        plt.ylabel('Noisy Signal')

        plt.subplot(7,2,9)
        plt.plot(cD6r)
        plt.ylabel('noisy')

        plt.subplot(7,2,10)
        plt.plot(cD5r)
        plt.ylabel("gamma")

        plt.subplot(7,2,11)
        plt.plot(cD4r)
        plt.ylabel("beta")

        plt.subplot(7,2,12)
        plt.plot(cD3r)
        plt.ylabel("alpha")

        plt.subplot(7,2,13)
        plt.plot(cD2r)
        plt.ylabel("theta")

        plt.subplot(7,2,14)
        plt.plot(cD1r)
        plt.ylabel("delta")

        plt.draw()
        plt.show()

    def filter_bank(self):
        global fs,filtraw,att_data,rel_data
        
        bandrawAtt = att_data.copy()
        bandrawRel = rel_data.copy()


        print("att filterbank",bandrawAtt.shape)
        
        deltaAtt = butter_bandpass_filter(bandrawAtt, fs, 4, .1, 4)
        print(deltaAtt)

        # change to numpy filtering NOT MNE
        deltaA = bandrawAtt.filter(.1,4).get_data()[0]
        thetaA = bandrawAtt.filter(4,8).get_data()[0] 

        alpA = bandrawAtt.filter(8,13)
        alphaA = alpA.get_data()[0]
        betA = bandrawAtt.filter(13,32)
        betaA = betA.get_data()[0]

        gammaA = bandrawAtt.filter(32,50).get_data()[0]
        bandA = [deltaA, thetaA, alphaA, betaA, gammaA]

        deltaB = bandrawRel.filter(.1,4).get_data()[0]
        thetaB = bandrawRel.filter(4,8).get_data()[0]

        alpB = bandrawRel.filter(8,13)
        alphaB = alpB.get_data()[0]
        betB = bandrawRel.filter(13,32)
        betaB = betB.get_data()[0]

        gammaB = bandrawRel.filter(32,50).get_data()[0]
        bandB = [deltaB, thetaB, alphaB, betaB, gammaB]

        timeAtt = bandrawAtt.times
        timeRel = bandrawRel.times

        
        # print("band delta:",band[0])
        bandname = ["delta","theta","alpha","beta","gamma"]
        fig, ax = plt.subplots(len(bandname))
        fig.set_size_inches(8,12)
        for i, b in enumerate (bandA):
            ax[i].plot(timeAtt,b,label=bandname[i])
            ax[i].set_xlabel("Times in Seconds")
            ax[i].set_title(bandname[i]+" band")
            ax[i].set_xlim(0,5)
        fig.suptitle("Attention Bandpower")
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(len(bandname))
        fig.set_size_inches(8,12)
        for i, b in enumerate (bandB):
            try:
                ax[i].plot(timeB,b,label=bandname[i])
            except ValueError:
                if (len(timeB)>len(b)):
                    timeB=np.delete(timeB,len(timeB)-1,0)
                else:
                    b = np.delete(b,len(b)-1,0)
                ax[i].plot(timeRel,b,label=bandname[i])
            ax[i].set_xlabel("Times in Seconds")
            ax[i].set_title(bandname[i]+" band")
            ax[i].set_xlim(0,5)
        fig.suptitle("Relax Bandpower")
        fig.tight_layout()
        plt.show() 
        # alpB = 

        alpB.drop_channels(["l","r","Stim"])
        betB.drop_channels(["l","r","Stim"])
        alpA.drop_channels(["l","r","Stim"])
        betA.drop_channels(["l","r","Stim"])

        betA.rename_channels({"eeg01": "beta Attention"})
        betB.rename_channels({"eeg01": "beta Relaxation"})
        alpA.rename_channels({"eeg01": "alpha Attention"})
        alpB.rename_channels({"eeg01": "alpha Relaxation"})



        betA.add_channels([alpA,betA], force_update_info=True)
        betB.add_channels([alpB,betB], force_update_info=True)

        betA.plot()
        betB.plot()
        
        # print(feat)
        
    def clf_logreg(self):
        global x_train,y_train,epochs
        clf = make_pipeline(Scaler(epochs.info),
                            Vectorizer(),
                            LogisticRegression(solver="liblinear"))

        scores = cross_val_multiscore(clf,x_train,y_train, cv=5, n_jobs=1)
        score = np.mean(scores,axis=0) # cross validation score

        print("Train X Shape: ", x_train.shape)
        print("Logistic Regression Classify Score: %0.2f%%" % (100 * score,))

    def clf_SVM(self):
        from sklearn.svm import SVC   
        global x_train,y_train,epochs
        
        clf= make_pipeline(Scaler(epochs.info),
                           Vectorizer(),
                           SVC())
        
        scores = cross_val_multiscore(clf,x_train,y_train, cv=5, n_jobs=1)
        score = np.mean(scores,axis=0) # cross validation score

        print("Train X Shape: ", x_train.shape)
        print("Single Value Decomposition Classify Score: %0.2f%%" % (100*score,))

    def save_exp(self):
        global raw,saveName
        raw.save(str(saveName)+"_raw.fif",overwrite=True)
        print("Save File as", str(saveName)+"_raw.fif")

    def plot_fft(self,ch,fftAt,allText):
        global fs,data_count,ftarray,channel_count,unicorn,unitz,this_y,rsarray,ftch

        ftch=np.asarray(ftarray[ch], dtype=np.float64)
        
        try:
            # ftarray = data[np.where(data!=0)]
            normft=np.linalg.norm(ftch)
            if (normft):
                ftch = np.asarray([(x/normft) for x in ftch])
            ftch = ftch[np.where(ftch<ftch.max())]
            ftch = ftch[np.where(ftch>ftch.min())]

        except :
            print("fft buffering")
        # print(ftarray[:,ch])
        
        # print("after",ftarray)

        fftsize=128
        try:
            ftarraysize = len(ftch[np.where(ftch!=0)])
        except:
            ftarraysize=1000
        
        fftAt.setText("FFT Calc Size: "+str(fftsize))
        allText.setText("All data Incoming: "+str(data_count))

        # print(len(ftarray))
        # print(np.count_nonzero(data.astype(int)))
        if (ftarraysize>(fftsize*5)):

            fft_result=np.fft.fft(ftch)
            ab_fft=np.abs(fft_result)

            freq=fft.fftfreq(ftch.shape[0],d=1/fs)
            ab_freq=np.abs(freq)
            # ab_fft = fft.rfft(ftarray)/(ftarraysize/2)
            # ab_fft = np.abs(ab_fft)
            # ab_freq = fft.rfftfreq(ftarraysize,d=1/fs)

            if (ch==0):     # channel 1 filter
                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
                # filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)
                # filtered_thisY=butter_lowpass_filter(this_y,30,fs,4)
                filtered_data=gf.butter_bandstop_filter(ftch,fs,4,49,51)
                # filtered_data = ftarray
        
            elif (ch==1):     # channel 1 filter
                filtered_data=gf.butter_bandstop_filter(ftch,fs,4,47,53)
                # filtered_data = ftarray

                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
            elif (ch==2):     # channel 1 filter
                # filtered_data=butter_bandstop_filter(ftarray,fs,4,28,34)
                filtered_data = ftch
            else:
                filtered_data= ftch

            filtered_fft_result = np.fft.fft(filtered_data)
            abs_filtered_fft_result=np.abs(filtered_fft_result)
            try:
                self.fftcurves[ch].setData(ab_freq,ab_fft)
                self.filtfftcurves[ch].setData(ab_freq, abs_filtered_fft_result)
            except:
                print("can't plot fft") 

    def reset_buffer(self,ch):
        print("Resetting buffer Channel:", ch+1)

        self.fftcurves[ch].setData(self.empty,self.empty)
        self.filtfftcurves[ch].setData(self.empty,self.empty)
        # self.buffer=np.empty(self.buffer.shape)
        # print("resest buffer", self.buffer.shape[0])
        
def main():
    # first resolve a marker stream on the lab network
    print("looking for a marker stream...")
    global rsarray,ftarray,channel_count,fs
    streams = pylsl.resolve_stream('type', 'EEG')
    # mstreams=pylsl.resolve_stream("type","Markers")
    # print(mstreams)
    # minlet=pylsl.StreamInlet(mstreams[0])
    inlets: List[Inlet] = []
    #asd?
    print("nominal srate: ",streams[0].nominal_srate())
    
    app=QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.resize(1680,1000)
    win.setWindowTitle('Multichannel LSL - EEG')

    btn_panel = QtWidgets.QWidget()
    plot_panel=pg.GraphicsLayoutWidget()
    save_panel=QtWidgets.QWidget()
    browse_panel=QtWidgets.QWidget()
    featex_panel=QtWidgets.QWidget()
    clf_panel=QtWidgets.QWidget()
    log_panel= QtWidgets.QWidget()

    btn_panel.setLayout(QtWidgets.QVBoxLayout())
    plot_panel.setLayout(QtWidgets.QVBoxLayout())
    save_panel.setLayout(QtWidgets.QHBoxLayout())
    browse_panel.setLayout(QtWidgets.QHBoxLayout())
    featex_panel.setLayout(QtWidgets.QHBoxLayout())
    clf_panel.setLayout(QtWidgets.QHBoxLayout())
    log_panel.setLayout(QtWidgets.QGridLayout())

    Start_Button = QtWidgets.QPushButton("Start",win)
    Stop_Button = QtWidgets.QPushButton("Stop",win)
    # Connect_Button = QtWidgets.QPushButton("connect",win)

    DB_text = QtWidgets.QLabel("DB:")
    DB_markertext = QtWidgets.QLabel("MArker:")
    fixationText =  QtWidgets.QLabel("fix:")
    bufferText = QtWidgets.QLabel("Buf:")
    dataAmountText = QtWidgets.QLabel("dat:")
    fftAt = QtWidgets.QLabel("fft:")
    allText = QtWidgets.QLabel("All:")

    rec_text = QtWidgets.QLabel(":Record")
    prep_text= QtWidgets.QLabel(":Preprocess")
    fext_text = QtWidgets.QLabel(":Feature Extraction")
    clf_text = QtWidgets.QLabel(":Classify")
    typ_text = QtWidgets.QLabel(":Inlet Source")


    Reset_Button = QtWidgets.QPushButton("Reset",win)
    PlotRaw_Button = QtWidgets.QPushButton("PlotRaw",win)
    PlotEvents_Button = QtWidgets.QPushButton("PlotEvents",win)
    PlotEpochs_Button = QtWidgets.QPushButton("PlotEpochs",win)
    PlotSpectral_Button = QtWidgets.QPushButton("PlotSpectral",win)

    BandPower_Button = QtWidgets.QPushButton("FilterBank", win)
    DWavelets_Button = QtWidgets.QPushButton("DiscreteWavelets",win)
    Hilbert_Button = QtWidgets.QPushButton("HilbertTransform",win)

    Classifylog_Button = QtWidgets.QPushButton("Classify LogReg",win)
    ClassiftSVM_Button = QtWidgets.QPushButton("Classify SVM", win)
    
    Save_Name = QtWidgets.QLineEdit(win)
    Save_Button = QtWidgets.QPushButton("SaveRaw",win)
    
    LoadFile_Button = QtWidgets.QPushButton("Browse",win)
    LoadFile_Name = QtWidgets.QLineEdit(win)
    UseLoadedRaw_Button=QtWidgets.QPushButton("UseLoadedRaw",win)

    Unicorn_Button = QtWidgets.QPushButton("General",win)
    Unicorn_Button.setCheckable(True)
    Unicorn_Button.setStyleSheet("background-color:blue")

    def toggle():
        global unicorn
        if Unicorn_Button.isChecked():
            Unicorn_Button.setText("Unicorn")
            Unicorn_Button.setStyleSheet("background-color:red")
            unicorn=True
        else:
            Unicorn_Button.setText("General")
            Unicorn_Button.setStyleSheet("background-color:blue")
            unicorn=False

    def changeName():
        global saveName
        saveName = Save_Name.text()

    DB_text.setText("raw")
    DB_text.setStyleSheet('color: White')
    
    DB_markertext.setText("marker")
    DB_markertext.setStyleSheet('color: White')

    fixationText.setText("Fix")
    fixationText.setStyleSheet('color: White')

    bufferText.setStyleSheet("color: White")
    dataAmountText.setStyleSheet("color: White")
    fftAt.setStyleSheet("color: White")
    allText.setStyleSheet("color: White")

    rec_text.setStyleSheet("color: White")
    prep_text.setStyleSheet("color: White")
    fext_text.setStyleSheet("color: White")
    clf_text.setStyleSheet("color: White")
    typ_text.setStyleSheet("color: White")
    
    btn_panel.layout().setContentsMargins(0,0,0,0)
    btn_panel.layout().setSpacing(0)
    btn_panel.layout().addWidget(rec_text)
    btn_panel.layout().addWidget(Start_Button)
    btn_panel.layout().addWidget(Stop_Button)
    # btn_panel.layout().addWidget(Connect_Button)

    btn_panel.layout().addWidget(log_panel)

    log_panel.layout().addWidget(dataAmountText,0,0)
    log_panel.layout().addWidget(DB_text,1,0)
    log_panel.layout().addWidget(DB_markertext,2,0)
    log_panel.layout().addWidget(fixationText,0,1)
    log_panel.layout().addWidget(bufferText,3,0)
    log_panel.layout().addWidget(fftAt,1,1)
    log_panel.layout().addWidget(allText,3,1)

    btn_panel.layout().addWidget(Reset_Button)
    btn_panel.layout().addWidget(PlotRaw_Button)

    save_panel.layout().setContentsMargins(0,0,0,0)
    save_panel.layout().setSpacing(2)
    btn_panel.layout().addWidget(save_panel)
    save_panel.layout().addWidget(Save_Name)
    save_panel.layout().addWidget(Save_Button)

    browse_panel.layout().setContentsMargins(0,0,0,0)
    browse_panel.layout().setSpacing(0)
    btn_panel.layout().addWidget(browse_panel)
    browse_panel.layout().addWidget(LoadFile_Name)
    browse_panel.layout().addWidget(LoadFile_Button)
    browse_panel.layout().addWidget(UseLoadedRaw_Button)

    btn_panel.layout().addWidget(prep_text)
    btn_panel.layout().addWidget(PlotSpectral_Button)
    btn_panel.layout().addWidget(PlotEvents_Button)
    btn_panel.layout().addWidget(PlotEpochs_Button)

    featex_panel.layout().setContentsMargins(0,0,0,0)
    featex_panel.layout().setSpacing(0)
    btn_panel.layout().addWidget(fext_text) 
    btn_panel.layout().addWidget(featex_panel)
    featex_panel.layout().addWidget(BandPower_Button,1)
    featex_panel.layout().addWidget(DWavelets_Button,1)
    featex_panel.layout().addWidget(Hilbert_Button,1)

    clf_panel.layout().setContentsMargins(0,0,0,0)
    clf_panel.layout().setSpacing(0)
    btn_panel.layout().addWidget(clf_text)
    btn_panel.layout().addWidget(clf_panel)
    clf_panel.layout().addWidget(Classifylog_Button)
    clf_panel.layout().addWidget(ClassiftSVM_Button)

    btn_panel.layout().addWidget(typ_text)
    btn_panel.layout().addWidget(Unicorn_Button)

    layout= QtWidgets.QGridLayout()
    layout.addWidget(btn_panel,0,0)
    layout.addWidget(plot_panel,0,1)

    win.setLayout(layout)
    win.show()

    for info in streams:
        channel_count = info.channel_count()-1
        # override channelcount
        # channel_count=2
        # channel_count =2
        mch= [None]*channel_count # channel handler
        fftch=[None]*channel_count # FFT channel handler
        filtch=[None]*channel_count # filt channel handler
        filtfftch=[None]*channel_count # filt FFT channel handler
        rsarray=[[]for _ in range(channel_count)]
        ftarray=[[]for _ in range(channel_count)]
        fs=info.nominal_srate
        # print("Init RS:",rsarray)
        


    for ch in range(channel_count):
        # main channel
        mch[ch]=plot_panel.addPlot()
        mch[ch].enableAutoRange(x=False, y=True)
        plot_panel.nextRow()

        # fft channel
        fftch[ch]=plot_panel.addPlot()
        fftch[ch].enableAutoRange(x=True, y=True)
        plot_panel.nextRow()

        # filtered channel
        filtch[ch]=plot_panel.addPlot()
        filtch[ch].enableAutoRange(x=False,y=True)
        plot_panel.nextRow()

        # filtered fft channel
        filtfftch[ch]=plot_panel.addPlot()
        filtfftch[ch].enableAutoRange(x=True, y=True)
        plot_panel.nextRow()

        if (info.type() =="EEG"):
            print("adding Data inlet: "+ info.name())
            inlets.append(DataInlet(info, mch[ch],fftch[ch],filtch[ch],filtfftch[ch],bufferText))
            
    tch=plot_panel.addPlot()
    tch.enableAutoRange(x=False,y=True)

    callbackcc =functools.partial(gf.changeCode,100,tch,fixationText)
    saveraw = functools.partial(DataInlet.save_exp,inlets)

    att_interval=5000  # Attention Epoch Interval
    att_timer = QtCore.QTimer()
    att_timer.timeout.connect(callbackcc)

    Start_Button.clicked.connect(lambda: gf.changeCode(96,tch,fixationText))
    Start_Button.clicked.connect(lambda: att_timer.start(att_interval))
    Stop_Button.clicked.connect(lambda: gf.changeCode(97,tch,fixationText))
    Stop_Button.clicked.connect(lambda: att_timer.stop())
    # Connect_Button.clicked.connect(lambda: print("Connecting To Unreal TCP Socket"))
    # Reset_Button.clicked.connect(lambda: reset(inlets,ch))

    Reset_Button.clicked.connect(lambda: reset(inlets))
    PlotRaw_Button.clicked.connect(lambda: DataInlet.convertToRaw(inlets))
    PlotEvents_Button.clicked.connect(lambda: DataInlet.find_events(inlets))
    PlotEpochs_Button.clicked.connect(lambda: DataInlet.epoch_plot(inlets))
    PlotSpectral_Button.clicked.connect(lambda: DataInlet.spect_plot(inlets))
    Unicorn_Button.clicked.connect(toggle)
    Hilbert_Button.clicked.connect(lambda: DataInlet.hilbert_plot(inlets))
    Classifylog_Button.clicked.connect(lambda: DataInlet.clf_logreg(inlets))
    ClassiftSVM_Button.clicked.connect(lambda: DataInlet.clf_SVM(inlets))
    BandPower_Button.clicked.connect(lambda: DataInlet.filter_bank(inlets))
    DWavelets_Button.clicked.connect(lambda: DataInlet.discrete_wavelet(inlets))

    Save_Name.textChanged.connect(changeName)
    Save_Button.clicked.connect(saveraw)
    
    plt.ion()

    def browsefiles():
        global fn
        directory = os.getcwd()
        fname= QtWidgets.QFileDialog.getOpenFileName(win,"Open file",directory)
        fn=fname[0]
        # print(fn)
        LoadFile_Name.setText(fn)

    LoadFile_Button.clicked.connect(browsefiles)
    UseLoadedRaw_Button.clicked.connect(lambda: DataInlet.loadraw(inlets))

    def reset(inlets):
        global mmarray,rsarray,data_count,db_counter,record,att_count
        mmarray=[]
        rsarray=[[]for _ in range(channel_count)]
        ftarray=[[]for _ in range(channel_count)]
        db_counter=0
        data_count=0
        record=False
        att_count=0
        for ch,inlet in enumerate(inlets):
            inlet.reset_buffer(ch)
        print("--program reset--")


    def scroll():
        fudge_factor = pull_interval * .01
        plot_time = pylsl.local_clock()
        for pltnumber in range(channel_count):
            mch[pltnumber].setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)
            filtch[pltnumber].setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)
            tch.setXRange(plot_time-plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        mintime = pylsl.local_clock() - plot_duration
        for ch,inlet in enumerate(inlets):
            inlet.pull_and_plot(mintime, ch,tch,DB_text,DB_markertext,fixationText,dataAmountText)

            # if triggerLine.mouseClickEvent:
            #     inlet.update_trigLine(triggerVal)
    def update_fft():
        for ch,inlet in enumerate(inlets):
            inlet.plot_fft(ch,fftAt,allText)

    
    
    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    fft_timer = QtCore.QTimer()
    fft_timer.timeout.connect(update_fft)
    fft_timer.start(fft_interval)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # app = QtWidgets.QApplication(sys.argv) 
        # app = QtGui.QGuiApplication(sys.argv)
        app.exec()

    return 0
    # Connection Section
    # global conn
    # global command

    # s_sender = socket.socket()         # Create a socket object
    # host = "127.0.0.1"              # Bind Local
    # port = 9030                # Reserve a port for your service.
    # connected=False
    # s_sender.bind((host,port))

    # # Listen to incoming connection
    # s_sender.listen(5)

    # # Connecting Prompt
    # try:
    #     conn, addr =s_sender.accept()
    #     print ('Open Listening Server', host, port)
    #     connected=True
    #     # Establish connection with client.
    #     print('Got connection from', addr)
    # except s_sender.error :
    #     connected=False
    #     print("Connection DOOM, did you open the Client Yet?")
        
    
    # while connected:
    #     try:
    #         DataInlet.sendCommand(command,conn,ch,mintime)
    #     except ConnectionAbortedError:
    #         print("Connection Doom` due to something wa")
    #         connected = False

    # cmd_timer = QtCore.QTimer()
    # cmd_timer.timeout.connect(DataInlet.sendCommand(command,conn,ch,mintime))
    # cmd_timer.start(cmd_interval)

if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv) 
    main()