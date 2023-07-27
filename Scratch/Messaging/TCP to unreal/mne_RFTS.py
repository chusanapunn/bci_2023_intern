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

import random
import socket
import time

import sys

plot_duration = 4 # how many seconds of data to show
update_interval = 12  # ms between screen updates
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
unitz = 100000
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
        self.bufsize,self.buffer,self.empty=set_buffer(info,dtypes,bufferText)
        
        
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
                filtered_thisY=butter_bandstop_filter(this_y,fs,4,49,51)
                # print("skip filter")
                # filtered_thisY = this_y
                
            elif (ch==1):     # channel 1 filter
                filtered_thisY=butter_bandstop_filter(this_y,fs,4,47,53)
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
                        changeCode(0,tch,fixationText)
                
            dataAmountText.setText("Data Input Amount: "+str(int(db_counter)))
            DB_text.setText("Raw Data Count: " + str(len(rsarray[1])))
            DB_markertext.setText("Marker Data Count: " + str(len(mmarray)))
            # print("Y",y.shape)
            # print("RS",rsarray.shape)
            # print(mmarray)
            
            # if (DB_markertext.text()==DB_text.text()):
            #     fixationText.setText("Matched!!")
            # print("To be raw data"+str(filtered_thisY[0:10]))
            # print("To be stim channel"+str(mmarray))
    # def toggleUnicorn(self,unicorn):
        

    def convertToRaw(self):
        global rsarray,mmarray,channel_count,fs,filtraw,ch1_pick,ch2_pick,unicorn
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
        print(info,stim_info)

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
            filtraw=filtraw.filter(1,35)
        else:
            filtraw=raw.copy()
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
        raw.plot(n_channels=2,show_scrollbars=True, show=True,scalings="auto",title="Raw Signal")
        filtraw.plot(n_channels=2,show_scrollbars=True, show=True,scalings="auto",title = "Filtered Signal")

        
        # plt.draw()
        # events,sfreq=raw.info["sfreq"],first_samp=raw.first_samp,event_id=events_dict
        # fig=mne.viz.plot_events(events,sfreq)
        # fig.subplots_adjust(right=0.7)
    def spect_plot(self):
        global filtraw,ch1_pick,ch2_pick
        spectrum = filtraw.compute_psd()
        spectrum.plot(average=True, picks=["eeg01"], exclude='bads')
        spectrum.plot(average=True, picks=["eeg02"], exclude='bads')

    def find_events(self):
        global filtraw,fs,events_dict,events
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
        filtraw.plot(events,event_color={1000:"r",2000:"g",100:"b",96:"y"},scalings="auto",title="Filtered Events")

    def epoch_plot(self):
        global filtraw,events,events_dict
        epochs = mne.Epochs(filtraw, events, event_id=events_dict, preload=True)
        epochs.plot(
            events=events,event_id=events_dict,title= "Epochs",scalings="auto"
        )
        # epochs["relax"].plot(
        #     events=events,event_id=events_dict,scalings="auto"
        # )


    def plot_fft(self,ch,fftAt,allText):
        global fs,data_count,ftarray,channel_count,unicorn,unitz,this_y,rsarray,ftch
        # debug fft plot interval
        # ftarraysize=len(ftarray)
        # print("be",data)
        # s  data= np.asarray([a/unitz for a in data])
        # ftarray=np.empty(self.buffer.shape)
        # print("before",ftarray.shape)
        # print(ftarray.shape)
        # print(this_y.shape)
        # print(ftarray.shape)
        # print(ftarray)
        ftch=np.asarray(ftarray[ch], dtype=np.float64)
        # ftch=np.append(ftch,this_y)
        # print(ftch.shape)
        # ftch = np.append(ftch,this_y)
        
        
        try:
            # ftarray = data[np.where(data!=0)]
            # ftarray =ftarray[:,ch]
            # ftarray = ftarray[np.where(ftarray!=0)]
            # print("MAX: ",ftarray.max())
            # print("Min: ",ftarray.min())
            # print(ftarray)
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
                filtered_data=butter_bandstop_filter(ftch,fs,4,49,51)
                # filtered_data = ftarray
        
            elif (ch==1):     # channel 1 filter
                filtered_data=butter_bandstop_filter(ftch,fs,4,47,53)
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
        

def set_buffer(info,dtypes,bufferText):
    bufsize = (math.ceil(1.5*math.ceil(info.nominal_srate()*plot_duration)), info.channel_count())
    buffer = np.empty(bufsize, dtype=dtypes[info.channel_format()])
    empty = np.array([])
    bufferText.setText("bufferSize is: "+str(bufsize[0])+". Stream Channel Count:"+str(bufsize[1]))

    # rsarray=np.empty(bufsize, dtype=dtypes[info.channel_format()])
    # print("RS",rsarray.transpose())
    return bufsize,buffer,empty

def butter_bandstop_filter(data, fs, order, a, b):
        # Get the filter coefficients  
        b, a = sig.butter(order, [a,b], 'bandstop', fs=fs, output='ba')
        y = sig.filtfilt(b, a, data)
        return y

def butter_lowpass_filter(data, cutOff, fs, order):
        
        # Get the filter coefficients 
        b_lp, a_lp = sig.butter(order, cutOff, 'lowpass', fs=fs, output='ba')
        y = sig.filtfilt(b_lp, a_lp, data)
        return y


    # tmarray = []

def changeCode(code,tch:pg.PlotItem,fixationText):
    global mmarray,record,att_count,codearray
    labela=["Up","Down",'Right',"Left"]
    label=None

    if (code==0):
        # label=random.choice(labela)
        label=None
    elif(code==96):
        # if(mmarray):
        #     print(mmarray.pop())
        label="Start Recording"
        record=True
        codearray=[]


    elif(code==97 and record):
        if(mmarray):
            mmarray.pop()
        label="Stop Recording"
        record=False
    # elif(code==99):

        # att_count+=1
        # if(mmarray):
        #     mmarray.pop()

    elif(code==100):
        # if(mmarray):
        #     print(mmarray.pop())
        if (att_count<max_epoch):
            label="Attention :" +str(att_count+1)
            code=1000
            record=True
            
        elif (att_count>=max_epoch and att_count<max_epoch*2):
            label="Relax :" +str(att_count-max_epoch+1)
            code=2000
            
        elif (att_count>max_epoch*2):
            code=0
            changeCode(97,tch,fixationText)
        
        if(mmarray and record):
            mmarray.pop()         
        att_count+=1
    else:
        label=None
        code=0
    mx=pylsl.local_clock()
    
    if (label and label not in labela):
        mcurve=pg.InfiniteLine(mx, angle=90, movable=False, label=label,pen="Red")
        tch.addItem(mcurve)
        fixationText.setText(label)

    if (record):
        mmarray.append(code)
        # print("Marker:",mmarray)
        if (code!=0):
            codearray.append(code)
            print("Code:",codearray)
            # if (mmarray>rsarray):
                # mmarray.pop()
        # if (code==96):
        #     if(mmarray):
        #         print(mmarray.pop())
    # codearray=mmarray[np.where(mmarray!=0)]
    # print(codearray)

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
    win.resize(1680,1500)
    win.setWindowTitle('Multichannel LSL - EEG')

    btn_panel = QtWidgets.QWidget()
    plot_panel=pg.GraphicsLayoutWidget()

    btn_panel.setLayout(QtWidgets.QVBoxLayout())
    plot_panel.setLayout(QtWidgets.QVBoxLayout())

    Start_Button = QtWidgets.QPushButton("Start",win)
    Stop_Button = QtWidgets.QPushButton("Stop",win)
    Connect_Button = QtWidgets.QPushButton("connect",win)
    DB_text = QtWidgets.QLabel("DB:")
    DB_markertext = QtWidgets.QLabel("MArker:")
    fixationText =  QtWidgets.QLabel("fix:")
    bufferText = QtWidgets.QLabel("Buf:")
    dataAmountText = QtWidgets.QLabel("dat:")
    fftAt = QtWidgets.QLabel("fft:")
    allText = QtWidgets.QLabel("All:")
    Reset_Button = QtWidgets.QPushButton("Reset",win)
    PlotRaw_Button = QtWidgets.QPushButton("PlotRaw",win)
    PlotEvents_Button = QtWidgets.QPushButton("PlotEvents",win)
    PlotEpochs_Button = QtWidgets.QPushButton("PlotEpochs",win)
    PlotSpectral_Button = QtWidgets.QPushButton("PlotSpectral",win)
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

    btn_panel.layout().addWidget(Start_Button)
    btn_panel.layout().addWidget(Stop_Button)
    btn_panel.layout().addWidget(Connect_Button)

    btn_panel.layout().addWidget(dataAmountText)
    btn_panel.layout().addWidget(DB_text)
    btn_panel.layout().addWidget(DB_markertext)
    btn_panel.layout().addWidget(fixationText)
    btn_panel.layout().addWidget(bufferText)
    btn_panel.layout().addWidget(fftAt)
    btn_panel.layout().addWidget(allText)

    btn_panel.layout().addWidget(Reset_Button)
    btn_panel.layout().addWidget(PlotRaw_Button)
    btn_panel.layout().addWidget(PlotSpectral_Button)
    btn_panel.layout().addWidget(PlotEvents_Button)
    btn_panel.layout().addWidget(PlotEpochs_Button)
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

    callbackcc =functools.partial(changeCode,100,tch,fixationText)
    att_interval=1000  # Attention Epoch Interval
    att_timer = QtCore.QTimer()
    att_timer.timeout.connect(callbackcc)

    Start_Button.clicked.connect(lambda: changeCode(96,tch,fixationText))
    Start_Button.clicked.connect(lambda: att_timer.start(att_interval))
    Stop_Button.clicked.connect(lambda: changeCode(97,tch,fixationText))
    Stop_Button.clicked.connect(lambda: att_timer.stop())
    Connect_Button.clicked.connect(lambda: print("Connecting To Unreal TCP Socket"))
    # Reset_Button.clicked.connect(lambda: reset(inlets,ch))

    
    Reset_Button.clicked.connect(lambda: reset(inlets))
    PlotRaw_Button.clicked.connect(lambda: DataInlet.convertToRaw(inlets))
    PlotEvents_Button.clicked.connect(lambda: DataInlet.find_events(inlets))
    PlotEpochs_Button.clicked.connect(lambda: DataInlet.epoch_plot(inlets))
    PlotSpectral_Button.clicked.connect(lambda: DataInlet.spect_plot(inlets))
    Unicorn_Button.clicked.connect(toggle)

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