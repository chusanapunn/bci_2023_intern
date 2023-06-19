
# Run this Second, then UE socket Connect

import pylsl
import math as math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from typing import List
import scipy.signal as sig

import socket
import time

# Monitor
plot_duration = 5  # how many seconds of data to show
update_interval = 12  # ms between screen updates
pull_interval = 60 # ms between each pull operation
fft_interval = 500 # ms between each FFT calculation

# Socket NETWORK
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
#      # Establish connection with client.
#     print('Got connection from', addr)
# except s_sender.error :
#     connected=False
#     print("Connection DOOM, did you open the Client Yet?")

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
    def __init__(self,info:pylsl.StreamInfo,plt:pg.PlotItem,fftplt:pg.PlotItem,filtplt:pg.PlotItem,filtfftplt:pg.PlotItem):
        dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
        super().__init__(info)

        self.bufsize,self.buffer,empty=set_buffer(info,dtypes)
        global N
        
        
        # signal Curves
        self.curves = [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.fftcurves= [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtcurves= [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtfftcurves=[pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        
        # Set Triggervalue and TriggerLine
        # global triggerValA,triggerValB
        # triggerValA=1.3
        # triggerValB=0.5
        # global triggerLine
        # triggerLine=pg.InfiniteLine(label="SimpleTriggerLine",pos=triggerValA,movable=True,angle=0)
        

        for curve in self.curves:
            plt.addItem(curve)
            # plt.addItem(triggerLineA)
        for fftcurve in self.fftcurves:
            fftplt.addItem(fftcurve)
        for filtcurve in self.filtcurves:
            filtplt.addItem(filtcurve)
            # filtplt.addItem(triggerLineA)
        for filtfftcurve in self.filtfftcurves:
            filtfftplt.addItem(filtfftcurve)

    # def update_trigLine(triggerVal):
    #     print("update TriggerLine"+str(triggerLine.value())) 
    #     triggerVal=triggerLine.value
    #     self.triggerLine.setValue(triggerVal)
    #     return triggerVal

    def pull_and_plot(self, plot_time, ch):
        global _,ts,this_x,this_y,filtered_thisY
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        this_x=None
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
            this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
            this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch] - ch))
             # Filter and plot signal
            if (ch==0):     # channel 1 filter
                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
                # filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)
                # filtered_thisY=butter_lowpass_filter(this_y,30,fs,4)
                filtered_thisY=butter_bandstop_filter(this_y,fs,4,2,8)
                
            elif (ch==1):     # channel 1 filter
                filtered_thisY=butter_bandstop_filter(this_y,fs,4,18,24)
                # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)

            elif (ch==2):     # channel 1 filter
                filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)

            self.curves[ch].setData(this_x, this_y)
            self.filtcurves[ch].setData(this_x,filtered_thisY)
        #     fft_N=int(self.fftbufsize[0]/fs)
        # #     # plot fft
        # #     # global this_y,filtered_thisY
        #     fft_xlength=len(this_x)
        #     frequencies=np.fft.fftfreq(fft_xlength,d=1/fs)
        #     freq_fftx=frequencies[:fft_xlength]
        #     freq_fftx=abs(freq_fftx)
            
        #     this_fftY=np.fft.fft(this_y)
        #     this_fftY=np.abs(this_fftY)/(fft_N)
        #     filtfft_y=np.fft.fft(filtered_thisY)
        #     filtfft_y=np.abs(filtfft_y)/(fft_N)
            
            
        #     self.fftcurves[ch].setData(freq_fftx,this_fftY)
        #     self.filtfftcurves[ch].setData(freq_fftx,filtfft_y)
    def plot_fft(self,ch):

        data=self.buffer[:,ch]
        fft_result=np.fft.fft(data)
        fft_result=np.abs(fft_result)
        freq=np.fft.fftfreq(data.size,d=1/fs)
        freq=np.abs(freq)
        self.fftcurves[ch].setData(freq,fft_result)

        if (ch==0):     # channel 1 filter
            # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
            # filtered_thisY=butter_bandstop_filter(this_y,fs,4,28,34)
            # filtered_thisY=butter_lowpass_filter(this_y,30,fs,4)
            filtered_data=butter_bandstop_filter(data,fs,4,8,12)
        elif (ch==1):     # channel 1 filter
            filtered_data=butter_bandstop_filter(data,fs,4,19,23)
            # filtered_thisY=butter_lowpass_filter(this_y,26,fs,4)
        elif (ch==2):     # channel 1 filter
            filtered_data=butter_bandstop_filter(data,fs,4,28,34)
        else:
            filtered_data=data
        filtered_fft_result = np.fft.fft(filtered_data)
        filtered_fft_result=np.abs(filtered_fft_result)
        self.filtfftcurves[ch].setData(freq, filtered_fft_result)
        
    #     # global command
    #     # triggerVal=self.update_trigLine()
        # if (filtered_thisY[[xlength-1]]>triggerValA):
        #     command="A"
        #     print("Channel: "+ str(ch+1) +" [{:2.3f}] :".format(filtered_thisY[xlength-1])\
        #           +"{:1.3f}".format(filtered_thisY[xlength-1]-triggerValA)\
        #               + " More than "+str(triggerValA)+": Command = "+command)
            
        # elif (filtered_thisY[[xlength-1]]<triggerValB):
        #     command="S"
        #     print("Channel: "+ str(ch+1) +" [{:2.3f}] :".format(filtered_thisY[xlength-1])\
        #             +"{:1.3f}".format(filtered_thisY[xlength-1]-triggerValB)\
        #                 + " More than "+str(triggerValB)+": Command = "+command)
            
    # def sendCommand(cmd,c):
    #     c.send(cmd.encode())
    #     print("--Sent message--")
        
    # while connected:
    #     try:
    #         time.sleep(1)
    #         sendCommand(command,conn)
    #     except ConnectionAbortedError:
    #         print("Connection Doom` due to something wa")
    #         connected = False

def set_buffer(info,dtypes):
    bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
    buffer = np.empty(bufsize, dtype=dtypes[info.channel_format()])
    empty = np.array([])
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

def main():
    # first resolve a marker stream on the lab network
    print("looking for a marker stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    inlets: List[Inlet] = []

    win = pg.GraphicsLayoutWidget(show=True, title="lsl multichannel")
    win.resize(1000,600)
    win.setWindowTitle('Multichannel LSL - EEG')


    for info in streams:
        print(info)
        channel_count = info.channel_count()
        mch= [None]*channel_count # channel handler
        fftch=[None]*channel_count # FFT channel handler
        filtch=[None]*channel_count # filt channel handler
        filtfftch=[None]*channel_count # filt FFT channel handler

    # override channelcount
    channel_count=4

    for ch in range(channel_count):
        # main channel
        mch[ch]=win.addPlot()
        mch[ch].enableAutoRange(x=False, y=True)
        win.nextRow()

        # fft channel
        fftch[ch]=win.addPlot()
        win.nextRow()

        # filtered channel
        filtch[ch]=win.addPlot()
        filtch[ch].enableAutoRange(x=False,y=True)
        win.nextRow()

        # filtered fft channel
        filtfftch[ch]=win.addPlot()
        win.nextRow()
        if info.type() =="EEG":
            print("adding Data inlet: "+ info.name())
            inlets.append(DataInlet(info, mch[ch],fftch[ch],filtch[ch],filtfftch[ch]))

    def scroll():
        fudge_factor = pull_interval * .01
        plot_time = pylsl.local_clock()
        for pltnumber in range(channel_count):
            mch[pltnumber].setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)
            filtch[pltnumber].setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)
    
    def update():
        mintime = pylsl.local_clock() - plot_duration
        for ch,inlet in enumerate(inlets):
            inlet.pull_and_plot(mintime, ch)

            # if triggerLine.mouseClickEvent:
            #     inlet.update_trigLine(triggerVal)
    def update_fft():
        for ch,inlet in enumerate(inlets):
            inlet.plot_fft(ch)
    
        
    
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

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtWidgets.QApplication(sys.argv) 
        app.exec()




if __name__ == '__main__':
    main()