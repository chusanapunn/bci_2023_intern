import pylsl
import math as math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from typing import List
import scipy.signal as sig

# animate plot parameters
plot_duration = 5  # how many seconds of data to show
update_interval = 12  # ms between screen updates
pull_interval = 60 # ms between each pull operation
fft_interval = 500 # ms between each FFT calculation


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
        N=int(self.bufsize[0]/fs)
        
        # signal Curves
        self.curves = [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.fftcurves= [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtcurves= [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        self.filtfftcurves=[pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        
        # Set Triggervalue and TriggerLine
        global triggerVal
        triggerVal=1.3
        global triggerLine
        triggerLine=pg.InfiniteLine(label="SimpleTriggerLine",pos=triggerVal,movable=True,angle=0)
        

        for curve in self.curves:
            plt.addItem(curve)
            # plt.addItem(triggerLine)
        for fftcurve in self.fftcurves:
            fftplt.addItem(fftcurve)
        for filtcurve in self.filtcurves:
            filtplt.addItem(filtcurve)
            filtplt.addItem(triggerLine)
        for filtfftcurve in self.filtfftcurves:
            filtfftplt.addItem(filtfftcurve)

    # def update_trigLine(triggerVal):
    #     print("update TriggerLine"+str(triggerLine.value())) 
    #     triggerVal=triggerLine.value
    #     self.triggerLine.setValue(triggerVal)
    #     return triggerVal

    def pull_and_plot(self, plot_time, ch):
        global _,ts,this_x,this_y
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0:ts.size, :]
            this_x = None
            old_offset = 0
            new_offset = 0
            old_x, old_y = self.curves[ch].getData()

            old_offset = old_x.searchsorted(plot_time)
            new_offset = ts.searchsorted(plot_time)
            this_x = np.hstack((old_x[old_offset:], ts[new_offset:]))
            this_y = np.hstack((old_y[old_offset:], y[new_offset:, ch] - ch))

            self.curves[ch].setData(this_x, this_y)
            # Debug
            # print("X:"+str(this_x),end="")
            # print("Y:"+str(this_y))
            

            
    def plot_fft(self,ch):
        # plot fft
        global filtered_thisY

        xlength=len(this_x)

        frequencies=np.fft.fftfreq(xlength,d=1/fs)
        new_fftX=frequencies[:xlength]
        new_fftX=abs(new_fftX)
        new_fftY=np.fft.fft(this_y)
        new_fftY=np.abs(new_fftY)/(N)
        # new_fftX=new_fftX.flatten()
        # new_fftY=new_fftY.flatten()

        self.fftcurves[ch].setData(new_fftX,new_fftY)

        # Filter and plot signal
        if (ch==0):     # channel 1 filter
            filtered_thisY=butter_bandstop_filter(this_y,fs,4,17,23)
        elif (ch==1):   # channel 2 filter
            filtered_thisY=butter_bandstop_filter(this_y,fs,4,2,8)
        # filtered_thisY=butter_bandstop_filter(this_y,fs,4,17,23)
        self.filtcurves[ch].setData(this_x,filtered_thisY)

        # fft of filtered signal
        filtfft_y=np.fft.fft(filtered_thisY)
        filtfft_y=np.abs(filtfft_y)/(N)
        self.filtfftcurves[ch].setData(new_fftX,filtfft_y)
        global command
        # triggerVal=self.update_trigLine()
        if (filtered_thisY[[xlength-1]]>triggerVal):
            command="W"
            print("Channel: "+ str(ch+1) +" [{:2.3f}] :".format(filtered_thisY[xlength-1])\
                  +"{:1.3f}".format(filtered_thisY[xlength-1]-triggerVal)\
                      + " More than "+str(triggerVal)+command)
        

def set_buffer(info,dtypes):
    bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
    buffer = np.empty(bufsize, dtype=dtypes[info.channel_format()])
    empty = np.array([])
    return bufsize,buffer,empty

def butter_bandstop_filter(data, fs, order, a,b):
        # Get the filter coefficients  
        b, a = sig.butter(order, [a,b], 'bandstop', fs=fs, output='ba')
        y = sig.filtfilt(b, a, data)
        return y

def butter_lowpass_filter(data, cutOff,fs, order):
        # Get the filter coefficients 
        b_lp, a_lp = sig.butter_lowpass(order, cutOff, 'bandstop', fs=fs, output='ba')
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
    channel_count=1

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
        print("-----------FFT Update-----------")

    
        
    
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