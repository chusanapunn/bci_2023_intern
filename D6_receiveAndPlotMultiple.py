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

class DataInlet(Inlet):  
    def __init__(self,info:pylsl.StreamInfo,plt:pg.PlotItem):
        dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]
        super().__init__(info)
        self.bufsize,self.buffer,empty=set_buffer(info,dtypes)
        self.curves = [pg.PlotCurveItem(x=empty, y=empty , autoDownsample=True) for _ in range(self.channel_count)]
        for curve in self.curves:
            plt.addItem(curve)
            
    def pull_and_plot(self, plot_time, ch):
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

def set_buffer(info,dtypes):
    bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
    buffer = np.empty(bufsize, dtype=dtypes[info.channel_format()])
    empty = np.array([])
    return bufsize,buffer,empty

def butter_50hz_filter(data, fs, order):
        # Get the filter coefficients 
        b, a = sig.butter(order, [47,53], 'bandstop', fs=fs, output='ba')
        y = sig.filtfilt(b, a, data)
        return y
def butter_lowpass_filter(data, cutOff,fs, order):
        # Get the filter coefficients 
        b_lp, a_lp = sig.butter_lowpass(order, [47,53], 'bandstop', fs=fs, output='ba')
        y = sig.filtfilt(b_lp, a_lp, data)
        return y

def main():
    # first resolve a marker stream on the lab network
    print("looking for a marker stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    # inlet = pylsl.StreamInlet(streams[0])
    #info = pylsl.StreamInfo('openvibe', 'EEG', 4, 500, 'float32')
    inlets: List[Inlet] = []

    win = pg.GraphicsLayoutWidget(show=True, title="lsl multichannel")
    win.resize(1000,600)
    win.setWindowTitle('Multichannel LSL - EEG')

    for info in streams:
        print(info)
        mch=[None]*info.channel_count() #channel handler

    for ch in range(info.channel_count()):
        mch[ch]=win.addPlot()
        # pw = pg.plot(title='multiple Channel lsl Plot')
        # pw[ch] = mch[ch].getPlotItem()
        mch[ch].enableAutoRange(x=False, y=True)
        win.nextRow()
        if info.type() =="EEG":
            print("adding Data inlet: "+ info.name())
            inlets.append(DataInlet(info, mch[ch]))

    def scroll():
        fudge_factor = pull_interval * .01
        plot_time = pylsl.local_clock()
        for pltnumber in range(info.channel_count()):
            mch[pltnumber].setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        mintime = pylsl.local_clock() - plot_duration
        #for pltnumber in range(info.channel_count()):
        for ch,inlet in enumerate(inlets):
            inlet.pull_and_plot(mintime, ch)
    
    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtWidgets.QApplication(sys.argv) 
        app.exec()

if __name__ == '__main__':
    main()