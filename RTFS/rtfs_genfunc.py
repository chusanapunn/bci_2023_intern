import math as math
import numpy as np
import pyqtgraph as pg
import scipy.signal as sig
import pylsl

max_epoch=5
plot_duration = 5

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

def butter_bandpass_filter(data, fs, order, a, b):
        # Get the filter coefficients  
        b, a = sig.butter(order, [a,b], 'bandpass', fs=fs, output='ba')
        y = sig.filtfilt(b, a, data)
        return y

def butter_lowpass_filter(data, cutOff, fs, order):
        
        # Get the filter coefficients 
        b_lp, a_lp = sig.butter(order, cutOff, 'lowpass', fs=fs, output='ba')
        y = sig.filtfilt(b_lp, a_lp, data)
        return y


def butter_highpass_filter(data, cutOff, fs, order):
        nyq=0.5*fs
        normcutoff = cutOff/nyq
        # Get the filter coefficients 
        b_hp, a_hp = sig.butter(order, normcutoff, 'highpass', fs=fs, output='ba')
        y = sig.filtfilt(b_hp, a_hp, data)
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

