import pylsl
import math as math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtGui import QFont
from typing import List
import scipy.signal as sig
import scipy.fftpack as fft
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pywt as pywt
import spkit as sp
import random
import socket
import time

import rfc_markerUI as mui     # Focus Experiment UI

from rfc_genFunction import *  # General Function for application

import sys
import os as os    
import functools   # function call with fixed parameter for QTimer

# Disable warning for classifier of whom can using GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plot_duration = 5 # how many seconds of data to show
update_interval = 20  # ms between screen updates
pull_interval = 60 # ms between each pull operation
fft_interval = 500 # ms between each FFT calculation and triggering
global rawSignalArray,unicorn,unitz,triggerCode,epochs
triggerCode=0   
markerSignalArray = []
codearray=[]
inputSignalCounter=0
record=False
att_count=0
rlx_count=0
max_epoch=10
signalDataCount=0
unicorn = False 
loadraw=False
unitz = 1000000

# Initialize app
app=QtWidgets.QApplication(sys.argv)

def main():
    # first resolve a marker stream on the lab network
    print("looking for a marker stream...")
    global rawSignalArray,fftSignalArray,channel_count,fs,unicorn,analyzeChannel,triggerCode,saveName
    streams = pylsl.resolve_stream('type', 'EEG')
    # mstreams=pylsl.resolve_stream("type","Markers")
    # print(mstreams)
    # minlet=pylsl.StreamInlet(mstreams[0])
    inlets: List[Inlet] = []
    analyzeChannel = 0
    #asd?
    # print("nominal srate: ",streams[0].nominal_srate())

    win = pg.GraphicsLayoutWidget()
    win.resize(1400,800)
    win.setWindowTitle('Multichannel LSL - EEG')

    marker_ui = mui.MarkerWindow()
    marker_ui.close()

    for info in streams:
        channel_count = info.channel_count()-1
        # override channelcount
        # channel_count =3
        mch= [None]*channel_count # channel handler
        fftch=[None]*channel_count # FFT channel handler
        filtch=[None]*channel_count # filt channel handler
        # filtfftch=[None]*channel_count # filt FFT channel handler
        rawSignalArray=[[]for _ in range(channel_count)]
        fftSignalArray=[[]for _ in range(channel_count)]
        fs=info.nominal_srate
        # print("Init RS:",rawSignalArray)

    def toggle():
        global unicorn
        if Unicorn_Button.isChecked():
            Unicorn_Button.setText("Unicorn")
            Unicorn_Button.setStyleSheet("background-color:red")
            unicorn=True
            mui.unicorn= True
        else:
            Unicorn_Button.setText("General")
            Unicorn_Button.setStyleSheet("background-color:blue")
            unicorn=False
            mui.unicorn= False

    def changeName():
        global saveName
        saveName = Save_Name.text()
        mui.saveName = saveName

    def selectChannel():
        global analyzeChannel
        for c,i in enumerate(ch_radio_buttons):
            if (i.isChecked() ==True):
                analyzeChannel = c
                print(i.isChecked(),str(c))
        print("Channel To analyze : "+str(analyzeChannel+1))
    
    plot_panel=pg.GraphicsLayoutWidget()
    btn_panel = QtWidgets.QWidget()

    save_panel=QtWidgets.QWidget()
    browse_panel=QtWidgets.QWidget()
    featex_panel=QtWidgets.QWidget()
    clf_panel=QtWidgets.QWidget()
    log_panel= QtWidgets.QWidget()
    rec_panel = QtWidgets.QWidget()
    prp_panel = QtWidgets.QWidget()

    mainPanels =[btn_panel,plot_panel]
    
    subPanels = [ save_panel,browse_panel,featex_panel,
                 clf_panel,rec_panel,prp_panel]

    log_panel.setLayout(QtWidgets.QGridLayout())

    for i in mainPanels:
        i.setLayout(QtWidgets.QVBoxLayout())

    for i in subPanels:
        i.setLayout(QtWidgets.QHBoxLayout())
    
    Start_Button = QtWidgets.QPushButton("Start",win)
    Stop_Button = QtWidgets.QPushButton("Stop",win)
    Reset_Button = QtWidgets.QPushButton("Reset",win)
    PlotRaw_Button = QtWidgets.QPushButton("PlotRaw",win)

    recButton = [Start_Button,Stop_Button,Reset_Button,PlotRaw_Button]

    DB_text = QtWidgets.QLabel("DB:")
    DB_markertext = QtWidgets.QLabel("MArker:")
    fixationText =  QtWidgets.QLabel("Fix:")
    fixationText2= QtWidgets.QLabel("Fix2:")
    bufferText = QtWidgets.QLabel("Buf:")
    dataAmountText = QtWidgets.QLabel("dat:")
    fftAt = QtWidgets.QLabel("fft:")
    allText = QtWidgets.QLabel("All:")

    logText = [DB_text,DB_markertext,fixationText,fixationText2,
               bufferText,dataAmountText,fftAt,allText]

    rec_text = QtWidgets.QLabel(":Record")
    prep_text= QtWidgets.QLabel(":Preprocess")
    fext_text = QtWidgets.QLabel(":Feature Extraction")
    clf_text = QtWidgets.QLabel(":Classify")
    typ_text = QtWidgets.QLabel(":Inlet Source")

    titleText = [rec_text,prep_text,fext_text,clf_text,typ_text]

    PlotEvents_Button = QtWidgets.QPushButton("PlotEvents",win)
    PlotEpochs_Button = QtWidgets.QPushButton("PlotEpochs",win)
    PlotSpectral_Button = QtWidgets.QPushButton("PlotSpectral",win)
    PlotPCA_Button = QtWidgets.QPushButton("PlotPCA",win)

    ch_radio_buttons = list()
    for i in range(channel_count):
        t_radio = QtWidgets.QRadioButton(str(i+1))
        ch_radio_buttons.append(t_radio)

    for ii in ch_radio_buttons:
        ii.clicked.connect(selectChannel)

    ch_radio_buttons[0].setChecked(True)

    BandPower_Button = QtWidgets.QPushButton("FilterBank", win)
    DWavelets_Button = QtWidgets.QPushButton("DiscreteWavelets",win)
    Hilbert_Button = QtWidgets.QPushButton("HilbertTransform",win)

    Classifylog_Button = QtWidgets.QPushButton("Classify LogReg",win)
    ClassifySVM_Button = QtWidgets.QPushButton("Classify SVM", win)
    ClassifyDNN_Button = QtWidgets.QPushButton("Classify DNN", win)

    Save_Name = QtWidgets.QLineEdit("savefile name",win)
    Save_Button = QtWidgets.QPushButton("SaveRaw",win)
    
    LoadFile_Button = QtWidgets.QPushButton("Browse",win)
    LoadFile_Name = QtWidgets.QLineEdit("loadfile name",win)
    UseLoadedRaw_Button=QtWidgets.QPushButton("UseLoadedRaw",win)

    Unicorn_Button = QtWidgets.QPushButton("General",win)
    Unicorn_Button.setCheckable(True)
    Unicorn_Button.setStyleSheet("background-color:blue; color:White")

    for i in (logText+titleText):
        i.setStyleSheet("color: White")

    font1 = QFont("Helvetica",12)
    font2 = QFont("Helvetica",8)

    for i in logText:
        i.setFont(font2)

    for i in titleText:
        i.setFont(font1)

    Unicorn_Button.setFont(font2)
    LoadFile_Button.setFont(font2)
    UseLoadedRaw_Button.setFont(font2)
    LoadFile_Name.setFont(font2)
    Save_Name.setFont(font2)
    Save_Button.setFont(font2)
    # Atar_Button.setFont(font2)

    for i in ch_radio_buttons:
        i.setFont(font2)
    
    for i in recButton:
        rec_panel.layout().addWidget(i)
    # (Connect_Button)

    log_panel.layout().addWidget(dataAmountText,0,0)
    log_panel.layout().addWidget(DB_text,1,0)
    log_panel.layout().addWidget(DB_markertext,2,0)
    log_panel.layout().addWidget(fixationText,0,1)
    log_panel.layout().addWidget(fixationText2,1,1)
    log_panel.layout().addWidget(bufferText,3,0)
    log_panel.layout().addWidget(fftAt,2,1)
    log_panel.layout().addWidget(allText,3,1)

    for i in subPanels:
        i.layout().setSpacing(0)
        i.layout().setContentsMargins(0,0,0,0)

    save_panel.layout().addWidget(Save_Name)
    save_panel.layout().addWidget(Save_Button)
    
    browse_panel.layout().addWidget(LoadFile_Name)
    browse_panel.layout().addWidget(LoadFile_Button)
    browse_panel.layout().addWidget(UseLoadedRaw_Button)
    
    prp_panel.layout().addWidget(PlotSpectral_Button)
    prp_panel.layout().addWidget(PlotPCA_Button)
    prp_panel.layout().addWidget(PlotEvents_Button)
    prp_panel.layout().addWidget(PlotEpochs_Button)
    
    
    for i in ch_radio_buttons:
        featex_panel.layout().addWidget(i)

    featex_panel.layout().addWidget(BandPower_Button,1)
    featex_panel.layout().addWidget(DWavelets_Button,1)
    featex_panel.layout().addWidget(Hilbert_Button,1)

    clf_panel.layout().addWidget(Classifylog_Button)
    clf_panel.layout().addWidget(ClassifySVM_Button)
    clf_panel.layout().addWidget(ClassifyDNN_Button)

    btn_panel.layout().addWidget(rec_text)
    btn_panel.layout().addWidget(rec_panel)

    

    btn_panel.layout().addWidget(log_panel)
    btn_panel.layout().addWidget(save_panel)
    btn_panel.layout().addWidget(browse_panel)
    btn_panel.layout().addWidget(prep_text)
    btn_panel.layout().addWidget(prp_panel)
    btn_panel.layout().addWidget(fext_text) 
    btn_panel.layout().addWidget(featex_panel)
    btn_panel.layout().addWidget(clf_text)
    btn_panel.layout().addWidget(clf_panel)
    btn_panel.layout().addWidget(typ_text)
    btn_panel.layout().addWidget(Unicorn_Button)
    

    layout= QtWidgets.QHBoxLayout()
    layout.addWidget(btn_panel)
    layout.addWidget(plot_panel)
    layout.setStretchFactor(btn_panel,1)
    layout.setStretchFactor(plot_panel,3)
    
    win.setLayout(layout)
    win.show()
    # .setLayout(layout)
    

    for ch in range(channel_count):
        # main channel
        mch[ch]=plot_panel.addPlot()
        mch[ch].enableAutoRange(x=False, y=True)
        plot_panel.nextRow()

        # fft channel
        # fftch[ch]=plot_panel.addPlot()
        # fftch[ch].enableAutoRange(x=True, y=True)
        # plot_panel.nextRow()

        # filtered channel
        filtch[ch]=plot_panel.addPlot()
        filtch[ch].enableAutoRange(x=False,y=True)
        plot_panel.nextRow()

        # filtered fft channel
        # filtfftch[ch]=plot_panel.addPlot()
        # filtfftch[ch].enableAutoRange(x=True, y=True)
        # plot_panel.nextRow()

        if (info.type() =="EEG"):
            print("adding Data inlet: "+ info.name())
            inlets.append(DataInlet(info, mch[ch],fftch[ch],filtch[ch],bufferText))  #,filtfftch[ch]
            
    tch=plot_panel.addPlot()
    tch.enableAutoRange(x=False,y=True)

    # callbackcc =functools.partial(changeCode,triggerCode,tch,fixationText,fixationText2)
    
    startRecord = functools.partial(changeCode,96,tch,fixationText,fixationText2)
    

    def triggering():
        global triggerCode 
        if (mui.start and not mui.relax):
            triggerCode = 1000
        elif (mui.start and mui.relax):
            triggerCode = 2000
        elif (not mui.start and mui.end):
            triggerCode = 97
            update_timer.stop()
            pull_timer.stop()
            
        else:
            triggerCode = 0
        changeCode(triggerCode,tch,fixationText,fixationText2)
    
    start_time = mui.start_time 
    start_timer = QtCore.QTimer()
    focus_timer = QtCore.QTimer()

    update_timer = QtCore.QTimer()
    pull_timer = QtCore.QTimer()

    focus_interval = 250   # change Epochs length
    focus_timer.timeout.connect(triggering)

    start_timer.timeout.connect(startRecord)
    start_timer.setSingleShot(True)

    def start_experiment():
        marker_ui.show()
        print("Call Marker UI")
        marker_ui.start_exp(max_epoch)
        
    Start_Button.clicked.connect(lambda: start_timer.start(start_time))
    Start_Button.clicked.connect(lambda: start_experiment())
    Start_Button.clicked.connect(lambda: focus_timer.start(focus_interval))
    # Start_Button.clicked.connect(lambda: att_timer.start(att_interval))
    
    Stop_Button.clicked.connect(lambda: changeCode(97,tch,fixationText,fixationText2))
    Stop_Button.clicked.connect(lambda: marker_ui.stop_exp())
    Stop_Button.clicked.connect(lambda: focus_timer.stop())
    Stop_Button.clicked.connect(lambda: update_timer.stop())
    Stop_Button.clicked.connect(lambda: pull_timer.stop())
    # Connect_Button.clicked.connect(lambda: print("Connecting To Unreal TCP Socket"))
    # Reset_Button.clicked.connect(lambda: reset(inlets,ch))
    Reset_Button.clicked.connect(lambda: reset(inlets))

    PlotRaw_Button.clicked.connect(lambda: DataInlet.convertToRaw(inlets,unicorn))
    PlotEvents_Button.clicked.connect(lambda: DataInlet.find_events(inlets,unicorn))
    PlotEpochs_Button.clicked.connect(lambda: DataInlet.epoch_plot(inlets))
    PlotSpectral_Button.clicked.connect(lambda: DataInlet.spect_plot(inlets))
    PlotPCA_Button.clicked.connect(lambda: DataInlet.plot_pca_tsne(inlets))
    # Atar_Button.clicked.connect(lambda: DataInlet.atar_eyeblink(inlets))

    BandPower_Button.clicked.connect(lambda: DataInlet.filter_bank(inlets))
    DWavelets_Button.clicked.connect(lambda: DataInlet.discrete_wavelet(inlets))
    Hilbert_Button.clicked.connect(lambda: DataInlet.hilbert_plot(inlets))

    Classifylog_Button.clicked.connect(lambda: DataInlet.clf_logreg(inlets))
    ClassifySVM_Button.clicked.connect(lambda: DataInlet.clf_SVM(inlets))
    ClassifyDNN_Button.clicked.connect(lambda: DataInlet.clf_DNN(inlets))
    
    Unicorn_Button.clicked.connect(toggle)

    # SelectCH_Button.clicked.connect()

    
    saveraw = functools.partial(DataInlet.save_exp)
    Save_Name.textChanged.connect(changeName)
    saveName = Save_Name.text()
    Save_Button.clicked.connect(saveraw)
    
    # plt.ion()
    
    def browsefiles():
        global fn
        directory = os.getcwd()
        fname= QtWidgets.QFileDialog.getOpenFileName(win,"Open file",directory)
        fn=fname[0]
        # print(fn)
        LoadFile_Name.setText(fn)

    LoadFile_Button.clicked.connect(browsefiles)
    UseLoadedRaw_Button.clicked.connect(lambda: DataInlet.loadraw(inlets,fn))

    def reset(inlets):
        global markerSignalArray,rawSignalArray,fftSignalArray,signalDataCount,inputSignalCounter,record,att_count,rlx_count
        markerSignalArray=[]
        rawSignalArray=[[]for _ in range(channel_count)]
        fftSignalArray=[[]for _ in range(channel_count)]
        inputSignalCounter=0
        signalDataCount=0
        record=False
        att_count=0
        rlx_count=0
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
            inlet.pull_and_plot(mintime, ch,tch,DB_text,DB_markertext,fixationText,fixationText2,dataAmountText,unicorn)

    # def update_fft():
    #     for ch,inlet in enumerate(inlets):
    #         inlet.plot_fft(ch,fftAt,allText)
    
    # create a timer that will move the view every update_interval ms
    
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    # fft_timer = QtCore.QTimer()
    # fft_timer.timeout.connect(update_fft)
    # fft_timer.start(fft_interval)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # app = QtWidgets.QApplication(sys.argv) 
        # app = QtGui.QGuiApplication(sys.argv)
        app.exec()

    return 0


if __name__ == '__main__':
    # app = QtWidgets.QApplication(sys.argv) 
    main()