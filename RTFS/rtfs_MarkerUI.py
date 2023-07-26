import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter,QFont, QBrush,QPen
from PyQt5.QtCore import Qt ,QPoint

import sys
from win32api import GetSystemMetrics

global focus,start,max_epoch,end,relax_interval,start_time,relax
max_x = GetSystemMetrics(0)
max_y = GetSystemMetrics(1)
paint_timer = QtCore.QTimer()
draw_timer = QtCore.QTimer()
start_timer = QtCore.QTimer()
relax_timer = QtCore.QTimer()
epoch_timer = QtCore.QTimer()
offset_timer = QtCore.QTimer()
draw_interval = 3000
start_time = 5000
relax_interval = 5000
trigger_code = 0
offset_time = 500

App = QtWidgets.QApplication(sys.argv)

class MarkerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        global focus,start,max_epoch,end,relax_interval,trigger_code,relax
        super().__init__()
        self.setStyleSheet("background-color: white;")

        # set the title
        self.setWindowTitle("Marker Cue")
        # setting  the geometry of window
        self.showFullScreen()
        self.c_circle = 0 
        self.start_offset=0
        self.desktop = QtWidgets.QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        start = False
        focus = False
        end = False
        relax = True
        print(self.width)
        print(self.height)
        # self.setCentralWidget(self.label)
        # self.start_paint()      
        # self.paintEvent()
        # self.draw_something()
        
    def start_exp(self,exp_epoch):
        global max_epoch
        # self.labelPrep.move(max_x/2,max_y/2)
        start_timer.timeout.connect(self.timer_start_exp)
        start_timer.setSingleShot(True)
        start_timer.start(start_time)
        max_epoch = exp_epoch
        
        print("Start Experiment")
    
    def drawFocus(self):
        global focus
        focus= True
        
        self.update()
        draw_timer.timeout.connect(self.updateRelax)
        draw_timer.setSingleShot(True)
        draw_timer.start(draw_interval)
        self.backFocus
        
    def frontFocus(self):
        global relax
        relax = False
        offset_timer.timeout.connect(self.drawFocus)
        offset_timer.setSingleShot(True)
        offset_timer.start(offset_time)
    
    def backFocus(self):

        offset_timer.timeout.connect(self.updateRelax)
        offset_timer.setSingleShot(True)
        offset_timer.start(offset_time)
        
    def updateRelax(self):
        global focus,start,relax_interval,relax
        self.update()
        focus = False
        relax = True
        relax_interval = np.random.randint(5,6) * 1000
        relax_timer.timeout.connect(self.frontFocus)
        relax_timer.setSingleShot(True)
        relax_timer.start(relax_interval)

    def timer_start_exp(self):
        # Trigger Interval
        global start
        print("Start Recording")
        start = True
        self.updateRelax()

    def stop_exp(self):
        print("stop Experiment")
        relax_interval = 5000
        relax_timer.timeout.connect(self.change_stop_val)
        relax_timer.start(relax_interval)

    def change_stop_val(self):
        global end,start
        end = True
        start = False
        print("Stop experiment Call FUnction")

    def paintEvent(self, e):
        global focus,start,end,max_epoch,trigger_code,relax  
        x=np.random.randint(max_x)
        y=np.random.randint(max_y)
        
        qp = QtGui.QPainter(self)
        center = QPoint(x,y)
        font = qp.font()
        font.setPixelSize(64)
        qp.setFont(font)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QPen(Qt.green,0.2,Qt.SolidLine))
        qp.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        # qp.drawLine(10, 10, 300, 200)

        if(not start and not end):
            print("prepare")
            qp.setPen(QPen(Qt.black,0.2,Qt.SolidLine))
            qp.drawText(max_x/2,max_y/2,"Prepare For Focus Experiment")

        if (self.c_circle< max_epoch):
            trigger_code = 100
            if (start and focus):    
                
                # print("Draw circle at X:",x,"  Y:",y)
                print("Attention Drawn: ", self.c_circle +1)
                qp.drawEllipse(center, 74,74)
                self.c_circle=self.c_circle+1
                draw_timer.start(draw_interval)

            elif (start and relax):
                print("Relax Round: ")
                qp.setPen(QPen(Qt.white,0.2,Qt.SolidLine))
                qp.drawText(max_x/2,max_y/2,"")

            # print("Center: ",center)
            # print("Is Focus: ", focus)
        elif (not end and (self.c_circle == max_epoch)):
            print("End Last Focus Epoch")
            self.stop_exp()

        elif (not start and end):
            qp.setPen(QPen(Qt.black,0.2,Qt.SolidLine))
            qp.drawText(max_x/2,max_y/2,"Experiment End")
            print("Experiment End")
            
        # else:
        #     print("countdown: ", self.start_offset)
        #     qp.setPen(QPen(Qt.black,0.2,Qt.SolidLine))
        #     qp.drawText(max_x/2,max_y/2,"Prepare For Focus Experiment")
        #     self.start_offset=self.start_offset+1
        
        qp.end()