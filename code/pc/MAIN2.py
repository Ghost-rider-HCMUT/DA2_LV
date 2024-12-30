from PyQt5 import QtWidgets,uic
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import QTimer, QTime, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
import cv2
from time import sleep
import numpy as np
import array as arr
import math
import sys
import serial
import os
from imutils.video import VideoStream
from imutils import face_utils
from PIL import Image
import imutils
import pytesseract
import pygame
pygame.mixer.init()
import time
from datetime import datetime
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
#ser = serial.Serial('/dev/ttyUSB0',9600)

ser = serial.Serial('/dev/tty.usbserial-14310',baudrate = 9600, timeout = 1)
print(ser.name)
vs = VideoStream(src=0).start() #cam vào
vs2 = VideoStream(src=1).start() #cam ra
mode = 0; dk_vao = 0; dk_ra = 0; so_xe = 0; con_lai = 0; kq = 0; cp = 0
mode_ht = 0; tt_vao = 0; tt_ra = 0; vr = 0; dc_ra = 0;
time2 = 0; time_vao = 0; time_ra = 0; tien = 0; tgian_vao = ""; tgian_ra = ""; indextg = 0
img = None
mang_tg_vao = [""]*10; mang_time_vao = [0.0]*10; mang_bien = [""]*10;
data = ''; vr = 0
def Time():
    global mode, dk_vao, dk_ra, so_xe, con_lai, kq, cp, time2, mode_ht, vr, tt_vao, tt_ra, dc_ra
    global vs, img
    global time_vao, time_ra, tien, indextg, mang_tg_vao, mang_time_vao, mang_bien, data, vr
    #-------------------thời gian-------------------
    today = datetime.today()
    day = today.strftime("%d/%m/%Y")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    time_present = current_time + " - " + day
    call.lb_tgian.setText(time_present)
    #-----------------nhận dữ liệu------------------
    if(ser.in_waiting > 0):
        s = ser.readline()          #Doc vao data
        dataa = s.decode()           # decode s
        dataa = dataa.rstrip()        # cut "\r\n" at last of string
        #print(dataa)                     #In ra man hinh       
    try:
        mang = dataa.split("|")
        mode_ht = int(mang[0])
        tt_vao = int(mang[1])
        tt_ra = int(mang[2])
        cp = int(mang[3])
        vr = int(mang[4])
        dc_ra = int(mang[5])
        so_xe = int(mang[6])
        #print("lll")
    except:
        a = 0
    con_lai = 3 - so_xe
    call.lb_so_xe.setText(str(so_xe))
    call.lb_con_lai.setText(str(con_lai))
    
    if mode_ht == 0:
        call.lb_mode.setStyleSheet("background-color: rgb(0, 85, 255)")
    if mode_ht == 1:
        call.lb_mode.setStyleSheet("background-color: rgb(255, 170, 0)") 
    if tt_vao == 0:
        #str_vao = "VÀO: ĐANG ĐÓNG"
        #call.BT_VAO.setText(str_vao)
        call.BT_VAO.setStyleSheet("background-color: rgb(255, 0, 0)") 
    else:
        #str_vao = "VÀO: ĐANG MỞ"
        #call.BT_VAO.setText(str_vao)
        call.BT_VAO.setStyleSheet("background-color: rgb(0, 255, 0)") 
    if tt_ra == 0:
        #str_ra = "RA: ĐANG ĐÓNG"
        #call.BT_RA.setText(str_ra)
        call.BT_RA.setStyleSheet("background-color: rgb(255, 0, 0)") 
    else:
        #str_ra = "RA: ĐANG MỞ"
        #call.BT_RA.setText(str_ra)
        call.BT_RA.setStyleSheet("background-color: rgb(0, 255, 0)") 
    if so_xe < 3:
        '''
        call.lb_so_xe.setStyleSheet("background-color: rgb(85, 255, 0)")
        call.lb_con_lai.setStyleSheet("background-color: rgb(85, 255, 0)")
        '''
        call.lb_tb.setText("CÒN CHỖ TRỐNG")
        call.lb_tb.setStyleSheet("background-color: rgb(85, 255, 0)")
    else:
        '''
        call.lb_so_xe.setStyleSheet("background-color: rgb(255, 0, 0)")
        call.lb_con_lai.setStyleSheet("background-color: rgb(255, 0, 0)")
        '''
        call.lb_tb.setText("ĐÃ HẾT CHỖ")
        call.lb_tb.setStyleSheet("background-color: rgb(255, 0, 0)")
    try:
        # Đọc ảnh từ camera
        frame = vs.read()
        frame2 = vs2.read()
        #frame = cv2.flip(frame,1)
        # chỉnh kích thước ảnh để tăng tốc độ xử lý
        frame = imutils.resize(frame, width=800)
        frame2 = imutils.resize(frame2, width=800)
        # Hien thi len man hinh
        imgg = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(imgg)
        call.lb_vid.setPixmap(pix)

        imgg2 = QImage(frame2, frame2.shape[1], frame2.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix2 = QPixmap.fromImage(imgg2)
        call.lb_vid2.setPixmap(pix2)
        if cp == 1:
            if vr == 1:
                if so_xe < 3:
                    img = frame
                    xulyanh()
                    pix2 = QPixmap('bien.jpg')
                    call.lb_bien.setPixmap(pix2)
                    time1 = time.time()
                    if time1 - time2 >= 7:
                        pygame.mixer.music.load("SOUND\WELCOME.MP3")
                        pygame.mixer.music.play()
                        time2 = time.time()
                    mang_tg_vao[indextg] = time_present
                    mang_time_vao[indextg] = time.time()
                    mang_bien[indextg] = data
                    call.lb_tgian_vao.setText(mang_tg_vao[indextg])
                    #call.lb_tgian_vao.setText(str(mang_time_vao[indextg]))
                    indextg = indextg + 1
                    if indextg >= 3:
                        indextg = 0
                    
                else:
                    print("day xe")
                    time1 = time.time()
                    if time1 - time2 >= 7:
                        pygame.mixer.music.load("SOUND\FULL.MP3")
                        pygame.mixer.music.play()
                        time2 = time.time()
            if vr == 2:
                img = frame2
                xulyanh()
                pix2 = QPixmap('bien.jpg')
                call.lb_bien2.setPixmap(pix2)
            cp = 0
        if dc_ra == 1:
            vitri = 0
            for ii in range(0,3):          
                if mang_bien[ii] == data:
                    vitri = ii
                    print("ii = ",ii)
            time_vao = mang_time_vao[vitri]
            call.lb_tgian_ra.setText(time_present)
            call.lb_tgian_vao.setText(mang_tg_vao[vitri])
            time_ra = time.time()
            tgian_gui = round(time_ra - time_vao)
            tgian_gui = int(tgian_gui)
            if tgian_gui <= 30:
                tien = "5.000 VND"
            elif tgian_gui <= 40:
                tien = "10.000 VND"
            else:
                tien = "15.000 VND"
            call.lb_tgian_gui.setText(str(tgian_gui) + " s")
            call.lb_tien.setText(tien)
            time1 = time.time()
            if time1 - time2 >= 3:
                pygame.mixer.music.load("SOUND\BYE.MP3")
                pygame.mixer.music.play()
                time2 = time.time()
                dc_ra = 0
        if dc_ra == 2:
            time1 = time.time()
            if time1 - time2 >= 7:
                pygame.mixer.music.load("SOUND\ERROR.MP3")
                pygame.mixer.music.play()
                time2 = time.time()
                dc_ra = 0
    except:
        a = 0
#------------------------------------------------------------------#
def thoat():
    ser.close()
    call.close()
    exit(app.exec())
#------------------------------------------------------------------#
def auto():
    ser.write(b'a')
    ser.write(b'|')
    ser.write(b'0')
    ser.write(b'\r\n')
    ser.flush()
#------------------------------------------------------------------#
def manual():
    ser.write(b'a')
    ser.write(b'|')
    ser.write(b'1')
    ser.write(b'\r\n')
    ser.flush()
#------------------------------------------------------------------#
def vao():
    global dk_vao
    dk_vao = dk_vao + 1
    if dk_vao >= 2:
        dk_vao = 0
    ser.write(b'b')
    ser.write(b'|')
    ser.write(str(dk_vao).encode())
    ser.write(b'\r\n')
    ser.flush()
#------------------------------------------------------------------#
def ra():
    global dk_ra
    dk_ra = dk_ra + 1
    if dk_ra >= 2:
        dk_ra = 0
    ser.write(b'c')
    ser.write(b'|')
    ser.write(str(dk_ra).encode())
    ser.write(b'\r\n')
    ser.flush()
#------------------------------------------------------------------#
def xulyanh():
    global img, data, vr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    contours,h = cv2.findContours(thresh,1,2)
    largest_rectangle = [0,0]
    for cnt in contours:
        lenght = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, lenght, True)
        if len(approx)==4: 
            area = cv2.contourArea(cnt)
            if area > largest_rectangle[0]:
                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
    x,y,w,h = cv2.boundingRect(largest_rectangle[1])

    image=img[y:y+h, x:x+w]
    cv2.drawContours(img,[largest_rectangle[1]],0,(0,255,0),2)

    cropped = img[y:y+h, x:x+w]
    cv2.imshow('Dinh vi bien so xe', img)
    
    cv2.drawContours(img,[largest_rectangle[1]],0,(255,255,255),5)
    #DOC HINH ANH CHUYEN THANH FILE TEXT
    pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #cv2.imshow('Crop', thresh)
    cv2.imwrite("bien.jpg",thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')

    print("Bien so xe la:")
    #data = data.decode()           # decode s
    data = str(data)
    data = data.rstrip()        # cut "\r\n" at last of string
    #data = data.replace('\r', '*')
    data = data.replace('\n', '*')
    print(data)
    if vr == 1:
        call.lb_bien_vao.setText(data)
    if vr == 2:
        call.lb_bien_ra.setText(data)
    ser.write(b'd')
    ser.write(b'|')
    ser.write(data.encode())
    ser.write(b'\r\n')
    ser.flush()

#------------------------------------------------------------------#

#------------------------------------------------------------------#
while True:
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app=QtWidgets.QApplication([])
    call=uic.loadUi("GUID2.ui")
    call.QUIT.clicked.connect(thoat)
    call.BT_AUTO.clicked.connect(auto)
    call.BT_MAN.clicked.connect(manual)
    call.BT_VAO.clicked.connect(vao)
    call.BT_RA.clicked.connect(ra)
    call.timer = QTimer()
    call.timer.timeout.connect(Time)
    call.timer.start(50)
    index1 = 1
    call.show()
    app.exec()
