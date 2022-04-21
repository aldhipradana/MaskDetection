import cv2
from numpy.core.records import recarray
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
from playsound import playsound

from mtcnn import MTCNN


import tkinter as tk
from tkinter import *
from tkmacosx import Button
from tkinter import filedialog


from keras.preprocessing import image
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model =load_model('model8020_9600.h5')

img_width , img_height = 150,150

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_count_full = 0

font = cv2.FONT_HERSHEY_SIMPLEX
org = (1,1)
class_label = ''
fontScale = 1
color = (255,0,0)
thickness = 2


def haarCam():
    i = 1
    haarApps(i)

def haarVid():
    i = 2
    haarApps(i)

def mtcnnCam():
    i = 1
    mtcnnApps(i)

def mtcnnVid():
    i = 2
    mtcnnApps(i)


def haarApps(x):
    x = x

    if x==1:
        cap = cv2.VideoCapture(1)

        
    elif x==2:
        cwd = os.getcwd()
        filenames = filedialog.askopenfilename(initialdir=cwd ,title="Select Videos" )
        # cap = os.listdir(filenames)
        cap = cv2.VideoCapture(filenames)

        print(cap)

    
    # FPS
    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    global img_count_full
    while True:
        img_count_full += 1
        response , color_img = cap.read()
        

        if response == False:
            break

        gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)


        # Calculating the fps -------------------------------------------------------------------------------------------
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(color_img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        # End Calculating the fps -------------------------------------------------------------------------------------------

        img_count = 0
        for (x,y,w,h) in faces:
            org = (x-10,y-10)
            img_count += 1
            # color_face = color_img[y:y+h,x:x+w]
            color_face = color_img[y:y+h,x:x+w]
            cv2.imwrite('input/%d%dface.jpg'%(img_count_full,img_count),color_face)
            img = load_img('input/%d%dface.jpg'%(img_count_full,img_count),target_size=(img_width,img_height))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            prediction = model.predict(img)


            if prediction==0:
                class_label = "Mask"
                color = (0,255,0)

            else:
                class_label = "No Mask"
                color = (0,0,255)
                # playsound('beep.mp3')
                # print("No Maskk!")


            # cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.rectangle(color_img,(x,y),(x+w,y+h),color,3)

            # cv2.putText(roi, class_label, org, font ,fontScale, color, thickness,cv2.LINE_AA)
            cv2.putText(color_img, class_label, org, font ,fontScale, color, thickness,cv2.LINE_AA)


        # cv2.imshow('ROI', roi)
        # cv2.moveWindow('ROI', 700,0)
        cv2.imshow('Face mask detection', color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    


def mtcnnApps(x):
    x = x
    
    if x==1:
        cap = cv2.VideoCapture(1)

        
    elif x==2:
        cwd = os.getcwd()
        filenames = filedialog.askopenfilename(initialdir=cwd ,title="Select Videos" )
        # cap = os.listdir(filenames)
        cap = cv2.VideoCapture(filenames)
        print(filenames)
    

    global img_count_full

    detector = MTCNN()

    # FPS
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0


    while True:
        img_count_full += 1

        ret,frame = cap.read()

        # scale = 50
        # my_width = int(frame.shape[1]*scale/100)
        
        # my_height = int(frame.shape[0]*scale/100)
        # dim = (my_width, my_height)

        # frame = cv2.resize(frame, dim ,interpolation= cv2.INTER_AREA)

        # Calculating the fps -------------------------------------------------------------------------------------------
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        # End Calculating the fps -------------------------------------------------------------------------------------------



        output = detector.detect_faces(frame)

        img_count = 0
        for single_output in output:

            # Get Box
            x,y,w,h = single_output['box']

            # Detecting
            org = (x-10,y-10)
            color_face = frame[y:y+h,x:x+w]
            cv2.imwrite('input/%d%dface.jpg'%(img_count_full,img_count),color_face)
            img = load_img('input/%d%dface.jpg'%(img_count_full,img_count),target_size=(img_width,img_height))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            prediction = model.predict(img)


            if prediction==0:
                class_label = "Mask"
                color = (0,255,0)
                # print(" Maskk!")


            else:
                class_label = "No Mask"
                color = (0,0,255)
                # playsound('beep.mp3')
                # print("No Maskk!")


            #  Rectangle
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,thickness)
            cv2.putText(frame, class_label, org, font ,fontScale, color, thickness,cv2.LINE_AA)

        cv2.imshow('win',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
    

root = Tk()

haarLabel = Label(root, text="Haar Cascade").grid(row=0, column=2, padx=10, pady=10)
mtcnnLabel = Label(root, text="MTCNN").grid(row=0, column=4, padx=10, pady=10)

camHaar = Button(root, text="Camera", padx=10, pady=10, foreground="#ffffff", background="#353b48", borderless=1,
            activebackground="#ffffff", activeforeground='#353b48', overbackground="#353b48", command=haarCam)
camHaar.grid(row=1, column=2)

vidHaar = Button(root, text="Video", padx=10, pady=10, foreground="#ffffff", background="#353b48", borderless=1,
            activebackground="#ffffff", activeforeground='#353b48', overbackground="#353b48", command=haarVid)
vidHaar.grid(row=2, column=2)

camMtcnn = Button(root, text="Camera", padx=10, pady=10, foreground="#ffffff", background="#353b48", borderless=1,
            activebackground="#ffffff", activeforeground='#353b48', overbackground="#353b48", command=mtcnnCam)
camMtcnn.grid(row=1, column=4)

vidMtcnn = Button(root, text="Video", padx=10, pady=10, foreground="#ffffff", background="#353b48", borderless=1,
            activebackground="#ffffff", activeforeground='#353b48', overbackground="#353b48", command=mtcnnVid)
vidMtcnn.grid(row=2, column=4)

root.mainloop()

