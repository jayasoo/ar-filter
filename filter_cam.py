#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:22:58 2019

@author: jayasoo
"""

import pickle
import numpy as np
import cv2
from stream import Stream

from train import getModel

def start_cam():
    model = getModel()
    model.load_weights('./weights.hdf5')
    
    with open('y_scaler.pickle', 'rb') as handle:
        y_scaler = pickle.load(handle)
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    stream = Stream(0).start()
    
    while True:
        if (cv2.waitKey(1) & 0xFF == ord("e")) or stream.stopped:
            stream.stop()
            break
        
        img = stream.frame
        img_copy = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            face = gray[y:y+h,x:x+w]
            face_clr = img[y:y+h,x:x+w]
            
            face_input = cv2.resize(face, (96,96))
            face_input = np.reshape(face_input, (1,96,96,1)) 
            face_input = face_input / 255.
            
            # Predict facial keypoints
            keypoints = model.predict(face_input)
            keypoints = y_scaler.inverse_transform(keypoints)[0]
            
            # Scale the predictions
            keypoints[::2] = w * (keypoints[::2]/96)
            keypoints[1::2] = h * (keypoints[1::2]/96)
            
            for i in range(0,30,2):
                cv2.circle(face_clr, (keypoints[i], keypoints[i+1]), 2, (255,255,0), -1)
            
            nose_tip_coords = (int(keypoints[20]), int(keypoints[21]))
            left_eye_coords = (int(keypoints[6]), int(keypoints[7]))
            right_eye_coords = (int(keypoints[10]), int(keypoints[11]))
            brow_coords = (int(keypoints[14]), int(keypoints[15]))
            
            # Calculate width and height of filter
            shade_width = int((left_eye_coords[0] - right_eye_coords[0]) * 1.5)
            shade_height = int((left_eye_coords[1] - brow_coords[1]) * 4.0)
            
            shade = cv2.imread("./filters/shade.png", -1)
            shade = cv2.resize(shade, (shade_width, shade_height))
            f_h, f_w, _ = shade.shape
            
            for i in range(f_h):
                for j in range(f_w):
                    if shade[i,j][3] != 0:
                        img_copy[y+brow_coords[1]-int(0.2*f_h)+i, x+nose_tip_coords[0]-int(0.5*f_w)+j] = shade[i,j][0:3]
            
            
        cv2.imshow('landmark',img)
        cv2.imshow('filter', img_copy)
        
if __name__ == "__main__":
    start_cam()