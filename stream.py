#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:29:53 2019

@author: jayasoo
"""

import cv2
from threading import Thread

class Stream:
    def __init__(self, src):
        self.src = src
        self.stream = cv2.VideoCapture(self.src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.get, args=()).start()
        return self
        
    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
            
    def stop(self):
        self.stopped = True