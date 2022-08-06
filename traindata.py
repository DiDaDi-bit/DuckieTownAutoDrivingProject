print('Start')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pandas as pd
from sklearn.model_selection import train_test_split
#from utlis import *
from pickle import FALSE, TRUE
from PIL import Image
import time
import cv2 as cv
import tensorflow as tf
import sys
from datetime import datetime
import gym
import numpy as np
import pyglet
import matplotlib.pyplot as plt
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import xlsxwriter
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling


def main():
    path='C:\\Users\\Newton\\OneDrive\\桌面\\ELEN 90088 Machine Learning\\workshops\\LDI project\\Autodriving\\AD_kit-Canvas_v1\\AD_kit-Canvas_v1\\gym-duckietown-kit\\starterkit\\csvdata\\'
    path2='datacollect\\'
    data=importdatainfo(path)
    print(data.head)
    imagespath,steerings=loaddata(path2,data)
    xtrain,xval,ytrain,yval=train_test_split(imagespath,steerings,test_size=0.2,random_state=10)
    print('total training images: ',len(xtrain))
    print('total validation images: ',len(xval))
    print(imagespath[0])

    im = cv.imread(imagespath[0])
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    canny = do_canny(im_gray)
    segment = do_segment(canny)
    #use hough to train in the first
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 50, maxLineGap = 60)
    #use line to train in the second
    lines_visualize = visualize_lines(segment, hough)
    print('hough',hough.shape)
    print('segment',segment.shape)
    print('im',im.shape)
    model=createModel()
    history=model.fit(dataGen(xtrain,ytrain,100),steps_per_epoch=100,epochs=10,validation_data=dataGen(xval,yval,50),validation_steps=50)
    model.save('model.h5')
    print('Model Saved')

    #plt.plot(history.history['loss'])
    #plt.plot(history,history['val_loss'])
    #plt.show()

def dataGen(imagesPath,steeringList,batchSize):
    while True:
        imgBatch=[]
        steeringBatch=[]

        for i in range(batchSize):
            index=np.random.randint(0,len(imagesPath)-1)
            im=cv.imread(imagesPath[index])
            steering=steeringList[index]
            im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            canny = do_canny(im_gray)
            segment = do_segment(canny)
            hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 50, maxLineGap = 60)
            imgBatch.append(im)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))


def createModel():
    model =Sequential()
    #卷积——》池化——》卷积——》池化——》全连接
    model.add(Conv2D(24,(5,5),(2,2),input_shape=(480,640,3),activation='relu'))
    model.add(Conv2D(36,(5,5),(2,2),activation='relu'))
    #model.add(MaxPooling(pool_size=(2,2)))
    model.add(Conv2D(48,(5,5),(2,2),activation='relu'))
    #model.add(MaxPooling(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    #model.add(MaxPooling(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),loss='mse')
    return model


def do_canny(frame):
    blur = cv.GaussianBlur(frame, (5, 5), 0)
    return cv.Canny(blur, 50, 150)

def do_segment(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    mask = np.zeros_like(frame)
    mask_points = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (int(width/2), int(height/3)),
        (0, int(height*0.6))
    ]])
    cv.fillPoly(mask, mask_points, 255)
    return cv.bitwise_and(frame, mask)

def visualize_lines(frame, lines_arr):
    line_frame = np.zeros_like(frame)
    color = (255, 255, 255)
    thickness = 5

    if lines_arr is not None:
        for line in lines_arr:
            for x1, y1, x2, y2 in line:
                cv.line(line_frame, (x1, y1), (x2, y2), color, thickness)
    return line_frame

def importdatainfo(path):
    cols=['center','steering']
    data=pd.DataFrame()
    data=pd.read_csv(os.path.join(path,f'dataset1.csv'))
    return data

def loaddata(path,data):
    imagespath=[]
    steering=[]
    for i in range(len(data)):
        indexed_data=data.iloc[i]
        imagespath.append(os.path.join(path,indexed_data[0]))
        steering.append(float(indexed_data[1]))
    imagespath=np.asarray(imagespath)
    steering=np.asarray(steering)
    return imagespath,steering

if __name__  == '__main__':
    main()