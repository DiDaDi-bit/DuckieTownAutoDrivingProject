print('Start')
import os
from random import sample, shuffle
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
#from tensorflow.keras.layers import MaxPooling


def main():
    #csv file directory
    path='C:\\Users\\Newton\\OneDrive\\桌面\\ELEN 90088 Machine Learning\\workshops\\LDI project\\Autodriving\\AD_kit-Canvas_v1\\AD_kit-Canvas_v1\\gym-duckietown-kit\\starterkit\\csvdata\\'
    #image file directory
    path2='datacollectfinal\\'
    data=importdatainfo(path)
    print(data.head)
    data=balancedata(data)
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
    print('lines_visualize',lines_visualize.shape)
    print('im',im.shape)
    model=createModel()
    history=model.fit(dataGen(xtrain,ytrain,100),
                                steps_per_epoch=100,epochs=20,validation_data=dataGen(xval,yval,50),validation_steps=50)
    model.save('test111.h5')
    print('Model Saved')

    #plt.plot(history.history['loss'])
    #plt.plot(history,history['val_loss'])
    #plt.legend(['Training','Validation'])
    #plt.title('Loss')
    #plt.xlabel('Epoch')
    #plt.show()

##visualize and balance data
def balancedata(data,display=True):
    nbin=31
    samplesPerbin=200
    hist,bins=np.histogram(data['steering_angle'],nbin)
    if display:
        center=(bins[:-1]+bins[1:])*0.5
        plt.bar(center,hist,width=0.03)
        plt.plot((np.min(data['steering_angle']),np.max(data['steering_angle'])),(samplesPerbin,samplesPerbin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('Number of samples')
        plt.show()
    removeindexList=[]
    for j in range(nbin):
        binDataList=[]
        for i in range (len(data['steering_angle'])):
            if data['steering_angle'][i]>=bins[j] and data['steering_angle'][i]<=bins[j+1]:
                binDataList.append(i)
        shuffle(binDataList)
        binDataList=binDataList[samplesPerbin:]
        removeindexList.extend(binDataList)

    print('Removed Images:',len(removeindexList))
    data.drop(data.index[removeindexList],inplace=True)
    print('Remaining Images:',len(data))
    if display:
        hist,_=np.histogram(data['steering_angle'],(nbin))
        plt.bar(center,hist,width=0.03)
        plt.plot((np.min(data['steering_angle']),np.max(data['steering_angle'])),(samplesPerbin,samplesPerbin))
        plt.title('Balanced data')
        plt.xlabel('Steering angle')
        plt.ylabel('Number of samples')
        plt.show()
    return data

def augmentImage(imgpath,steering):
    img=cv.imread(imgpath)
    if np.random.rand() <0.5:
        img=cv.flip(img,1)
        steering=-steering
    #if np.random.rand() <0.5:
        #zoom=cv.Affine(scale=(1,1.2))
        #img=zoom.augment_image(img)
    return img,steering


def dataGen(imagesPath,steeringList,batchSize):
    while True:
        imgBatch=[]
        steeringBatch=[]

        for i in range(batchSize):
            index=np.random.randint(0,len(imagesPath)-1)
            im=cv.imread(imagesPath[index])
            steering=steeringList[index]
            ##augmentImage
            if np.random.rand() <0.5:
                 im=cv.flip(im,1)
                 steering=-steering

            im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            canny = do_canny(im_gray)
            segment = do_segment(canny)
            hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 50, maxLineGap = 60)
            lines_visualize = visualize_lines(segment, hough)
            imgBatch.append(lines_visualize)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asarray(steeringBatch))



def createModel():
    model =Sequential()
    #卷积——》池化——》卷积——》池化——》全连接
    model.add(Conv2D(24,(5,5),(2,2),input_shape=(480,640,1),activation='relu'))
    model.add(Conv2D(36,(5,5),(2,2),activation='relu'))
    #model.add(MaxPooling(pool_size=(2,2)))
    model.add(Conv2D(48,(5,5),(2,2),activation='relu'))
    #model.add(MaxPooling(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    #model.add(MaxPooling(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(50,activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(10,activation='relu',kernel_regularizer=regularizers.l2(0.1)))
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
    data=pd.read_csv(os.path.join(path,f'datasetfinal.csv'))
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