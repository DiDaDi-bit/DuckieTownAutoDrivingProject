#!/usr/bin/env python3
# import relevant libraries
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
from keras.layers import Dropout
from keras import regularizers
#from tensorflow.keras.layers import MaxPooling
from tensorflow.keras.models import load_model
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import xlsxwriter
import cv2
from tensorflow.keras.models import load_model


def main():
    # create environment with a map
    model=load_model('final.h5')
    env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)
    env.reset()
    env.render()
    # define total reward
    total_reward = 0
    # main loop
    while True:
        # getting our lane position
        # in practice these should be provided by the camera/vision system!
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad
        
        im = Image.fromarray(env.render_obs())
        im=np.asarray(im)
        #im=np.array([im])
        #print(im.shape)
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        canny = do_canny(im_gray)
        segment = do_segment(canny)
        #print('segment',segment.shape)
        hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 50, maxLineGap = 60)
        lines_visualize = visualize_lines(segment, hough)
        lines_visualize=lines_visualize.reshape(480,640,1)
        #print('segment',segment.shape)
        lines_visualize=np.asarray(lines_visualize)
        lines_visualize=np.array([lines_visualize])
        steering_angle=float(model.predict(lines_visualize))
        if((steering_angle>=0.5)|(steering_angle<=-0.5)):
            car_speed=0.05
        else:
            car_speed=0.1
        obs, reward, done, info = env.step([car_speed, steering_angle])

        # update total reward
        total_reward += reward

        print(
            "step=%s, current reward=%.3f, total reward=%.3f"
            % (env.step_count, reward, total_reward)
        )
        print("car speed is",car_speed ,"steering angle is",steering_angle)
        # let the simulator update its display
        env.render()

        if env.step_count<=1500:
            im = Image.fromarray(env.render_obs())

        if done:
            if reward < 0:
                print("*** CRASHED ***")
            print("Exiting")
            print(info)
            break    


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

if __name__  == '__main__':
    main()
