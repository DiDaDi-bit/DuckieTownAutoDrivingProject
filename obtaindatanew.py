#!/usr/bin/env python3

# import relevant libraries
from pickle import FALSE, TRUE
from PIL import Image
import time
import cv2
from datetime import datetime
import gym
import numpy as np
import pyglet
import pandas
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import xlsxwriter

def main():
    workbook=xlsxwriter.Workbook('datasetfinal2.csv')
    worksheet=workbook.add_worksheet()
    # create environment with a map
    env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)
    env.reset()
    env.render()
    # define total reward
    total_reward = 0
    worksheet.write('A1','imag')
    worksheet.write('B1','steering_angle')    
    row=2
    PATH="C:\\Users\\Newton\\OneDrive\\桌面\\ELEN 90088 Machine Learning\\workshops\\LDI project\\Autodriving\\AD_kit-Canvas_v1\\AD_kit-Canvas_v1\\gym-duckietown-kit\\starterkit\\datacollectfinal2\\"
    # main loop
    while True:
        # getting our lane position
        # in practice these should be provided by the camera/vision system!
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad

        # let's try PD control!
        k_p = 10
        k_d = 1

        car_speed = 0.1  # fix speed, forward direction

        # decide on steering using PD control
        steering_angle = (
            k_p * distance_to_road_center + k_d * angle_from_straight_in_rads
        )
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
        if env.step_count<=3000:
            im = Image.fromarray(env.render_obs())
            if np.random.rand() <0.5:
                 im=cv2.flip(im,1)
                 steering_angle=-steering_angle
                 print('fliped')
            im.save(PATH+str(env.step_count)+"screen.png")
        worksheet.write('A'+str(row),str(env.step_count)+"screen.png")
        worksheet.write('B'+str(row),steering_angle)
        row+=1

        if done:
            if reward < 0:
                print("*** CRASHED ***")
            print("Exiting")
            print(info)
            break
    workbook.close()        

if __name__  == '__main__':
    main()
