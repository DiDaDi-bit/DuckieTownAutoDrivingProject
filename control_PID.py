#!/usr/bin/env python3

"""
Contents:
* Creates a duckietown environment
* Shows how ground truth data can be sampled from the simulator
* Gives PID control applied to sim data
"""

# import relevant libraries
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import xlsxwriter



def main():
    #create excel to record data
    workbook=xlsxwriter.Workbook('PID_data.xlsx')
    worksheet=workbook.add_worksheet()
    # create environment with a map
    env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)
    env.reset()
    env.render()
    # define total reward
    total_reward = 0
    row = 2
    worksheet.write('A1','step')
    worksheet.write('B1','reward')
    worksheet.write('C1','total_reward')
    worksheet.write('D1','car_speed')
    worksheet.write('E1','steering_angle')
    worksheet.write('F1','distance')
    worksheet.write('G1','angle') 
    worksheet.write('H1','acc') 
    acc = 0
    #accLimit = 10
    # main loop
    while True:
        # getting our lane position
        # in practice these should be provided by the camera/vision system!
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad

        # let's try PID control!
        k_p = 10
        k_i = 0.1
        k_d = 5
        acc += distance_to_road_center
        car_speed = 0.1  # fix speed, forward direction

        # decide on steering using PID control
        steering_angle = (
            k_p * distance_to_road_center  + k_i*acc + k_d * angle_from_straight_in_rads
        )
        
        obs, reward, done, info = env.step([car_speed, steering_angle])

        # update total reward
        total_reward += reward
        #record data into excel
        worksheet.write('A'+str(row),env.step_count)
        worksheet.write('B'+str(row),reward)
        worksheet.write('C'+str(row),total_reward)
        worksheet.write('D'+str(row),car_speed)
        worksheet.write('E'+str(row),steering_angle)
        worksheet.write('F'+str(row),distance_to_road_center)
        worksheet.write('G'+str(row),angle_from_straight_in_rads)
        worksheet.write('H'+str(row),acc)
        row+=1

        # let the simulator update its display
        env.render()

        if done:
            if reward < 0:
                print("*** CRASHED ***")
            print("Exiting")
            print(info)
            break
    workbook.close()        

if __name__  == '__main__':
    main()
