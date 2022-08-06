#!/usr/bin/env python3
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
import cv2
from tensorflow.keras.models import load_model


def main():
    # create environment with a map
    model=load_model('model1.h5')
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

        # let's try PD control!
        k_p = 10
        k_d = 1

        car_speed = 0.1  # fix speed, forward direction
        im = Image.fromarray(env.render_obs())
        im=np.asarray(im)
        im=np.array([im])
        # decide on steering using PD control
        steering_angle=float(model.predict(im))
        # You should use your control skills here!
        # Try adaptive PID or model-predictive control, for example!
        # And/or you can use your reinforcement learning skills!

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

if __name__  == '__main__':
    main()
