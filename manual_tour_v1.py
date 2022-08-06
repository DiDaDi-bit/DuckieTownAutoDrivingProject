#!/usr/bin/env python3

"""
Tansu Alpcan
v1  2022-3-14

An example of manual vehicle control using the keyboard during simulation

Contents:
* Keypress handling example
* Duckietown Action handling example
* Simulator main loop example

Notes:
* This simulator IGNORES HELD KEYS; to speed up, press "up" repeatedly
"""

from pickle import FALSE, TRUE
from PIL import Image
import argparse
import sys
import time

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv


# global variables
simcontinue = True
action_input = np.array([0.0, 0.0])

'''
Information about how Actions are defined:

The simulator uses continuous actions by default.
Actions passed to the step() function should be numpy arrays containining two numbers between -1 and 1.
These two numbers correspond to forward velocity, and a steering angle, respectively.
A positive velocity makes the robot go forward, and a positive steering angle makes the robot turn left.
There is also a Gym wrapper class named DiscreteWrapper
which allows you to use discrete actions (turn left, move forward, turn right)
instead of continuous actions if you prefer.
'''


def key_press(symbol, modifiers):
    """
    This processes keyboard commands that control the simulation.
    Available commands:
     * up, down, left, right arrows - move the vehicle
     * enter/return - save the current view as a screenshot
     * Q - quit the simulator
    Note: this function IGNORES HELD KEYS; to speed up, press "up" repeatedly
    """
    global simcontinue, action_input

    # navigate
    if symbol in [key.UP, key.DOWN, key.LEFT, key.RIGHT]:
        # incrementally change action_input based on inputs
        if symbol == key.UP:
            action_input += np.array([0.1, 0.0])
        elif symbol == key.DOWN:
            action_input += np.array([-0.1, 0.0])
        elif symbol == key.LEFT:
            action_input += np.array([0.0, 1])
        elif symbol == key.RIGHT:
            action_input += np.array([0.0, -1])
        print("New vel - forward: %5.2f, left: %5.2f" % tuple(action_input))

    if symbol == key.RETURN:
        print('saving screenshot')
        # there are two options here:
        # 1. screen image: env.render('rgb_array')
        # 2. camera observation: env.render_obs()
        # Observation data is smaller than the screen image and easier to process
        im = Image.fromarray(env.render_obs())
        im.save("screen.png")

    # exit sim if Q is pressed
    if symbol == key.Q:
        global simcontinue
        simcontinue = False


def main():
    """
    Simulation main loop.
    1. Applies the current action (to move the vehicle)
    2. Updates the display
    3. Pauses briefly to lower CPU usage
    """
    while simcontinue:
        # This is the standard OpenAI gym format for reinforcement learning
        # input --> environment --> (observation, reward, done or not, information)
        obs, reward, done, info = env.step(action_input)
        env.render()
        time.sleep(0.05) # to slow down things a bit, change to your liking


if __name__ == "__main__":
    print("Initializing environment")

    # create environment with a map
    env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)
    # try alternatives, e.g. map_name="straight_road" (see the main readme of gym-duckietown)
    env.reset()
    env.render()
    # associate our keypress function with env
    env.window.on_key_press = key_press

    print("Entering main simulation loop")
    main()

    print("Simulation terminated, exiting")
    env.close()
