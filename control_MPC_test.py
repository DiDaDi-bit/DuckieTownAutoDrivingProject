#!/usr/bin/env python3

"""
Contents:
* Creates a duckietown environment
* Shows how ground truth data can be sampled from the simulator
* Gives an MPC model to dotrajectory tracking
"""

# import relevant libraries
from pickle import FALSE, TRUE
import time

import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo


#create MPC model
def MPC(cte, epsi):
    N = 5 # forward predict steps(prediction horizon)
    ns = 2  # state numbers (cte,epsi)  

    #create model
    model = pyo.ConcreteModel()
    model.ek = pyo.RangeSet(0,N-1)
    model.uk = pyo.RangeSet(0,N-2)
    model.delta_uk = pyo.RangeSet(0,N-3)

    #parameters
    #wg-weight coefficient 
    #dt-time interval
    #Lf-estimated length of car
    #ref_cte-the reference of cte
    #ref_epsi-the reference of epsi
    #s0-initial state of the vehicle
    model.wg = pyo.Param(pyo.RangeSet(0, 3), initialize={0:1, 1:0.5, 2:60., 3:0.5}, mutable=True) 
    model.dt = pyo.Param(initialize=0.1,mutable=True)
    model.Lf = pyo.Param(initialize=2,mutable=True)
    model.carSpeed = pyo.Param(initialize=1,mutable=True)
    model.ref_cte = pyo.Param(initialize=0,mutable=True)
    model.ref_epsi = pyo.Param(initialize=0,mutable=True)
    model.s0 = pyo.Param(pyo.RangeSet(0, ns-1), initialize={0:cte, 1:epsi}, mutable=True)


    #variables
    #creater variable array for cte and epsi, 0: cte, 1: epsi
    model.varArray = pyo.Var(pyo.RangeSet(0,ns-1),model.ek)
    #bounds of vehicle's steering angle
    model.steering_angle = pyo.Var(model.uk, bounds=(-1,1))

    #constraints
    #fit the initial state to varArray
    model.s0_fit = pyo.Constraint(pyo.RangeSet(0, ns-1), rule = lambda model, i: model.varArray[i,0] == model.s0[i])
    #cte(k+1) and epsi(k+1) expressions
    model.cte_next = pyo.Constraint(model.ek,rule=lambda model,k:
                                model.varArray[0,k+1]==model.varArray[0,k]+model.carSpeed*pyo.sin(model.varArray[1,k])*model.dt
                                if k<N-1 else pyo.Constraint.Skip)
    model.epsi_next = pyo.Constraint(model.ek,rule=lambda model,k:
                                model.varArray[1,k+1]==-model.varArray[1,k]+model.carSpeed*pyo.tan(model.steering_angle[k])/model.Lf*model.dt
                                if k<N-1 else pyo.Constraint.Skip)                              

    #objective
    #each sections of objective
    model.obj_cte = model.wg[2]*sum((model.varArray[0,k]-model.ref_cte)**2 for k in model.ek)
    model.obj_epsi = model.wg[2]*sum((model.varArray[1,k]-model.ref_epsi)**2 for k in model.ek)
    model.obj_stAngle = model.wg[1]*sum(model.steering_angle[k]**2 for k in model.uk)
    model.obj_eStAngle = model.wg[3]*sum((model.steering_angle[k+1]-model.steering_angle[k])**2 for k in model.delta_uk)
    #add each sections 
    model.obj = pyo.Objective(expr = model.obj_cte + model.obj_epsi + model.obj_stAngle + model.obj_eStAngle,sense=pyo.minimize)

    #solve this optimisation problem and output the first steering angle
    pyo.SolverFactory('ipopt').solve(model)
    steering_angle = model.steering_angle[0]()
    return steering_angle


def main():
    #create excel to record data
    workbook=xlsxwriter.Workbook('MPC_data.xlsx')
    worksheet=workbook.add_worksheet()
    # create environment with a map
    env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)
    env.reset()
    env.render()
    # define total reward
    total_reward = 0
    row = 2
    #excel titles
    worksheet.write('A1','step')
    worksheet.write('B1','reward')
    worksheet.write('C1','total_reward')
    worksheet.write('D1','car_speed')
    worksheet.write('E1','steering_angle')
    worksheet.write('F1','distance')
    worksheet.write('G1','angle') 
    
     # main loop
    while True:
        # getting our lane position
        # in practice these should be provided by the camera/vision system!
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad
        car_speed = 0.05

        # decide on steering angle using MPC
        steering_angle = MPC(distance_to_road_center , angle_from_straight_in_rads)
 
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
