import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data from excel
excel = pd.read_excel('PID_data.xlsx', sheet_name='Sheet1')
steering_angle = excel.iloc[0:1500,4]
step_count = excel.iloc[0:1500,0]
distance_to_road_center = excel.iloc[0:1500,5]
angle_from_straight_in_rads = excel.iloc[0:1500,6] #1500

#reference trajectory
distance_reference = [0]*1500

#plot
plt.plot(step_count,distance_reference,label='distance_reference')
plt.plot(step_count,distance_to_road_center,label='distance_to_road_center')
plt.ylim(-1, 1)
plt.xlabel('step_count')
plt.ylabel('distance to roard_center')
plt.title('PID performance')
plt.legend()
plt.show()
