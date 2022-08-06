#THIS IS A README FILE

#Training data in CNN and neural network: 
Traindatahough.py use the segment data as input to train, size(640,480,1)
Traindata.py use the RGB image directly as input to train, size(640,480,3)

#Train data second stage
loadandtainmodel.py
loadandtrainv2.py
Those two .py file will train the data using more processed data.

#obtaindata
obtaindata and put into two data set, one data set contains all image data, another is the image data file name and steering angle and other parameters.
control_template_v1.py   change some parameters to optimize the control process and obtain data

#control with trained model
controlwithmodel.py control using the RGB model
controlwithhoughmodel.py control using segment data, can control the car directly and more precisely

#imageconverttest
imageconvertest.py

#model.h5
model.h5 RGB data model
modelreghough.h5 Segment data model
model1.h5 Segment data model train using regularizations and drop out function

control_MPC_test.py 
The main code of MPC model

control_PID.py
The main code of PID control

Result_MPC.py
Plot the trajectory of MPC model simulation

Result_PID.py
Plot the trajectory of PID control simulation
