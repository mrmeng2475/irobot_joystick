##==库导入=====================================================
import math 
import sys
import time 
import keyboard
import numpy as np

import serial
from communicate.channel import ChannelSubscriber, ChannelFactoryInitialize
from RNN_publisher import th_Data
from joystick import joystick_data

#映射函数
def map_to_range(x, x_min, x_max, y_min, y_max):  
    # 确保 x 在 [x_min, x_max] 之间  
    # if x < x_min or x > x_max:  
    #     raise ValueError(f"x must be in the range [{x_min}, {x_max}]")  

    return y_min + (y_max - y_min) * ((x - x_min) / (x_max - x_min))
#映射矩阵
mapft=np.array([
         [np.pi,-np.pi],
         [-np.pi,np.pi],
         [-np.pi,np.pi],
         [-np.pi,np.pi],
         [np.pi,-np.pi],
         [-np.pi,np.pi],
         [-np.pi,np.pi],])



##==主程序====================================================
#飞特舵机串口
# sys.path.append("C:\ENpath_nospace\HRIprogram\shijiao_caiji\program1\SCServo_Python")
from SCServo_Python.scservo_sdk import *                      # Uses SCServo SDK library
# Initialize PortHandler instance
portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = sms_sts(portHandler)
#teacherHandler = scscl(portHandler)
# Open port # Set port baudrate
if portHandler.openPort() and portHandler.setBaudRate(1000000):
    print("Succeeded to open the port")

SCS_MOVING_SPEED = 300  # SCServo moving speed  
SCS_MOVING_ACC = 100    # SCServo moving acceleration 
        

#记录程序启动的时间
#starttime=time.time()
for ft_id in range(1,9):
    packetHandler.WritePosEx(ft_id, 2048, 300, 50)
print('启动')

# Define the servo IDs you want to read
arm_servo_id=[1,2,3,4,5,6,7,8]

wr_pos={}
# Main program logic

#启动检测
count=0
data=[0]*7
time.sleep(1)
print('示教程序启动')

#缓慢到达起点
time.sleep(3)
packetHandler.WritePosEx(8, 1600, 200, 100)#写入夹爪初始角度
print("Initializing ChannelFactory...")
ChannelFactoryInitialize()
print("Creating ChannelSubscriber...")
sub_th = ChannelSubscriber("topic", th_Data)
sub_th.Init()

sub_joystick = ChannelSubscriber("topic", joystick_data)
sub_joystick.Init()

print("初始化完成")

while KeyboardInterrupt!=True:
    msg_joystick = sub_joystick.Read()
    msg_th = sub_th.Read()

    # 将读取的角度数据写入机械臂
    # print(msg)
    if msg_joystick.matrix_data[6] == 1:
        packetHandler.WritePosEx(8, 2350, 200, 100)
    if msg_joystick.matrix_data[7] == 1:
        packetHandler.WritePosEx(8, 1600, 200, 100)
    for k in range(1,8):
        #等待订阅消息角度 pos, result, error=teacherHandler.ReadPos(tech_servo_ids[k-1])
        wr_pos[k]=map_to_range(msg_th.matrix_data[k-1], mapft[k-1,0], mapft[k-1,1], 0, 4096)
        # if k==7:
        packetHandler.WritePosEx(arm_servo_id[k-1], int(wr_pos[k]), SCS_MOVING_SPEED, SCS_MOVING_ACC)
    print(wr_pos)




time.sleep(2)
for ft_id in range(1,9):
    packetHandler.WritePosEx(ft_id, 2048, 300, 50)
packetHandler.WritePosEx(4, 1024, 300, 50)
packetHandler.WritePosEx(8, 1600, 300, 50)
time.sleep(2)
# Close port
portHandler.closePort()

