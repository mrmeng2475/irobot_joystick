import pygame
import time
import numpy as np
import time
import sys
sys.path.append("/home/mrmeng/work_space/irobot_joystick")
from communicate.channel import ChannelPublisher, ChannelFactoryInitialize
from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array


INITIAL_TARGET_MM = (345, -160, 1100, 345, 160, 1100)


@dataclass
class joystick_data(IdlStruct, typename="joystick_data"):
    matrix_data: array[float,8]    # 7x1 matrix represented as a list of 7 floats


def initialize_joystick():
    pygame.init()
    pygame.joystick.init()
    
    # 检查连接的手柄数量
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("未检测到手柄连接")
        return None
    
    # 初始化第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"已连接手柄: {joystick.get_name()}")
    print(f"摇杆数量: {joystick.get_numaxes()}")
    print(f"按钮数量: {joystick.get_numbuttons()}")
    
    return joystick

def read_joystick_data(joystick):
    # 处理事件队列
    pygame.event.pump()
    
    # 获取摇杆数据
    axes = []
    for i in range(joystick.get_numaxes()):
        axis_value = joystick.get_axis(i)
        axes.append(axis_value)
    
    # 获取按钮状态
    buttons = []
    for i in range(joystick.get_numbuttons()):
        button_state = joystick.get_button(i)
        buttons.append(button_state)
    
    return axes, buttons

def exchange_joystick(axes):
    if axes > 0.1 or axes < 0.1:
        control = round(axes, 0)
    else:
        control = 0
    return control


def main():
    joystick = initialize_joystick()
    if joystick is None:
        return
    msg_joystick = list(INITIAL_TARGET_MM)
    msg_control = [0,0,0,0,0,0,0,0]
    ChannelFactoryInitialize()

    pub = ChannelPublisher("topic",joystick_data)
    pub.Init()

    try:
        while True:
            axes, buttons = read_joystick_data(joystick)

            if buttons[2] == 1:
                msg_joystick = list(INITIAL_TARGET_MM)
                
            #左臂
            msg_control[0] = exchange_joystick(axes[0])
            msg_control[1] = exchange_joystick(axes[1])
            if axes[2] < 0:
                msg_control[2] = 0
            else:
                msg_control[2] = -1
            msg_control[3] = buttons[4]

            #右臂
            msg_control[4] = exchange_joystick(axes[3])
            msg_control[5] = exchange_joystick(axes[4])
            if axes[5] < 0:
                msg_control[6] = 0
            else:
                msg_control[6] = -1
            msg_control[7] = buttons[5]



            #左臂遥控
            msg_joystick[0] = msg_joystick[0] - msg_control[5]*10 
            msg_joystick[1] = msg_joystick[1]- msg_control[4]*10    
            msg_joystick[2] = msg_joystick[2] + msg_control[6]*10 + msg_control[7]*10
            #右臂遥控
            msg_joystick[3] = msg_joystick[3] - msg_control[1]*10 
            msg_joystick[4] = msg_joystick[4]- msg_control[0]*10  
            msg_joystick[5] = msg_joystick[5] + msg_control[2]*10 + msg_control[3]*10


            matrix_data = [
                float(msg_joystick[0]),
                float(msg_joystick[1]),
                float(msg_joystick[2]),
                float(msg_joystick[3]),
                float(msg_joystick[4]),
                float(msg_joystick[5]),
                float(buttons[0]),
                float(buttons[1])
            ]
            
            msg = joystick_data(matrix_data=matrix_data)
            if pub.Write(msg,0.5):
                print("Pubulish success. msg:",msg)
            else:
                print("Waiting for subscriber.")
            time.sleep(0.1)  # 控制输出频率
            
    except KeyboardInterrupt:
        print

if __name__ == "__main__":
    main()
