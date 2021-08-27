import sys
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import time

rob = urx.Robot("192.168.1.102")
robotiqgrip = Robotiq_Two_Finger_Gripper(rob)

for i in range(0,255,10):
    robotiqgrip.gripper_action(i)
    time.sleep(0.2)

for i in range(255,0,-10):
    robotiqgrip.gripper_action(i)
    time.sleep(0.2)


#robotiqgrip.open_gripper()
#robotiqgrip.close_gripper()

