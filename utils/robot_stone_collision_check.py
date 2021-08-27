import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import os
import numpy as np
from math import *
import math3d as m3d
import random

import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import urx
import math3d as m3d
import logging

import matplotlib.pyplot as plt
import time
import socket
import pybullet as p
import pybullet_data
import pybullet_utils
from collision_utils import get_collision_fn
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point
from shapely.geometry import LineString
import json

class Robot():

    def __init__(self):

        # robot gripper parameter 
        self.finger_length = 0.125
        self.l0 = 0.0125
        self.l1 = 0.1
        self.l2l = 0.019
        self.l2r = 0.01
        
        self.collision_check_initialization(bowl_position = [0.05, 0.695, 0.035])

    '''
    def __init__(self):
        # robot gripper parameter 
        self.finger_length = 0.125
        self.l0 = 0.0125
        self.l1 = 0.1
        self.l2l = 0.019
        self.l2r = 0.01
        
        self.collision_check_initialization(bowl_position = [0.05, 0.695, 0.03])
    '''

    def set_joint_positions_collision_check(self, body, joints, values):
        assert len(joints) == len(values)
        for joint, value in zip(joints, values):
            p.resetJointState(body, joint, value)

    def collision_check_initialization(self, bowl_position):
        self.THUMB_JOINT_INDICES = [0]
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
        p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))
        self.CLIENT = 0
        self.finger_collision_check = p.loadURDF('/home/terry/catkin_ws/src/scoop/models/urdf/Finger-2.9-urdf.urdf', basePosition=[0, 0, 0], baseOrientation=[0,0,1,0], useFixedBase=False)
        self.thumb_collision_check = p.loadURDF('/home/terry/catkin_ws/src/scoop/models/urdf/Assem-1.9-urdf.urdf', basePosition=[0, 0, 0], baseOrientation=[0,0,1,0],useFixedBase=False)
        self.bowl_position = bowl_position
        self.bowl_circular_collision_check = p.loadURDF('/home/terry/catkin_ws/src/scoop/models/urdf/Bowl-circular-urdf.urdf', basePosition=self.bowl_position, baseOrientation=[0,0,0,1], useFixedBase=True)
        self.obstacles_collision_check = [self.bowl_circular_collision_check]

    def collision_check_scooping(self, pos, rot_z, ini_aperture, theta = 60*pi/180, roll=0):
        if LineString([[pos[0]+0.01*cos(rot_z), pos[1]-0.01*sin(rot_z)], [pos[0]-0.01*cos(rot_z), pos[1]+0.01*sin(rot_z)]]).within(Point([self.bowl_position[0],self.bowl_position[1]]).buffer(0.06))==False:
            return True
        if LineString([[pos[0]-ini_aperture/sin(theta)*sin(rot_z)+0.01*cos(rot_z), pos[1]-ini_aperture/sin(theta)*cos(rot_z)-0.01*sin(rot_z)], [pos[0]-ini_aperture/sin(theta)*sin(rot_z)-0.01*cos(rot_z), pos[1]-ini_aperture/sin(theta)*cos(rot_z)+0.01*sin(rot_z)]]).within(Point([self.bowl_position[0],self.bowl_position[1]]).buffer(0.06))==False:
            return True

        #start_conf_thumb = [self.control_exted_thumb.shortest_thumb_length-(self.finger_length-ini_aperture/tan(theta))]
        start_conf_thumb = [0.1-(self.finger_length-ini_aperture/tan(theta))]
        fingerOrientation_raw = R.from_dcm([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) * R.from_euler('z', rot_z, degrees=False) * R.from_euler('y', pi/2-theta, degrees=False) * R.from_euler('x', roll, degrees=False)
        fingerOrientation = fingerOrientation_raw.as_quat().tolist()
        ThumbTFinger_normal = fingerOrientation_raw.as_dcm()[:,0].tolist()
        fingertipPosition = pos
        p.resetBasePositionAndOrientation(self.finger_collision_check, fingertipPosition, fingerOrientation)
        collision_fn = [len(p.getClosestPoints(bodyA=self.finger_collision_check, bodyB=self.bowl_circular_collision_check, distance=0, physicsClientId=self.CLIENT)) != 0]

        fingerBasePosition = list(p.getLinkState(self.finger_collision_check, 0)[0])
        thumbOrientation = fingerOrientation
        for finger_thumb_distance in np.arange(ini_aperture+self.l2l+self.l2r, -0.0001+self.l2l+self.l2r, -ini_aperture/5):
            thumbBasePosition = [fingerBasePosition[0]-finger_thumb_distance*ThumbTFinger_normal[0], fingerBasePosition[1]-finger_thumb_distance*ThumbTFinger_normal[1], fingerBasePosition[2]-finger_thumb_distance*ThumbTFinger_normal[2]]
            current_aperture = finger_thumb_distance-self.l2l-self.l2r
            #self.set_joint_positions_collision_check(self.thumb_collision_check, self.THUMB_JOINT_INDICES, [self.control_exted_thumb.shortest_thumb_length-(self.finger_length-current_aperture/tan(theta))])
            self.set_joint_positions_collision_check(self.thumb_collision_check, self.THUMB_JOINT_INDICES, [0.1-(self.finger_length-current_aperture/tan(theta))])
            p.resetBasePositionAndOrientation(self.thumb_collision_check, thumbBasePosition, thumbOrientation)
            collision_fn.append(len(p.getClosestPoints(bodyA=self.thumb_collision_check, bodyB=self.bowl_circular_collision_check, distance=0, physicsClientId=self.CLIENT)) != 0)
        result = False
        for element in collision_fn:
            result = result or element
        #print(collision_fn)
        return result
    

if __name__ == '__main__':
    robot = Robot()
    print(robot.collision_check_scooping([0.05, 0.695, 0.055], 0, 0.025, theta = 60*pi/180))
    




