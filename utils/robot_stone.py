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
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import math3d as m3d
import logging

import matplotlib.pyplot as plt
import time
import socket

from Arduino_motor import Arduino_motor
import pybullet as p
import pybullet_data
import pybullet_utils
from collision_utils import get_collision_fn
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point
from shapely.geometry import LineString
import json

class Robot():

    def __init__(self, tcp_host_ip):

        self.camera_width = 960
        self.camera_height = 540
        self.workspace_limits = np.array([[-0.015, 0.115], [0.63, 0.76], [0.02, 0.08]])
        self.heightmap_resolution = 0.0013/4.0

        # robot gripper parameter 
        self.finger_length = 0.125
        self.l0 = 0.0125
        self.l1 = 0.1
        self.l2l = 0.019
        self.l2r = 0.01
        logging.basicConfig(level=logging.WARN)
        self.rob = urx.Robot(tcp_host_ip) #"192.168.1.102"
        
        self.control_exted_thumb=Arduino_motor()
        self.tcp_host_ip = tcp_host_ip
        self.resetRobot()
        self.resetFT300Sensor()
        self.setCamera()
        self.collision_check_initialization(bowl_position = [0.05, 0.695, 0.035])
        self.DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C", "0B64"]
        
    def resetRobot(self):
        self.go_to_home_up()
        self.rob.set_tcp((0, 0.0, 0, 0, 0, 0))
        time.sleep(0.2)
        self.control_exted_thumb.set_thumb_length_int(180)
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)
        self.robotiqgrip.gripper_activate()
        self.gp_control_distance(0.03)
        self.go_to_home()
        self.baseTee = self.rob.get_pose()
        self.baseTee.orient =  np.array([[0,  1, 0], [ 1,  0,  0], [ 0, 0, -1]])
        print("reset")

    def resetFT300Sensor(self):
        HOST = self.tcp_host_ip
        PORT = 63351
        self.serialFT300Sensor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serialFT300Sensor.connect((HOST, PORT))

    def getFT300SensorData(self):
        while True:
            data = str(self.serialFT300Sensor.recv(1024),"utf-8").replace("(","").replace(")","").split(",")
            try:
                data = [float(x) for x in data]
                if len(data)==6:
                    break
            except:
                pass
        return data

    def find_device_json_input_interface(self) :
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices();
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No product line device that has json input interface")



    def setCamera(self):
        jsonDict = json.load(open("/home/terry/catkin_ws/src/dg_learning_real_one_net/utils/camera_info.json"))
        jsonString= str(jsonDict).replace("'", '\"')
        #pipeline = rs.pipeline()
        #config = rs.config()
        try:
            dev = self.find_device_json_input_interface()
            ser_dev = rs.serializable_device(dev)
            ser_dev.load_json(jsonString)
            print("loaded json")
        except Exception as e:
            print(e)
            pass
        self.cam_intrinsics = np.asarray([[670.74, 0, 488.591], [0, 671.034, 268.555], [0, 0, 1]])
        self.eeTcam = np.array([[0, -1, 0, 0.186+0.0015], #0.17
                           [1, 0, 0, -0.004-0.005], #0.017
                           [0, 0, 1, 0.086], #0.105   0.07  0.092
                           [0, 0, 0, 1]])

        self.baseTcam = np.matmul(self.baseTee.get_matrix(), self.eeTcam)
        print("baseTcam", self.baseTcam)

    def getCameraData(self, data_order=0):
        # Set the camera settings.
        # Setup:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        #print("depth_scale", depth_scale)

        # Store next frameset for later processing:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Cleanup:
        pipeline.stop()
        #print("Frames Captured")

        color = np.asanyarray(color_frame.get_data())
        #color_img_path = './picture_20210422/color_img/'+str(data_order)+'_color.jpg'
        #cv2.imwrite(color_img_path, color)

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()
        depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = depth_image_raw*depth_scale-0.01
        #depth_img_path = './picture_20210422/depth_img/'+str(data_order)+'_depth.jpg'
        #cv2.imwrite(depth_img_path, depth_image*1000)
        #colorizer = rs.colorizer()
        #colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        return color, depth_image

        #return color, depth_image, color_img_path, depth_img_path

    def gp_control_int(self, aperture_int, delay_time = 0.2):
        self.robotiqgrip.gripper_action(aperture_int)
        time.sleep(delay_time)

    def gp_control_distance(self, aperture_distance, delay_time = 0.2): #meter
        int_list=np.array([0,20,40,60,80,100,120,130,140,150,160,170,180,185,190,195,200,205,210]) #
        distance_list= np.array([125,116.39,105.33,93.91,82.48,70.44,58.09,51.90,45.47,39.05,32.84,26.76,19.65,16.39,13.32,10.20,7.76,3.89,0])/1000
        func_from_distance_to_int = interpolate.interp1d(distance_list, int_list, kind = 3)
        self.robotiqgrip.gripper_action(int(func_from_distance_to_int(aperture_distance)))
        time.sleep(delay_time)

    def go_to_home(self):
        home_position = [101.21,-78.70,-128.37,-62.32,90.40,-79.24]
        Hong_joint0 = radians(home_position[0])
        Hong_joint1 = radians(home_position[1])
        Hong_joint2 = radians(home_position[2])
        Hong_joint3 = radians(home_position[3])
        Hong_joint4 = radians(home_position[4])
        Hong_joint5 = radians(home_position[5])

        self.rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), acc=0.6, vel=0.8)
        #self.rob.translate((0.03, 0, 0), 0.3, 0.5)

    def go_to_home_up(self):
        home_position = [100.21,-77.82,-121.66,-69.66,90.38,-81.95]
        Hong_joint0 = radians(home_position[0])
        Hong_joint1 = radians(home_position[1])
        Hong_joint2 = radians(home_position[2])
        Hong_joint3 = radians(home_position[3])
        Hong_joint4 = radians(home_position[4])
        Hong_joint5 = radians(home_position[5])

        self.rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), acc=0.6, vel=0.8)

    def Frame(pos, ori):
        mat = R.from_quat(ori).as_matrix()
        F = np.concatenate(
            [np.concatenate([mat, [[0, 0, 0]]], axis=0), np.reshape([*pos, 1.], [-1, 1])], axis=1
        )
        return F

    def exe_scoop(self, pos, rot_z, ini_aperture, thumb_extend=0, theta = 60*pi/180, roll = 0):  # rot_z rad   aperture distance
        '''
        if self.collision_check_scooping(pos, rot_z, ini_aperture, theta)==True:
            print("Collision!")
            return -1
        '''
        thumb_extend-=0.005
        self.go_to_home()
        self.rob.translate((0,0,0.06), acc=0.5, vel=0.8)
        if ini_aperture == 0.05:
            self.rob.set_tcp((0.0295, 0.0, 0.3389, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.04:
            self.rob.set_tcp((0.0245, 0.0, 0.3396, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.035:
            self.rob.set_tcp((0.022, 0.0, 0.3396, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.03:
            self.rob.set_tcp((0.0195, 0.0, 0.3414, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.025:
            self.rob.set_tcp((0.017, 0.0, 0.3414, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.02:
            self.rob.set_tcp((0.0145, 0.0, 0.342, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.015:
            self.rob.set_tcp((0.012, 0.0, 0.3425, 0, 0, 0))   #need change
            time.sleep(0.3)
        else:
            print(ini_aperture, "Wrong ini aperture!")
            raise NameError("Wrong ini aperture!")
        self.gp_control_distance(ini_aperture, delay_time = 0.5)
        #time.sleep(0.5)
        self.control_exted_thumb.set_thumb_length_int(0, wait_time = 0.5)
        #time.sleep(0.5)
        eefPose = self.rob.get_pose()
        time.sleep(0.3)
        eefPose = eefPose.get_pose_vector()
        self.rob.translate((pos[0]-eefPose[0]-0.004,pos[1]-eefPose[1],0), acc=0.5, vel=0.8)
        self.rob.movel_tool((0,0,0,0,0,rot_z), acc=0.5, vel=0.8)
        self.rob.translate((0,0,pos[2]-eefPose[2]+0.01), acc=0.05, vel=0.05)
        self.rob.movel_tool((0,0,0,0,pi/2-theta,0), acc=0.5, vel=0.8, wait=True)
        self.rob.movel_tool((0,0,0,roll,0,0), acc=0.5, vel=0.8, wait=True)
        self.rob.translate((0, 0, -0.03), acc=0.01, vel=0.01, wait=False)
        ini_force_z = self.getFT300SensorData()[2]
        time0 = time.time()
        num_large_force = 0
        while True:
            if num_large_force == 3:
                self.rob.stopl()
                break
            force_z = self.getFT300SensorData()[2]
            if force_z < ini_force_z-1.3: #key 2  #Acrylic 1.8 #Go stone 1.3
                num_large_force += 1
            if time.time()-time0>3.3:  
                break 
        time.sleep(0.1)
        ini_torque_y = self.getFT300SensorData()[4]
        shortest_thumb_length = self.control_exted_thumb.shortest_thumb_length
        longest_thumb_length = self.finger_length-ini_aperture/tan(theta)+thumb_extend
        for current_thumb_length in np.arange(shortest_thumb_length,longest_thumb_length+(1e-8), (longest_thumb_length-shortest_thumb_length)/10):
            self.control_exted_thumb.set_thumb_length(current_thumb_length)     
            torque_y = self.getFT300SensorData()[4]
            if torque_y>ini_torque_y+0.2:
                break
        for aperture_distance in np.arange(ini_aperture, -1e-5, -0.002):
            aperture_angle = self.from_aperture_distance_to_angle(aperture_distance, self.l0, self.l1, self.l2l, self.l2r)
            next_aperture_angle = self.from_aperture_distance_to_angle(aperture_distance-0.002, self.l0, self.l1, self.l2l, self.l2r)
            translate_dir_dis, extension_distance = self.scooping_parameter_finger_fixed(aperture_angle, next_aperture_angle, theta*180/pi, self.l0, self.l1, self.l2l, self.l2r, self.finger_length)
            current_thumb_length += extension_distance  
            self.control_exted_thumb.set_thumb_length(current_thumb_length)
            self.gp_control_distance(max(aperture_distance, 0), delay_time = 0)
            self.rob.translate_tool((sin(theta)*translate_dir_dis[0]+cos(theta)*translate_dir_dis[1], 0, cos(theta)*translate_dir_dis[0]-sin(theta)*translate_dir_dis[1]), acc=0.1, vel=0.4, wait=False)
            #time.sleep(0.5)
        self.gp_control_int(220, delay_time = 0.1)
        self.rob.translate_tool((0, 0, -0.12), acc=0.4, vel=1, wait=True)
        self.go_to_home_up()
        self.go_to_home()
        self.gp_control_distance(0.03)
        while True:
            whether_successful = input("Whether successful? (y or n)")
            if whether_successful == 'n':
                return 0
            elif whether_successful == 'y':
                return 1


    def finger_tip_position_wrt_gripper_frame(self, alpha, l0, l1, l2l, l2r, flength_l, flength_r):
        alpha=alpha*pi/180
        left_fingertip_position=[-l0-l1*sin(alpha)+l2l, -l1*cos(alpha)-flength_l]
        right_fingertip_position=[l0+l1*sin(alpha)-l2r, -l1*cos(alpha)-flength_r]
        return left_fingertip_position, right_fingertip_position

    def from_aperture_distance_to_angle(self, distance, l0, l1, l2l, l2r):  #angle
        return asin((distance+l2l+l2r-2*l0)/(2*l1))*180/pi

    def scooping_parameter_finger_fixed(self, current_alpha, next_alpha, theta, l0, l1, l2l, l2r, flength_r):
        theta=theta*pi/180
        current_flength_l = flength_r - (2*l0+2*l1*sin(current_alpha*pi/180)-l2l-l2r)/tan(theta)
        current_left_fingertip_pos_g, current_right_fingertip_pos_g = self.finger_tip_position_wrt_gripper_frame(current_alpha, l0, l1, l2l, l2r, current_flength_l, flength_r)  
        next_left_fingertip_pos_g, next_right_fingertip_pos_g = self.finger_tip_position_wrt_gripper_frame(next_alpha, l0, l1, l2l, l2r, current_flength_l, flength_r) 
        traslating_wst_g = [current_right_fingertip_pos_g[0]-next_right_fingertip_pos_g[0], current_right_fingertip_pos_g[1]-next_right_fingertip_pos_g[1]]
        traslating_wst_world = np.array([[sin(theta), -cos(theta)], [cos(theta), sin(theta)]]).dot(np.array([[traslating_wst_g[0]], [traslating_wst_g[1]]])).ravel().tolist()
        next_left_fingertip_pos_g = [next_left_fingertip_pos_g[0]+traslating_wst_g[0], next_left_fingertip_pos_g[1]+traslating_wst_g[1]]
        next_right_fingertip_pos_g = [next_right_fingertip_pos_g[0]+traslating_wst_g[0], next_right_fingertip_pos_g[1]+traslating_wst_g[1]]
        extension_distance = -(next_right_fingertip_pos_g[1]+(2*l0+2*l1*sin(next_alpha*pi/180)-l2l-l2r)/tan(theta)-next_left_fingertip_pos_g[1])
        return traslating_wst_world, extension_distance

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

        start_conf_thumb = [self.control_exted_thumb.shortest_thumb_length-(self.finger_length-ini_aperture/tan(theta))]
        #start_conf_thumb = [0.1-(self.finger_length-ini_aperture/tan(theta))]
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
            self.set_joint_positions_collision_check(self.thumb_collision_check, self.THUMB_JOINT_INDICES, [self.control_exted_thumb.shortest_thumb_length-(self.finger_length-current_aperture/tan(theta))])
            p.resetBasePositionAndOrientation(self.thumb_collision_check, thumbBasePosition, thumbOrientation)
            collision_fn.append(len(p.getClosestPoints(bodyA=self.thumb_collision_check, bodyB=self.bowl_circular_collision_check, distance=0, physicsClientId=self.CLIENT)) != 0)
        result = False
        for element in collision_fn:
            result = result or element
        #print(collision_fn)
        return result

    def fromResultToRobotParameter(self, MaskRCNNResult, workspace_limits):
        random.shuffle(MaskRCNNResult)
        for ObjectPoseIndex in range(len(MaskRCNNResult)):
            ObjectPose = MaskRCNNResult[ObjectPoseIndex]
            if ObjectPose['z'] == 0:
                continue
            CamTObject = np.array([ObjectPose['x'], ObjectPose['y'], ObjectPose['z'], 1])
            BaseTObject = np.matmul(self.baseTcam, CamTObject).tolist()[0][0:3]
            #print('BaseTObject', BaseTObject)
            #print('workspace_limits', self.workspace_limits)
            if BaseTObject[0]<=workspace_limits[0][0] or BaseTObject[0]>=workspace_limits[0][1] or BaseTObject[1]<=workspace_limits[1][0] or BaseTObject[1]>=workspace_limits[1][1] or BaseTObject[2]<=workspace_limits[2][0] or BaseTObject[2]>=workspace_limits[2][1]:
                continue
            NormalInCamera = ObjectPose['normal'] / np.linalg.norm(ObjectPose['normal'])
            NormalInTcp = np.matmul(self.eeTcam[0:3, 0:3], NormalInCamera).tolist()
            alpha0 = atan2(NormalInTcp[1], NormalInTcp[0]) # is yaw
            if str(alpha0) == 'nan':
                continue
            beta = -(pi/2+atan2(NormalInTcp[2], sqrt(NormalInTcp[0]**2+NormalInTcp[1]**2))) #tune psi
            if str(beta) == 'nan':
                continue
            if abs(beta)<(5*pi/180):
                alpha_set = [0, pi/4, -pi/4, pi*3/4, -pi*3/4, pi,  pi/2, -pi/2]
                rot_z = alpha_set[random.randint(0,len(alpha_set))]
                pos = [BaseTObject[0]+0.01*sin(rot_z), BaseTObject[1]+0.01*cos(rot_z), BaseTObject[2]]
            else:
                rot_z = round(alpha0/(pi/4))*(pi/4)
                pos = [BaseTObject[0]+0.01*cos(beta)*sin(rot_z), BaseTObject[1]+0.01*cos(beta)*cos(rot_z), BaseTObject[2]]
            return pos, rot_z
        else:
            return None, None

            

    

if __name__ == '__main__':
    robot = Robot("192.168.1.102")
    #time.sleep(1)
    '''
    robot.control_exted_thumb.set_thumb_length_int(20)
    time.sleep(0.5)
    robot.control_exted_thumb.set_thumb_length_int(180)
    time.sleep(0.5)
    robot.control_exted_thumb.set_thumb_length_int(20)
    time.sleep(0.5)
    robot.control_exted_thumb.set_thumb_length_int(180)
    time.sleep(0.5)
    '''
    #print(robot.collision_check_scooping([0.05, 0.695, 0.035], 0, 0.03, theta = 60*pi/180))
    # extendable most activate
    #MaskRCNNResult = [{'x': 0.015240815418306903, 'y': -0.009129762957050931, 'z': 0.139, 'yaw': -2.525972764160683, 'pitch': 0.4808872801953352, 'normal': np.array([-0.29894826, -0.04516211, -0.59502014])}, {'x': -0.016908181205929552, 'y': -0.00791901611213648, 'z': 0.146, 'yaw': -2.152176510599796, 'pitch': 0.7853981633974487, 'normal': np.array([ 0.40818817,  0.12813395, -0.53918177])}, {'x': -0.004103737720303621, 'y': -0.02047507367685938, 'z': 0.133, 'yaw': -2.049518170790662, 'pitch': 0.5779019369622452, 'normal': np.array([-0.16970485, -0.28701037, -0.62522159])}, {'x': -0.00411056702536938, 'y': 0.019195781413677854, 'z': 0.149, 'yaw': 1.6539375586833378, 'pitch': 1.0595656145171073, 'normal': np.array([ 0.17626152,  0.50008089, -0.46678416])}]
    #print(robot.fromResultToRobotParameter(MaskRCNNResult))
    #test1_array = np.load("depth_img_ROS.npy")
    #cv2.imwrite('./test1.jpg', test1_array)

    #test2_array = np.load("depth_img_dipn.npy")*1000
    #cv2.imwrite('./test2.jpg', test2_array)
    robot.gp_control_distance(0.015, delay_time = 0.2)
    robot.go_to_home_up()
    robot.exe_scoop([0.05, 0.85, 0.05], 0, 0.025, thumb_extend=-0.005)
    #while True:
        #print(robot.getFT300SensorData())
    




