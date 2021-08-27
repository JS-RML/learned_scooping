import sys
sys.path.append("./utils")
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import time
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from robot import Robot
from logger import Logger
from robot_stone import Robot
import heightmap

import math3d as m3d
import random
from math import *
import scipy
from torch import nn
from PIL import Image
import tier1_module
import tier2_module
import tier3_module
from datetime import datetime
from random import choice

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

htmap_w = 200
htmap_h = 200
workspace_limits_raw = np.array([[-0.015, 0.115], [0.63, 0.76], [0.02, 0.1]])
workspace_limits_list = [np.array([[-0.015, 0.05], [0.63, 0.695], [0.02, 0.1]]), np.array([[0.0175, 0.0825], [0.63, 0.695], [0.02, 0.1]]), np.array([[0.05, 0.115], [0.63, 0.695], [0.015, 0.1]]), np.array([[-0.015, 0.05], [0.6625, 0.7275], [0.02, 0.1]]), np.array([[0.0175, 0.0825], [0.6625, 0.7275], [0.02, 0.1]]), np.array([[0.05, 0.115], [0.6625, 0.7275], [0.02, 0.1]]), np.array([[-0.015, 0.05], [0.695, 0.76], [0.02, 0.1]]), np.array([[0.0175, 0.0825], [0.695, 0.76], [0.02, 0.1]]), np.array([[0.05, 0.115], [0.695, 0.76], [0.02, 0.1]])]
heightmap_resolution = 0.0013/4.0
net = tier1_module.Tier1(pcpt_block=tier1_module.BasicBlock, pcpt_layers=[1,5,1], scoop_block=tier1_module.BasicBlock, scoop_layers=[1,5,1], h=htmap_h, w=htmap_w).cuda()     # define the network
net.load_state_dict(torch.load('netparam_Tier1.pkl'))    # domino
net_submodule2 = tier2_module.Tier2(pcpt_block=tier2_module.BasicBlock, pcpt_layers=[1,1,1], scoop_block=tier2_module.BasicBlock, scoop_layers=[1,1,1], h=200, w=200).cuda()
net_submodule2.load_state_dict(torch.load('netparam_Tier2.pkl'))
net_submodule3 = tier3_module.Tier3(pcpt_block=tier3_module.BasicBlock, pcpt_layers=[1,5,1], scoop_block=tier3_module.BasicBlock, scoop_layers=[1,5,1], h=200, w=200).cuda()
net_submodule3.load_state_dict(torch.load('netparam_Tier3.pkl'))

def point_position_after_rotation(current_xy, rotation_pole, desired_angle):
    desired_angle_rad = desired_angle*pi/180
    current_displacement = list(np.array(current_xy)-np.array(rotation_pole))
    rotation_matrix = [[cos(desired_angle_rad), -sin(desired_angle_rad)],[sin(desired_angle_rad), cos(desired_angle_rad)]]
    current_displacement = np.expand_dims(current_displacement, axis=1)
    temp = np.dot(rotation_matrix, current_displacement)
    xy_after_rotate = [list(rotation_pole[0]+temp[0])[0], list(rotation_pole[1]+temp[1])[0]]
    return xy_after_rotate 

def from_angle_index_to_gripper_rot_z(angle_index):
    if angle_index>=0 and angle_index<=8:
        rot_z = (-pi/8)*angle_index
    else:
        rot_z = pi/8*(16-angle_index)
    return rot_z

def from_pixel_to_world_position(pixel, workspace_limits, heightmap_resolution, depth_heightmap):
    htmap_h = int(round((workspace_limits[1][1]-workspace_limits[1][0])/heightmap_resolution))
    htmap_w = int(round((workspace_limits[0][1]-workspace_limits[0][0])/heightmap_resolution))
    pix_x = pixel[0]
    pix_y = pixel[1]
    pos = [pix_x * heightmap_resolution + workspace_limits[0][0], pix_y * heightmap_resolution + workspace_limits[1][0], depth_heightmap[pix_y][pix_x] + workspace_limits[2][0]]
    return pos

def main():
    total_index = 0
    robot = Robot("192.168.1.102")
    while True:
        while True:
            whether_continue = input('Whether continue? (y or n) Shuffle!')
            if whether_continue=='y':
                break
        max_score_pixel_list = []
        color_img_array, depth_array = robot.getCameraData()

        for workspace_limits in workspace_limits_list:
            #color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img_array, depth_array, cam_intrinsics, baseTcam, workspace_limits, heightmap_resolution)

            color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img_array, depth_array, robot.cam_intrinsics, robot.baseTcam, workspace_limits, robot.heightmap_resolution)
            color_heightmap_image = Image.fromarray(color_heightmap)
            depth_heightmap_image = Image.fromarray((depth_heightmap/0.1*255).astype(np.uint8))
            for rotate_index in range(16):
                rotated_color_heightmap = np.array(color_heightmap_image.rotate(angle = rotate_index*22.5, fillcolor = (0,0,0))).astype(np.uint8) 
                rotated_depth_heightmap = np.array(depth_heightmap_image.rotate(angle = rotate_index*22.5, fillcolor = 0)).astype(np.uint8)
                color_depth_heightmap = np.concatenate((rotated_color_heightmap[:,:,[2,1,0]],rotated_depth_heightmap[:, :, np.newaxis]), axis = 2)[np.newaxis, :, :, :]
                color_depth_heightmap = torch.from_numpy(color_depth_heightmap/255.0).permute(0,3,1,2).cuda().float()
                with torch.no_grad():
                    net.eval()
                    predicted_score = net(color_depth_heightmap).permute(0,2,3,1)
                    soft_max_function = nn.Softmax(dim=3)
                    predicted_score = soft_max_function(predicted_score)[:,:,:,1]
                    predicted_score = torch.reshape(predicted_score, (predicted_score.shape[1], predicted_score.shape[2])).cpu().numpy()
                torch.cuda.empty_cache()
                max_score_pixel = np.unravel_index(np.argmax(predicted_score), predicted_score.shape)
                max_score_pixel = [max_score_pixel[1], max_score_pixel[0]]
                max_score_pixel_raw = [max_score_pixel]
                max_score_pixel = point_position_after_rotation([max_score_pixel[0], htmap_h-max_score_pixel[1]], [htmap_w/2, htmap_h/2], -rotate_index*22.5)   # plus or minus
                max_score_pixel = [int(max_score_pixel[0]), int(htmap_h-max_score_pixel[1])]
                if max_score_pixel[0]>=htmap_w or max_score_pixel[1]>=htmap_h or max_score_pixel[0]<0 or max_score_pixel[1]<0:
                    continue
                pos = from_pixel_to_world_position(max_score_pixel, workspace_limits, heightmap_resolution, depth_heightmap)
                yaw = from_angle_index_to_gripper_rot_z(rotate_index)
                #pos = [pos[0]+0.002*sin(yaw), pos[1]+0.002*cos(yaw), pos[2]]
                #print(np.max(predicted_score))
                if np.max(predicted_score)>0.8:
                    thumb_position_predict = net_submodule2(color_depth_heightmap, torch.from_numpy(np.array(max_score_pixel_raw)).cuda().float()).cpu().detach().numpy().tolist()
                    if thumb_position_predict[0][0]<5:
                        continue
                    aperture_pixel = abs(thumb_position_predict-max_score_pixel_raw[0][1])[0][0]
                    #print(aperture_pixel)
                    aperture_real = aperture_pixel*heightmap_resolution*sin(60*pi/180)
                    max_score_pixel_list.append([pos, yaw, np.max(predicted_score),aperture_real])

        max_score_pixel_list.sort(key=lambda x: x[2], reverse=True)
        for max_score_pixel in max_score_pixel_list:
            pos = max_score_pixel[0]
            if pos[2]==0.03:
                continue
            yaw = max_score_pixel[1]
            ini_aperture = max_score_pixel[3]     #Go stone 0
            ini_aperture = round(ini_aperture/0.005)*0.005
            #color_heightmap_raw, _ = heightmap.get_heightmap(color_img_array, depth_array, cam_intrinsics, baseTcam, workspace_limits_raw, heightmap_resolution)
            color_heightmap_raw, depth_heightmap_raw = heightmap.get_heightmap(color_img_array, depth_array, robot.cam_intrinsics, robot.baseTcam, workspace_limits_raw, robot.heightmap_resolution)
            copy_color_heightmap = color_heightmap_raw.copy()
            pix_x = int(max(min(round((pos[0] - workspace_limits_raw[0][0])/heightmap_resolution), 400-1), 0))
            pix_y = int(max(min(round((pos[1] - workspace_limits_raw[1][0])/heightmap_resolution), 400-1), 0))
            #cv2.circle(copy_color_heightmap, (pix_x,pix_y), 5, (0, 0, 255), 3)
            pix_x_prime = int(max(min(round((pos[0]-ini_aperture*sin(yaw) - workspace_limits_raw[0][0])/heightmap_resolution), 400-1), 0))
            pix_y_prime = int(max(min(round((pos[1]-ini_aperture*cos(yaw) - workspace_limits_raw[1][0])/heightmap_resolution), 400-1), 0))
            pix_x_prime_enlong = int(max(min(round((pos[0]-(ini_aperture+0.003)*sin(yaw) - workspace_limits_raw[0][0])/heightmap_resolution), 400-1), 0))
            pix_y_prime_enlong = int(max(min(round((pos[1]-(ini_aperture+0.003)*cos(yaw) - workspace_limits_raw[1][0])/heightmap_resolution), 400-1), 0))
            pix_x_mid = int((pix_x+pix_x_prime)/2)
            pix_y_mid = int((pix_y+pix_y_prime)/2)
            color_difference1 = color_heightmap_raw[pix_y_mid, pix_x_mid]-color_heightmap_raw[pix_y_prime, pix_x_prime]
            color_difference2 = color_heightmap_raw[pix_y_mid, pix_x_mid]-color_heightmap_raw[pix_y_prime_enlong, pix_x_prime_enlong]
            from_thumb_to_finger_normal = (np.array([pix_x-pix_x_prime, pix_y-pix_y_prime])/np.linalg.norm(np.array([pix_x-pix_x_prime, pix_y-pix_y_prime]))).tolist()
            perp_from_thumb_to_finger_normal = [-from_thumb_to_finger_normal[1], from_thumb_to_finger_normal[0]]
            mid_a_little_left = [int(max(min(pix_x_mid+10*perp_from_thumb_to_finger_normal[0], 400-1), 0)), int(max(min(pix_y_mid+10*perp_from_thumb_to_finger_normal[1], 400-1), 0))]
            mid_a_little_right = [int(max(min(pix_x_mid-10*perp_from_thumb_to_finger_normal[0], 400-1), 0)), int(max(min(pix_y_mid-10*perp_from_thumb_to_finger_normal[1], 400-1), 0))]
            mid_a_little_left_depth = depth_heightmap_raw[mid_a_little_left[1], mid_a_little_left[0]]
            mid_a_little_right_depth = depth_heightmap_raw[mid_a_little_right[1], mid_a_little_right[0]]
            #print(mid_a_little_left, mid_a_little_right, mid_a_little_left_depth, mid_a_little_right_depth)

            mid_point = [int(pix_x_mid[0]), int(pix_y_mid[1])]
            if mid_point[1]-100<0:
                subimage_top = 0
                subimage_down = 200
            elif mid_point[1]+100>400:
                subimage_top = 200
                subimage_down = 400
            else:
                subimage_top = mid_point[1]-100
                subimage_down = mid_point[1]+100

            if mid_point[0]-100<0:
                subimage_left = 0
                subimage_right = 200
            elif mid_point[0]+100>400:
                subimage_left = 200
                subimage_right = 400
            else:
                subimage_left = mid_point[0]-100
                subimage_right = mid_point[0]+100

            sub_rgbd_image = rgbd_image[subimage_top: subimage_down, subimage_left: subimage_right][np.newaxis, :, :, :]
            input_channel_sub_rgbd_image = np.concatenate(([sub_rgbd_image]*9), axis=0)
            finger_position_in_image = (np.array([pix_x, pix_y]) - np.array([subimage_left, subimage_top])).tolist()
            thumb_position_in_image = (np.array([pix_x_prime, pix_y_prime]) - np.array([subimage_left, subimage_top])).tolist()
            input_channel_vector_list = []
            for pitch in [60, 53, 45]:
                for roll in [-10, 0, 10]:
                    input_channel_vector_list.append(finger_position_in_image+thumb_position_in_image+[pitch]+[roll])
            input_channel_vector = np.array(input_channel_vector_list)
            with torch.no_grad():
                net_pitch_roll.eval()
                predicted_score_pitch_roll = np.ravel(net(input_channel_sub_rgbd_image, input_channel_vector).cpu().numpy())
            torch.cuda.empty_cache()
            optimal_pitch_roll_index = np.argmax(predicted_score_pitch_roll)
            if int(int(optimal_pitch_roll_index)/3)==0:
                pitch = 60
            elif int(int(optimal_pitch_roll_index)/3)==1:
                pitch = 53
            elif int(int(optimal_pitch_roll_index)/3)==2:
                pitch = 45
            if int(int(optimal_pitch_roll_index)%3)==0:
                roll = -10
            elif int(int(optimal_pitch_roll_index)%3)==1:
                roll = 0
            elif int(int(optimal_pitch_roll_index)%3)==2:
                roll = 10
            print('pos: ', pos, ' yaw: ', yaw, ' aperture: ', ini_aperture, 'pitch: ', pitch, ' roll: ', roll)
            if robot.collision_check_scooping(pos, yaw, ini_aperture, theta=pitch*pi/180, roll=roll*pi/180)==True:
                continue
            cv2.arrowedLine(copy_color_heightmap, (pix_x_prime, pix_y_prime), (pix_x, pix_y), (0, 0, 255), 2)
            test_image = cv2.flip(copy_color_heightmap, 1)
            f = plt.figure(1)
            plt.imshow(test_image[:,:,[2,1,0]])
            plt.rcParams["keymap.quit"] = 'enter'
            plt.show()
            whether_collision = input("Whether collision? (y or n)")
            if whether_collision=='y':
                continue

            if yaw>0:
                pos=[pos[0]+0.001*sin(yaw), pos[1]+0.001*cos(yaw), pos[2]]
                pass
            grasp_success = robot.exe_scoop(pos, yaw, ini_aperture, thumb_extend=0.003, theta = pitch*pi/180, roll = roll*pi/180) #key 0.015 #Acrylic 0.008 #Go stone 0.003
            total_index += 1
            break

if __name__ == '__main__':
    main()
