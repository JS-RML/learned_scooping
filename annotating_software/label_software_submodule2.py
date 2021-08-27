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
import math3d as m3d
#from Mask_R_CNN_multi_category_heightmap import Mask_R_CNN_stone
import skimage.io
from math import *
import scipy
from PIL import Image
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LinearRing


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def def_func(x):
    pass

def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, mode_index, color_index, label_vis, color_heightmap_bgr, depth_heightmap_bgr, finger_position, thumb_contact_position_new
    color_index = cv2.getTrackbarPos('Color (White: 0, Red: 1, Green: 2)', 'label_vis')
    if color_index == 0:
        color = [255,255,255]
    elif color_index == 1:
        color = [0,0,255]
    elif color_index == 2:
        color = [69, 139, 0]

    mode_index = cv2.getTrackbarPos('Shape (Rectangle: 0, Circle: 1)', 'label_vis')
    if mode_index == 0:
        mode=True
    else:
        mode=False

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    #elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            label_vis_1 = color_heightmap_bgr.copy()
            cv2.circle(label_vis_1, (finger_position[0], finger_position[1]), 2, [0,0,255], -1)
            cv2.line(label_vis_1, (int(round(finger_position[0]-0.01/(0.0013/4))), finger_position[1]), (int(round(finger_position[0]+0.01/(0.0013/4))), finger_position[1]), [0,0,255], 2)
            cv2.circle(label_vis_1, (finger_position[0], y), 2, [69, 139, 0], -1)
            cv2.line(label_vis_1, (int(round(finger_position[0]-0.01/(0.0013/4))), y), (int(round(finger_position[0]+0.01/(0.0013/4))), y), [69, 139, 0], 2)
            label_vis[:, 0:200] = label_vis_1
            label_vis_2 = depth_heightmap_bgr.copy()
            cv2.circle(label_vis_2, (finger_position[0], finger_position[1]), 2, [0,0,255], -1)
            cv2.line(label_vis_2, (int(round(finger_position[0]-0.01/(0.0013/4))), finger_position[1]), (int(round(finger_position[0]+0.01/(0.0013/4))), finger_position[1]), [0,0,255], 2)
            cv2.circle(label_vis_2, (finger_position[0], y), 2, [69, 139, 0], -1)
            cv2.line(label_vis_2, (int(round(finger_position[0]-0.01/(0.0013/4))), y), (int(round(finger_position[0]+0.01/(0.0013/4))), y), [69, 139, 0], 2)
            label_vis[:, 200:400] = label_vis_2
            thumb_contact_position_new = np.array([y])
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def main():
    global mode_index, color_index, drawing, label_vis, color_heightmap_bgr, depth_heightmap_bgr, finger_position, thumb_contact_position_new
    input_rgbd_array = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_Acrylic_20210709.npy")
    input_data_num = input_rgbd_array.shape[0]
    input_data_index = 9029
    index = 2399
    while True:
        if input_data_index<0 or input_data_index>=input_data_num:
            break
        print('input_data_index', input_data_index)
        label_of_submodule_1 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_Acrylic/label_"+str(input_data_index+1)+".npy")
        target_finger_position_set = np.argwhere(label_of_submodule_1==128).tolist()
        target_finger_position_already_iterated = []
        for target_finger_position in target_finger_position_set:
            for already_target_finger_position in target_finger_position_already_iterated:
                if Point(target_finger_position).distance(Point(already_target_finger_position))<15:
                    break
            else:
                print('index', index)
                thumb_contact_position_new = np.array([250])
                target_finger_position_already_iterated.append(target_finger_position)
                color_heightmap_array = input_rgbd_array[input_data_index,:,:,0:3]
                depth_heightmap_array = input_rgbd_array[input_data_index,:,:,3:4]
                finger_position = target_finger_position[::-1]
                color_heightmap_bgr = cv2.cvtColor(color_heightmap_array, cv2.COLOR_RGB2BGR)
                label_vis_1 = color_heightmap_bgr.copy()
                #label_vis_1 = cv2.addWeighted(cv2.cvtColor(color_heightmap_array, cv2.COLOR_RGB2BGR), 0.8,label_array_color[:,:,[2,1,0]].astype(np.uint8), 0.2, 0)
                #prediction_vis = (0.8*cv2.cvtColor(color_heightmap_array, cv2.COLOR_RGB2BGR) + 0.2*label_array_color[:,:,[2,1,0]]).astype(np.uint8)
                cv2.circle(label_vis_1, (finger_position[0], finger_position[1]), 2, [0,0,255], -1)
                cv2.line(label_vis_1, (int(round(finger_position[0]-0.01/(0.0013/4))), finger_position[1]), (int(round(finger_position[0]+0.01/(0.0013/4))), finger_position[1]), [0,0,255], 2)
                depth_heightmap_bgr = cv2.cvtColor(depth_heightmap_array, cv2.COLOR_GRAY2BGR)
                label_vis_2 = depth_heightmap_bgr.copy()
                cv2.circle(label_vis_2, (finger_position[0], finger_position[1]), 2, [0,0,255], -1)
                cv2.line(label_vis_2, (int(round(finger_position[0]-0.01/(0.0013/4))), finger_position[1]), (int(round(finger_position[0]+0.01/(0.0013/4))), finger_position[1]), [0,0,255], 2)
                label_vis = np.hstack([label_vis_1, label_vis_2])

                drawing = False
                        
                cv2.namedWindow('label_vis', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow('label_vis', 800, 500)
                #cv2.createButton("Red color",def_func,None,cv2.QT_PUSH_BUTTON,1)
                cv2.setMouseCallback('label_vis', draw_shape)


                while(True):
                    cv2.imshow('label_vis',label_vis)
                    
                    k = cv2.waitKey(1)&0xff
                    if k == 27:
                        control_signal = "exit"
                        cv2.destroyAllWindows()
                        break
                    elif k==61 or k==43 or k==13:  #only for the key on my computer
                        control_signal = "next"
                        cv2.destroyAllWindows()
                        break
                thumb_contact_position_new = thumb_contact_position_new.astype(np.uint8)
                print(thumb_contact_position_new)
                if thumb_contact_position_new[0]<=200:
                    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/RGBD_figure/Acrylic_"+str(index)+".npy", input_rgbd_array[input_data_index,:,:,:])
                    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/finger_contact_position/Acrylic_"+str(index)+".npy", np.array([finger_position]))
                    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/thumb_position_modified_20210705/Acrylic_"+str(index)+".npy", thumb_contact_position_new)
                    #aaa = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_20210616/thumb_contact_position_domino/RGBD_figure/domino_"+str(index)+".npy")
                    #bbb = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_20210616/thumb_contact_position_domino/finger_contact_position/domino_"+str(index)+".npy")
                    #ccc = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_20210616/thumb_contact_position_domino/thumb_contact_position/domino_"+str(index)+".npy")
                    #print(aaa.shape)
                    #print(bbb.shape)
                    #print(ccc.shape)

                else:
                    continue
                #plt.imshow(label_array_new, cmap='gray')
                #plt.savefig("/home/terry/catkin_ws/src/dg_learning_real_one_net/gray.png")
                #plt.show() 
                if control_signal == "exit":
                    os._exit()
                elif control_signal == "next":
                    index += 1
                    continue
        input_data_index += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    '''
    #main()
    for index in range(1, 701):
        print(index)
        RGBD_figure = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/RGBD_figure_complement/Acrylic_"+str(index)+".npy")
        if 'input_data_array_submodule2_image_Acrylic' not in dir():
            input_data_array_submodule2_image_Acrylic = RGBD_figure[np.newaxis, :, :]
        else:
            input_data_array_submodule2_image_Acrylic = np.concatenate((input_data_array_submodule2_image_Acrylic, RGBD_figure[np.newaxis, :, :]), axis=0)

        finger_contact_position = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/finger_contact_position_complement/Acrylic_"+str(index)+".npy")
        if 'input_data_array_submodule2_finger_position_Acrylic' not in dir():
            input_data_array_submodule2_finger_position_Acrylic = finger_contact_position
        else:
            input_data_array_submodule2_finger_position_Acrylic = np.concatenate((input_data_array_submodule2_finger_position_Acrylic, finger_contact_position), axis=0)

        thumb_contact_position = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/thumb_contact_position_complement/Acrylic_"+str(index)+".npy")
        if 'input_data_array_submodule2_thumb_position_Acrylic' not in dir():
            input_data_array_submodule2_thumb_position_Acrylic = thumb_contact_position[np.newaxis, :]
        else:
            input_data_array_submodule2_thumb_position_Acrylic = np.concatenate((input_data_array_submodule2_thumb_position_Acrylic, thumb_contact_position[np.newaxis, :]), axis=0)


    print(input_data_array_submodule2_image_Acrylic.shape)
    print(input_data_array_submodule2_finger_position_Acrylic.shape)
    print(input_data_array_submodule2_thumb_position_Acrylic.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_image_Acrylic_20210711.npy", input_data_array_submodule2_image_Acrylic)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_finger_position_Acrylic_20210711.npy", input_data_array_submodule2_finger_position_Acrylic)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_thumb_position_Acrylic_20210711.npy", input_data_array_submodule2_thumb_position_Acrylic)

    '''
    input_data_array_submodule2_image_domino = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_domino/input_data_array_submodule2_image_domino.npy")
    input_data_array_submodule2_image_Acrylic = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_image_Acrylic_20210709.npy")
    input_data_array_submodule2_image_Acrylic_2 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_image_Acrylic_20210711.npy")
    input_data_array_submodule2_image_Go_stone = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Go_stone/input_data_array_submodule2_image_Go_stone_20210709.npy")
    input_data_array_submodule2_image = np.concatenate((input_data_array_submodule2_image_domino, input_data_array_submodule2_image_Acrylic, input_data_array_submodule2_image_Acrylic_2, input_data_array_submodule2_image_Go_stone), axis=0)
    print(input_data_array_submodule2_image.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_20210711/input_data_array_submodule2_image_20210711.npy", input_data_array_submodule2_image)

    input_data_array_submodule2_finger_position_domino = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_domino/input_data_array_submodule2_finger_position_domino.npy")
    input_data_array_submodule2_finger_position_Acrylic = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_finger_position_Acrylic_20210709.npy")
    input_data_array_submodule2_finger_position_Acrylic_2 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_finger_position_Acrylic_20210711.npy")
    input_data_array_submodule2_finger_position_Go_stone = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Go_stone/input_data_array_submodule2_finger_position_Go_stone_20210709.npy")
    input_data_array_submodule2_finger_position = np.concatenate((input_data_array_submodule2_finger_position_domino, input_data_array_submodule2_finger_position_Acrylic, input_data_array_submodule2_finger_position_Acrylic_2, input_data_array_submodule2_finger_position_Go_stone), axis=0)
    print(input_data_array_submodule2_finger_position.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_20210711/input_data_array_submodule2_finger_position_20210711.npy" ,input_data_array_submodule2_finger_position)

    input_data_array_submodule2_thumb_position_domino = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_domino/input_data_array_submodule2_thumb_position_domino_20210705.npy")
    input_data_array_submodule2_thumb_position_Acrylic = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_thumb_position_Acrylic_20210709.npy")
    input_data_array_submodule2_thumb_position_Acrylic_2 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Acrylic/input_data_array_submodule2_thumb_position_Acrylic_20210711.npy")
    input_data_array_submodule2_thumb_position_Go_stone = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_Go_stone/input_data_array_submodule2_thumb_position_Go_stone_20210709.npy")
    input_data_array_submodule2_thumb_position = np.concatenate((input_data_array_submodule2_thumb_position_domino, input_data_array_submodule2_thumb_position_Acrylic, input_data_array_submodule2_thumb_position_Acrylic_2, input_data_array_submodule2_thumb_position_Go_stone), axis=0)
    print(input_data_array_submodule2_thumb_position.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/thumb_contact_position_20210711/label_data_array_submodule2_thumb_position_20210711.npy",input_data_array_submodule2_thumb_position)
    
