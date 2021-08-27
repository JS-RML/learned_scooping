import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import os
import numpy as np
import cv2   #4.1.2
from math import *
import matplotlib.pyplot as plt


def def_func(x):
    pass

def draw_shape(event, x, y, flags, param):
    global ix, iy, rx, ry, drawing, showing_finger_line, mode_index, color_index, label_vis, color_heightmap_bgr, depth_heightmap_bgr, current_background, label_array
    color_index = cv2.getTrackbarPos('Color (White: 0, Red: 1, Green: 2)', 'label_vis')
    if color_index == 0:
        color = [255,255,255]
    elif color_index == 1:
        color = [0,0,255]
    elif color_index == 2:
        color = [69, 139, 0]

    mode_index = cv2.getTrackbarPos('Shape (Rectangle: 0, Circle: 1, Curve: 2)', 'label_vis')
    thickness = cv2.getTrackbarPos('Thickness', 'label_vis')

    label_vis_3 = label_vis[:, 400:600]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x,y

    elif event == cv2.EVENT_MBUTTONDOWN:
        current_background = label_vis.copy()
        showing_finger_line = True
        rx, ry = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode_index == 0:
                if ix>=0 and ix<200:
                    cv2.rectangle(label_vis[:, 400:600], (ix, iy), (max(min(x, 200), 0),max(min(y, 200), 0)), color, -1)
                elif ix>=200 and ix<400:
                    cv2.rectangle(label_vis[:, 400:600], (ix-200, iy), (max(min(x-200, 200), 0),max(min(y, 200), 0)), color, -1)
                elif ix>=400 and ix<600:
                    cv2.rectangle(label_vis[:, 400:600], (ix-400, iy), (max(min(x-400, 200), 0),max(min(y, 200), 0)), color, -1)        
            elif mode_index == 1:
                if ix>=0 and ix<200:
                    cv2.circle(label_vis[:, 400:600], (ix, iy), round(sqrt((max(min(x, 200), 0)-ix)**2+(max(min(y, 200), 0)-iy)**2)), color, -1)
                elif ix>=200 and ix<400:
                    cv2.circle(label_vis[:, 400:600], (ix-200, iy), round(sqrt((max(min(x-200, 200), 0)-ix+200)**2+(max(min(y, 200), 0)-iy)**2)), color, -1)
                elif ix>=400 and ix<600:
                    cv2.circle(label_vis[:, 400:600], (ix-400, iy), round(sqrt((max(min(x-400, 200), 0)-ix+400)**2+(max(min(y, 200), 0)-iy)**2)), color, -1)
            elif mode_index == 2:
                if ix>=0 and ix<200:
                    cv2.circle(label_vis[:, 400:600], (x, y), thickness, color, -1)
                elif ix>=200 and ix<400:
                    cv2.circle(label_vis[:, 400:600], (max(min(x-200, 200), 0), y), thickness, color, -1)
                elif ix>=400 and ix<600:
                    cv2.circle(label_vis[:, 400:600], (max(min(x-400, 200), 0), y), thickness, color, -1)
            label_vis[:, 400:600][label_array==0]=np.array([0,0,255])
            label_vis_3_b = label_vis_3[:,:,0].copy()
            label_vis_3_g = label_vis_3[:,:,1].copy()
            label_vis_3_r = label_vis_3[:,:,2].copy()
            label_vis_3_temp = label_vis_3_b*(256*256) + label_vis_3_g*256 + label_vis_3_r
            label_vis_1 = color_heightmap_bgr.copy()
            label_vis_1[label_vis_3_temp==255]=np.array([0,0,255])
            label_vis_1[label_vis_3_temp==69*256*256+139*256]=np.array([69, 139, 0])
            label_vis[:, 0:200] = label_vis_1
            label_vis_2 = depth_heightmap_bgr.copy()
            label_vis_2[label_vis_3_temp==255]=np.array([0,0,255])
            label_vis_2[label_vis_3_temp==69*256*256+139*256]=np.array([69, 139, 0])
            label_vis[:, 200:400] = label_vis_2
            if 69*256*256+139*256 in label_vis_3_temp:
                label_vis[:, 600:605] = np.array([[[69, 139, 0] for k0 in range(5)] for k1 in range(200)]).astype(np.uint8)
            elif 255 in label_vis_3_temp and label_vis_3.astype(np.uint8).tolist() != label_array_color.astype(np.uint8).tolist():
                label_vis[:, 600:605] = np.array([[[0,0,255] for k0 in range(5)] for k1 in range(200)]).astype(np.uint8)
            else:
                label_vis[:, 600:605] = np.array([[[0,255,255] for k0 in range(5)] for k1 in range(200)]).astype(np.uint8)
        elif showing_finger_line == True:
            current_background_copy = current_background.copy()
            if rx>=0 and rx<200:
                cv2.line(current_background_copy[:, 0:200], (int(round(max(min(x-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
                cv2.line(current_background_copy[:, 200:400], (int(round(max(min(x-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
                cv2.line(current_background_copy[:, 400:600], (int(round(max(min(x-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
            elif rx>=200 and rx<400:
                cv2.line(current_background_copy[:, 0:200], (int(round(max(min(x-200-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x-200+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
                cv2.line(current_background_copy[:, 200:400], (int(round(max(min(x-200-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x-200+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
                cv2.line(current_background_copy[:, 400:600], (int(round(max(min(x-200-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x-200+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
            elif rx>=400 and rx<600:
                cv2.line(current_background_copy[:, 0:200], (int(round(max(min(x-400-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x-400+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
                cv2.line(current_background_copy[:, 200:400], (int(round(max(min(x-400-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x-400+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2)
                cv2.line(current_background_copy[:, 400:600], (int(round(max(min(x-400-0.01/(0.0013/4), 200), 0))), y), (int(round(max(min(x-400+0.01/(0.0013/4), 200), 0))), y), [205,90,106], 2) 
            label_vis = current_background_copy

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    elif event == cv2.EVENT_MBUTTONUP:
        showing_finger_line = False
        label_vis = current_background.copy()

def main():
    global mode_index, color_index, drawing, showing_finger_line, ix, iy, rx, ry, label_vis, color_heightmap_bgr, depth_heightmap_bgr, label_array_color, label_array
    label_figure_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao"
    #label_figure_path = input("请输入图片数据集的地址")
    #input_data_index = int(input("请输入开始标定的图片的序号"))
    input_data_index_index = 0
    input_data_index_set = [6021, 6089, 6108, 6113, 6121, 6175, 6212, 6219, 6220, 6242, 6272, 6281, 6299, 6328, 6343, 6351, 6381, 6400, 6407, 6469, 6573, 6582, 6583, 6652, 6653, 6745, 6807, 6808, 6817, 6843, 6851, 6868, 6897, 6905, 6917, 6947, 6979, 6980]
    while True:
        input_data_index = input_data_index_set[input_data_index_index]
        print('图片序号', input_data_index)
        color_heightmap_array = cv2.imread(label_figure_path + "/figure/rgb_image_"+str(input_data_index)+".png")[:, :, [2,1,0]]
        depth_heightmap_array = cv2.imread(label_figure_path + "/figure/depth_image_"+str(input_data_index)+".png", -1)
        label_array = cv2.imread(label_figure_path + "/figure/label_initial_"+str(input_data_index)+".png", -1).astype(np.float32)
        label_array_color = np.ones((200,200,3))*255
        label_array_color[label_array==0]=np.array([0,0,255])
        label_array_color[label_array==128]=np.array([69, 139, 0])
        label_array_color = label_array_color.astype(np.uint8)
        color_heightmap_bgr = cv2.cvtColor(color_heightmap_array, cv2.COLOR_RGB2BGR)
        label_vis_1 = color_heightmap_bgr.copy()
        label_vis_1[label_array==0]=np.array([0,0,255])
        label_vis_1[label_array==128]=np.array([69, 139, 0])
        depth_heightmap_bgr = cv2.cvtColor(depth_heightmap_array, cv2.COLOR_GRAY2BGR)
        label_vis_2 = depth_heightmap_bgr.copy()
        label_vis_2[label_array==0]=np.array([0,0,255])
        label_vis_2[label_array==128]=np.array([69, 139, 0])
        if 128 in label_array:
            signal_color = np.array([[[69, 139, 0] for k0 in range(5)] for k1 in range(200)]).astype(np.uint8)
        else:
            signal_color = np.array([[[0,255,255] for k0 in range(5)] for k1 in range(200)]).astype(np.uint8)
        label_vis = np.hstack([label_vis_1, label_vis_2, label_array_color, signal_color])

        drawing = False
        showing_finger_line = False
        mode_index = 0
        color_index=0
        ix, iy = -1, -1
                
        cv2.namedWindow('label_vis', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('label_vis', 1000, 500)
        cv2.createTrackbar('Color (White: 0, Red: 1, Green: 2)','label_vis',0,2,def_func)
        cv2.setTrackbarPos('Color (White: 0, Red: 1, Green: 2)', 'label_vis', 1)
        cv2.createTrackbar('Shape (Rectangle: 0, Circle: 1, Curve: 2)','label_vis',0,2,def_func)
        cv2.setTrackbarPos('Shape (Rectangle: 0, Circle: 1, Curve: 2)', 'label_vis', 2)
        cv2.createTrackbar('Thickness','label_vis',1,10,def_func)
        cv2.setTrackbarPos('Thickness','label_vis',15)
        cv2.setMouseCallback('label_vis', draw_shape)


        while(True):
            cv2.imshow('label_vis',label_vis)
            
            k = cv2.waitKey(1)&0xff
            if k == ord('c'):
                print('you typed key c: shift between color')
                if color_index==0:
                    color_index = 1
                    cv2.setTrackbarPos('Color (White: 0, Red: 1, Green: 2)', 'label_vis', 1)
                elif color_index==1:
                    color_index = 2
                    cv2.setTrackbarPos('Color (White: 0, Red: 1, Green: 2)', 'label_vis', 2)
                elif color_index==2:
                    color_index = 0
                    cv2.setTrackbarPos('Color (White: 0, Red: 1, Green: 2)', 'label_vis', 0)
            elif k == ord('m'):
                print('you typed key m: shift between rectangle and circle')
                if mode_index==0:
                    mode_index = 1
                    cv2.setTrackbarPos('Shape (Rectangle: 0, Circle: 1, Curve: 2)', 'label_vis', 1)
                elif mode_index==1:
                    mode_index = 2
                    cv2.setTrackbarPos('Shape (Rectangle: 0, Circle: 1, Curve: 2)', 'label_vis', 2)
                elif mode_index==2:
                    mode_index = 0
                    cv2.setTrackbarPos('Shape (Rectangle: 0, Circle: 1, Curve: 2)', 'label_vis', 0)
            elif k == 27:
                control_signal = "exit"
                cv2.destroyAllWindows()
                break
            elif k==61 or k==43 or k==13:  #only for the key on my computer
                control_signal = "next"
                cv2.destroyAllWindows()
                break
            elif k==45:  #only for the key on my computer
                control_signal = "last"
                cv2.destroyAllWindows()
                break
        label_array_new = np.ones((200,200))*255
        label_color_new = label_vis[:, 400:600].copy()
        label_color_new_b = label_color_new[:,:,0].copy()
        label_color_new_g = label_color_new[:,:,1].copy()
        label_color_new_r = label_color_new[:,:,2].copy()
        label_color_new_temp = label_color_new_b*(256*256) + label_color_new_g*256 + label_color_new_r
        label_array_new[label_color_new_temp==255]=0
        label_array_new[label_color_new_temp==69*256*256+139*256]=128
        label_array_new = label_array_new.astype(np.uint8)
        np.save(label_figure_path + "/label" + "/label_"+str(input_data_index)+".npy", label_array_new)
        if control_signal == "exit":
            break
        elif control_signal == "next":
            input_data_index_index += 1
            continue
        elif control_signal == "last": 
            input_data_index_index -= 1 
            continue     

    cv2.destroyAllWindows()

if __name__ == '__main__':
    #main()
    '''
    for i in  range(21, 1000):
        print(i)
        left_fig = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_Acrylic/rgb_image_"+str(i)+".png")[:, :, [2,1,0]]
        plt.subplot(1, 2, 1)
        plt.imshow(left_fig)
        right_fig = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_Acrylic/label_"+str(i)+".npy")
        plt.subplot(1, 2, 2)
        plt.imshow(right_fig, cmap='gray')
        plt.rcParams["keymap.quit"] = 'enter'
        plt.show()
    '''
    for index in range(1, 12001):
        print(index)
        current_array = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_Acrylic/label_"+str(index)+".npy").astype(np.uint8)
        if 'label_data_array_Acrylic_taobao' not in dir():
            label_data_array_Acrylic_taobao = current_array[np.newaxis, :, :]
        else:
            label_data_array_Acrylic_taobao = np.concatenate((label_data_array_Acrylic_taobao, current_array[np.newaxis, :, :]), axis = 0)
    print(label_data_array_Acrylic_taobao.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_Acrylic_taobao.npy", label_data_array_Acrylic_taobao)

    '''
    for index in range(0, 1500):
        print(index)
        current_array = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_Go_stone/Go_stone_label_"+str(index)+".npy").astype(np.uint8)
        if 'label_data_array_Go_stone_taobao' not in dir():
            label_data_array_Go_stone_taobao = current_array[np.newaxis, :, :]
        else:
            label_data_array_Go_stone_taobao = np.concatenate((label_data_array_Go_stone_taobao, current_array[np.newaxis, :, :]), axis = 0)
    print(label_data_array_Go_stone_taobao.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_Go_stone_taobao.npy", label_data_array_Go_stone_taobao)
    '''


    label_1 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_domino_taobao.npy")
    print('1')
    label_2 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_Acrylic_taobao.npy")
    print('2')
    label_3 = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_Go_stone_taobao.npy")
    print('3')
    label_20210715 = np.concatenate((np.concatenate((label_1, label_2), axis = 0), label_3), axis = 0)
    print(label_20210715.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_20210715.npy", label_20210715)

    '''
    concatenate_list_domino = []
    for i in range(1, 8625):
        print('domino', i)
        current_figure_rgb = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_domino/rgb_image_"+str(i)+".png")[:, :, [2,1,0]][np.newaxis, :, :, :].astype(np.uint8)
        current_figure_depth = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_domino/depth_image_"+str(i)+".png", -1)[:, :, np.newaxis][np.newaxis, :, :, :].astype(np.uint8)
        current_figure_rgbd = np.concatenate((current_figure_rgb, current_figure_depth), axis = 3)
        concatenate_list_domino.append(current_figure_rgbd)
    input_data_domino_rgb = np.concatenate(concatenate_list_domino, axis = 0)
    print(input_data_domino_rgb.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_domino_20210703.npy", input_data_domino_rgb)
    '''
    '''
    concatenate_list_Acrylic = []
    for i in range(1, 12001):
        print('Acrylic', i)
        current_figure_rgb = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_Acrylic/rgb_image_"+str(i)+".png")[:, :, [2,1,0]][np.newaxis, :, :, :].astype(np.uint8)
        current_figure_depth = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_Acrylic/depth_image_"+str(i)+".png", -1)[:, :, np.newaxis][np.newaxis, :, :, :].astype(np.uint8)
        current_figure_rgbd = np.concatenate((current_figure_rgb, current_figure_depth), axis = 3)
        concatenate_list_Acrylic.append(current_figure_rgbd)
    input_data_Acrylic_rgb = np.concatenate(concatenate_list_Acrylic, axis = 0)
    print(input_data_Acrylic_rgb.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_Acrylic_20210715.npy", input_data_Acrylic_rgb)
    '''
    '''
    concatenate_list_Go_stone = []
    for i in range(0, 1500):
        print('Go_stone', i)
        current_figure_rgb = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_Go_stone/rgb_image_"+str(i)+".png")[:, :, [2,1,0]][np.newaxis, :, :, :].astype(np.uint8)
        current_figure_depth = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/figure_Go_stone/depth_image_"+str(i)+".png", -1)[:, :, np.newaxis][np.newaxis, :, :, :].astype(np.uint8)
        current_figure_rgbd = np.concatenate((current_figure_rgb, current_figure_depth), axis = 3)
        concatenate_list_Go_stone.append(current_figure_rgbd)
    input_data_Go_stone_rgb = np.concatenate(concatenate_list_Go_stone, axis = 0)
    print(input_data_Go_stone_rgb.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_Go_stone_20210709.npy", input_data_Go_stone_rgb)
    '''
    '''
    input_domino = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_domino_20210703.npy") 
    print('1')
    input_Acrylic = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_Acrylic_20210715.npy")
    print('2')
    input_Go_stone = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_Go_stone_20210709.npy")
    print('3')
    input_data = np.concatenate((input_domino, input_Acrylic, input_Go_stone), axis=0)
    print(input_data.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_20210715.npy", input_data)
    '''
    
    
    '''
    input_data = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/input_data_array_20210703.npy")
    label_data = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/label_taobao/label_data_array_20210703.npy")
    for i in  range(9000, 10000):
        print(i)
        left_fig = input_data[i, :, :, 0:3]
        plt.subplot(1, 2, 1)
        plt.imshow(left_fig)
        right_fig = label_data[i, :, :]
        plt.subplot(1, 2, 2)
        plt.imshow(right_fig, cmap='gray')
        plt.rcParams["keymap.quit"] = 'enter'
        plt.show()
    '''
    
    
