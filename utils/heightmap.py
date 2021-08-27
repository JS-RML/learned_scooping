import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math3d as m3d
import cv2
from scipy import ndimage

import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pcpt_res, dig_res

def get_pointcloud(color_img, depth_img, camera_intrinsics, is_sim):

    depth_img = depth_img # for sim
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution, is_sim=False):

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)
    

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics, is_sim)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(np.array(surface_pts)[:,2])
    surface_pts = surface_pts[sort_z_ind]
    #print('surface_pts', surface_pts)
    color_pts = color_pts[sort_z_ind]
 
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    if heightmap_valid_ind.ndim!=1:
        heightmap_valid_ind = np.array(heightmap_valid_ind)[:,0]
    #print('heightmap_valid_ind', heightmap_valid_ind.ndim, heightmap_valid_ind.shape)
    surface_pts = surface_pts[heightmap_valid_ind,:]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1],1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1],1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1],1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x,0] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x,0] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x,0] = color_pts[:,[2]]
    '''
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x,0] = np.array(color_pts[:,[0]])[:,0]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x,0] = np.array(color_pts[:,[1]])[:,0]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x,0] = np.array(color_pts[:,[2]])[:,0]
    '''

    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    #print("test3.3", color_heightmap.shape)
    #plt.imshow(depth_heightmap[::-1])
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    depth_heightmap+=0.01
    return color_heightmap, depth_heightmap

if __name__ == '__main__':
    color_img = cv2.imread("/home/terry/catkin_ws/src/dg_learning_real/0.jpeg")
    #color_img = scipy.misc.imread("/home/zhekai/tensorflow_proj/Mask_RCNN/samples/stones/JPEGImages/0.jpeg")
    #color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    #depth_img = np.load("/home/terry/catkin_ws/src/dg_learning_real/0.npy")
    depth_img = np.load('/home/terry/catkin_ws/src/dg_learning_real_one_net/picture_20210422/depth_img/1_depth_img.npy')
    print(depth_img)

    cam_intrinsics = np.asarray([[612.0938720703125, 0, 321.8862609863281], [0, 611.785888671875, 238.18316650390625], [0, 0, 1]])

    
    eeTcam = m3d.Transform()
    eeTcam.pos = (0.076173, -0.0934057, 0.0074811)
    eeTcam_e = np.array([-0.4836677963432222, 1.5227704838700455, -0.4651199335909967])
    eeTcam_e = np.array([0, 0, 0])
    eeTcam.orient.rotate_xb(eeTcam_e[0])
    eeTcam.orient.rotate_yb(eeTcam_e[1])
    eeTcam.orient.rotate_zb(eeTcam_e[2])
    baseTee = m3d.Transform()
    baseTee.pos = (-0.23202, 0.62931, 0.40371)
    baseTee.orient = np.array([[-0.01316639,  0.99982403,  0.01336236],
           [ 0.99944137,  0.01356954, -0.03054217],
           [-0.03071812,  0.01295276, -0.99944416]])
    print(eeTcam)

    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.39, -0.26], [0.63, 0.77], [-0.02, 0.3]])
    heightmap_resolution = 0.0005
    baseTcam = np.matmul(baseTee.get_matrix(), eeTcam.get_matrix())
    print("baseTcam", baseTcam)
    '''
    workspace_limits = np.asarray([[-0.39, -0.26], [0.63, 0.77], [-0.02, 0.3]])
    heightmap_resolution = 0.0005
    eeTcam = np.array([[0, -1, 0, 0.142],
                           [1, 0, 0, -0.003],
                           [0, 0, 1, 0.0934057+0.03],
                           [0, 0, 0, 1]])
    baseTee = np.array([[0, 1, 0, 0.05511], [1, 0, 0, 0.54732], [0, 0, -1, 0.37707], [0, 0, 0, 1]])
    baseTcam = np.matmul(baseTee, eeTcam)
    print("baseTcam", baseTcam)
    '''
    color_heightmap, depth_heightmap = get_heightmap(color_img, depth_img, cam_intrinsics, baseTcam, workspace_limits, heightmap_resolution, is_sim=True)
    color_heightmap_rgb = color_heightmap[:, :, [2,1,0]]
    #plt.imshow(color_heightmap[:, :, [2,1,0]])
    #plt.imshow(depth_heightmap)
    #plt.show()

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    print(depth_heightmap.shape)
    xx = np.arange(0,depth_heightmap.shape[1],1)
    yy = np.arange(0,depth_heightmap.shape[0],1)
    X, Y = np.meshgrid(xx, yy)
    #print()
    #xyzs = depth_heightmap

    range_color = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
    ]

    color_heightmap_rgb = color_heightmap_rgb.reshape(-1,3)
    print(color_heightmap_rgb.shape)


    #print(["#"+hex(color_point[0])[2::]+hex(color_point[1])[2::]+hex(color_point[2])[2::] for color_point in color_heightmap_rgb])
    ax.scatter3D(X.ravel(), Y.ravel(), depth_heightmap.ravel(), c=["#"+hex(color_point[0])[2::].zfill(2)+hex(color_point[1])[2::].zfill(2)+hex(color_point[2])[2::].zfill(2) for color_point in color_heightmap_rgb], s=0.5)  # 散点图
    plt.show()

