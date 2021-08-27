import numpy as np
import math
import copy
import pybullet as p
#from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped
#import rospy

R_h_ee_tool = np.matrix('1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0; 0.0 0.0 0.0 1.0')

def set_tool(tool_x, tool_y, tool_z):
    global R_h_ee_tool

    R_h_ee_tool[0, 3] = tool_x
    R_h_ee_tool[1, 3] = tool_y
    R_h_ee_tool[2, 3] = tool_z

def calc_waypoints_ARC(init_pose, center_point, axis, total_angle):
  
    if total_angle >= 0:
        direction = 1
    else:
        direction = -1

    way_points = []
  
    #first way point is the start point  
    way_points.append(copy.deepcopy(init_pose));

    #calculate the rotation matrix about input 'axis' for 5 degree
    step_angle = direction * 2.0 / 180 * 3.1415926
    current_angle = step_angle
    total_angle = total_angle / 180.0 * 3.1415926

    v_init = np.cross(center_point, axis)
    w_init = axis
    R_h_transform = calc_R_h_from_ksi(w_init, v_init, step_angle);

    #calcluate the matrix form of start point's position and orientation
    current_R_h = get_R_h_from_pose(init_pose);

    while (direction * current_angle <= direction * total_angle):
        #calculate the matrix form of next point after rotating about axis for 5 degree
        current_R_h = R_h_transform * current_R_h 

        #get position and orientation variable from matrix form
        new_target = get_pose_from_R_h(current_R_h)

        #add new waypoint to the way_points list
        way_points.append(new_target)

        #update current rotation angle 
        current_angle += step_angle


    # output is the waypoint list of the ARC trajectory  
    return way_points


def arc_get_tool_position(init_pose):
    #calculate center_point
    Frame_base_h = np.matrix('1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0')

    #calcluate the matrix form of start point's position and orientation
    current_R_h = get_R_h_from_pose(init_pose);

    Frame_tool_h = current_R_h * R_h_ee_tool * Frame_base_h

    center_point = [Frame_tool_h[0, 3], Frame_tool_h[1, 3], Frame_tool_h[2, 3]]

    return center_point

def calc_waypoints_tool_rotate(init_pose, input_axis, total_angle):
    global R_h_ee_tool

    if total_angle >= 0:
        direction = 1
    else:
        direction = -1

    way_points = []

    #first way point is the start point  
    way_points.append([init_pose[0], init_pose[1]]);


    #calculate center_point
    Frame_base_h = np.matrix('1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0')

    #calcluate the matrix form of start point's position and orientation
    current_R_h = get_R_h_from_pose(init_pose);

    Frame_tool_h = current_R_h * R_h_ee_tool * Frame_base_h

    center_point = [Frame_tool_h[0, 3], Frame_tool_h[1, 3], Frame_tool_h[2, 3]]


    axis = []

    if input_axis == 'x':
        axis = [Frame_tool_h[0, 0], Frame_tool_h[1, 0], Frame_tool_h[2, 0]]
    elif input_axis == 'y':
        axis = [Frame_tool_h[0, 1], Frame_tool_h[1, 1], Frame_tool_h[2, 1]]
    elif input_axis == 'z':
        axis = [Frame_tool_h[0, 2], Frame_tool_h[1, 2], Frame_tool_h[2, 2]]
    else:
        print('Error! input axis should be x, y, z')
        return way_points


    #calculate the rotation matrix about input 'axis' for 2 degree
    step_angle = direction * 2.0 / 180 * 3.1415926
    current_angle = step_angle
    total_angle = total_angle / 180.0 * 3.1415926

    v_init = np.cross(center_point, axis)
    w_init = axis
    R_h_transform = calc_R_h_from_ksi(w_init, v_init, step_angle);

    #calcluate the matrix form of start point's position and orientation
    current_R_h = get_R_h_from_pose(init_pose);

    while (direction * current_angle <= direction * total_angle):
        #calculate the matrix form of next point after rotating about axis for 5 degree
        current_R_h = R_h_transform * current_R_h 

        #get position and orientation variable from matrix form
        new_target = get_pose_from_R_h(current_R_h)
        #print("calc arc tool", new_target)

        #add new waypoint to the way_points list
        way_points.append(new_target)

        #update current rotation angle 
        current_angle += step_angle

        # output is the waypoint list of the ARC trajectory  


    return way_points



#solve rotation matrix using the formula to solve e^(ksi * theta)
def calc_R_h_from_ksi(w_init, v_init, theta):
  
    #initialize variables
    w_cross_v_m = np.matrix('0.0; 0.0; 0.0')
    R_ksi_1st = np.matrix('0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0')
    R_w_1st = np.matrix('0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0')
    R_w_hat = np.matrix('0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0')
    R_I = np.matrix('1.0 0.0 0.0; 0.0 1.0 0.0; 0 0 1')
    w_init_v = np.array([w_init[0], w_init[1], w_init[2]])
    v_init_v = np.array([v_init[0], v_init[1], v_init[2]])
    w_cross_v = np.cross(w_init_v, v_init_v)  
    w_cross_v_m[0, 0] = w_cross_v[0]
    w_cross_v_m[1, 0] = w_cross_v[1]
    w_cross_v_m[2, 0] = w_cross_v[2]

    # Solve the rotation matrix part
    theta_1st = theta
    R_w_hat = np.matrix([[0,             -w_init[2],  w_init[1]], [w_init[2],  0,              -w_init[0]], [-w_init[1], w_init[0],   0] ])
    R_w_1st = R_I + R_w_hat * math.sin(theta_1st) + R_w_hat * R_w_hat * (1 - math.cos(theta_1st));

    R_ksi_1st[0, 0] =  R_w_1st[0, 0]
    R_ksi_1st[0, 1] =  R_w_1st[0, 1]
    R_ksi_1st[0, 2] =  R_w_1st[0, 2]

    R_ksi_1st[1, 0] =  R_w_1st[1, 0]
    R_ksi_1st[1, 1] =  R_w_1st[1, 1]
    R_ksi_1st[1, 2] =  R_w_1st[1, 2]

    R_ksi_1st[2, 0] =  R_w_1st[2, 0]
    R_ksi_1st[2, 1] =  R_w_1st[2, 1]
    R_ksi_1st[2, 2] =  R_w_1st[2, 2]

    # Solve the translation part
    temp = (R_I - R_w_1st) * w_cross_v_m;
    R_ksi_1st[0, 3] = temp[0, 0]
    R_ksi_1st[1, 3] = temp[1, 0]
    R_ksi_1st[2, 3] = temp[2, 0]
    R_ksi_1st[3, 3] = 1

    # output is the trasformation matrix
    return R_ksi_1st


# get position and orientation variables from a transformation matrix
def get_pose_from_R_h(R_h):
    pose_target = []
    new_pos = np.squeeze(np.asarray(R_h[:3,3]))
    pose_target.append(tuple(new_pos))
    new_e = rotm2euler(R_h[:3,:3])
    new_q = p.getQuaternionFromEuler(new_e.tolist())
    pose_target.append(new_q)
    return pose_target
    

# get transformation matrix from position and orientation variables 
def get_R_h_from_pose(pose):
    pos = list(pose[0])
    ori = list(pose[1])
    #init_q = []
    #init_q.append(pose.orientation.x)
    #init_q.append(pose.orientation.y)
    #init_q.append(pose.orientation.z)
    #init_q.append(pose.orientation.w)
    init_e = p.getEulerFromQuaternion(ori)
    init_R = euler2rotm(init_e)
    init_R_h = np.matrix('0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0')
    init_R_h[:3,:3] = init_R
    #init_R_h = tf.quaternion_matrix(init_q)
    init_R_h[0, 3] = pos[0]
    init_R_h[1, 3] = pos[1] 
    init_R_h[2, 3] = pos[2]
    return init_R_h

def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R) :
 
    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R