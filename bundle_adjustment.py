import apriltag
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import glob
import os
import time
from scipy.optimize import minimize, least_squares

from ba_utils import load_intrinsics, get_RT_trans_from_serial, \
    from_world_to_cam_coord, load_point3d_structure, make_x0, \
    fun_all_camera, flatten_3d_structure
from collections import OrderedDict


class BundleAdjustment:
    def __init__(self):
        pass


if __name__ == '__main__':
    data_root = '/home/juni/project/dataset/multi_view_dataset'
    cam_serial_list = ['22240376', '22206080', '22240368',
                       '22240389', '22222961', '22240382']
    # Get RT in world coord.
    rt_trans_world_dict = OrderedDict()
    for i in range(len(cam_serial_list)):
        left = i
        right = i+1
        if i == 5:
            right = 0
        r_vec, t_vec = get_RT_trans_from_serial(data_root, cam_serial_list[left], cam_serial_list[right])
        rt_trans_world_dict[str(left)+"_"+str(right)] = {"r_vec": r_vec, "t_vec": t_vec}

    is_old = False

    mtx_list, dist_list = load_intrinsics(data_root, cam_serial_list)

    # Get Chain RT in world coord.
    rt_world_cam = OrderedDict()
    R_world_cams, T_world_cams = list(), list()
    rt_trans_world_keys = list(rt_trans_world_dict.keys())
    R_world_cam_last = np.identity(3)
    T_world_cam_last = np.zeros([3, 1])
    R_world_cams.append(R_world_cam_last)
    T_world_cams.append(T_world_cam_last)
    for i in range(len(cam_serial_list)-1):
        T_world_cam_last = T_world_cam_last + np.matmul(R_world_cam_last, rt_trans_world_dict[rt_trans_world_keys[i]]['t_vec'])
        R_world_cam_last = np.matmul(R_world_cam_last, rt_trans_world_dict[rt_trans_world_keys[i]]['r_vec'])
        R_world_cams.append(R_world_cam_last)
        T_world_cams.append(T_world_cam_last)
    # Get Chain RT in camera coord.
    R_cams, T_cams = list(), list()
    for R_w_cam, T_w_cam in zip(R_world_cams, T_world_cams):
        R_cam, T_cam = from_world_to_cam_coord(R_w_cam, T_w_cam)
        R_cams.append(R_cam)
        T_cams.append(T_cam)

    # Load 3d structures ; 3d obj points, 2d image points (left, right side respectively)
    point3d_structure_list = list()
    for i in range(len(cam_serial_list)):
        left = i
        right = i+1
        if i == 5:
            right = 0
        point3d_structure_list.append(load_point3d_structure(data_root, cam_serial_list[left], cam_serial_list[right]))

    # flatten all elements of 3d structures
    _, _, obj_pts, img_pts_l, img_pts_r = flatten_3d_structure(point3d_structure_list)
    print()

    # Bundle adjustment
    bundle_root = os.path.join(data_root, 'extrinsic_images', 'bundle_result')
    bundle_result_save_path = os.path.join(bundle_root, 'bundle_result.npy')
    # Make optimizer input x_0
    x0, num_cam = make_x0(R_cams, T_cams, mtx_list, dist_list)
    # Make objective function for all cameras
    no_distortion = False
    f0 = fun_all_camera(x0, obj_pts, img_pts_l, img_pts_r, no_distortion)
    print('initial distance is : ', f0)
    t0 = time.time()
    res = least_squares(fun_all_camera, x0, loss='soft_l1', f_scale=0.1,
                        args=(obj_pts, img_pts_l, img_pts_r, no_distortion),
                        verbose=2, method='trf', xtol=1e-8, )
    t1 = time.time()

    print('Time: ', t1 - t0)
    print('before optimization: ', f0)
    print('after optimization: ', fun_all_camera(res.x, obj_pts,
                                                 img_pts_l, img_pts_r,
                                                 no_distortion))

    bundle_result_savename = os.path.join(bundle_root, 'bundle_result_further.npy')
    np.save(bundle_result_savename, res.x)

