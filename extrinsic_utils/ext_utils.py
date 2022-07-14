import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R


def calculate_Tr(K_np, rvec_np, tvec_np):
    _R_cv2bcam = np.array([[1,0,0],
                       [0,-1,0],
                       [0,0,-1]])
    R, _ = cv2.Rodrigues(rvec_np)
    # t = tvec_np[_i]/100. #(mm to M)? scale problem exist.
    t = tvec_np/100. #(mm to M)? scale problem exist.
    # k = first_K_np
    R_bcam= np.matmul(_R_cv2bcam, R)
    T_bcam = np.matmul(_R_cv2bcam, t)
    R_world = R_bcam.transpose()
    T_world = -np.matmul(R_world, T_bcam)

    R_world = R_world.reshape(3,3)
    T_world = T_world.reshape(3,1)

    Tr = np.identity(4)
    Tr[:3,:3] = R_world
    Tr[:3,3:] = T_world

    return Tr# debugging image is not nessesary, apriltag detection already confirmed.


def distance_calculator(_rvec_cam_left, _tvec_cam_left,
                        _rvec_cam_right, _tvec_cam_right,
                        _april_objpoints, _left_april_imgpoints,
                        _right_april_imgpoints, _left_mtx, _left_dist,
                        _right_mtx, _right_dist):
    distance_holder = []
    for i in range(len(_april_objpoints)):
        left_success, left_rvec, left_tvec = cv2.solvePnP(_april_objpoints[i], _left_april_imgpoints[i], _left_mtx, _left_dist, 0)
        left_R = cv2.Rodrigues(left_rvec)[0]
        right_success, right_rvec, right_tvec = cv2.solvePnP(_april_objpoints[i], _right_april_imgpoints[i], _right_mtx, _right_dist, 0)
        right_R = cv2.Rodrigues(right_rvec)[0]

        left_object_points = (left_R.dot(_april_objpoints[i].T) + left_tvec).T # N, 3
        right_object_points = (right_R.dot(_april_objpoints[i].T) + right_tvec).T # N, 3

        R_cam_left = cv2.Rodrigues(_rvec_cam_left)[0]
        left_object_points = R_cam_left.T.dot(left_object_points.T - _tvec_cam_left).T

        R_cam_right = cv2.Rodrigues(_rvec_cam_right)[0]
        right_object_points = R_cam_right.T.dot(right_object_points.T - _tvec_cam_right).T

        assert left_object_points.shape == right_object_points.shape

        distance = np.mean(np.linalg.norm(left_object_points - right_object_points, axis=1))

        distance_holder.append(distance)
    return np.mean(distance_holder)


def get_RT_world_from_rtvec(_rvec=[.0,.0,.0], _tvec=[.0,.0,.0]):
    _R_c, _ = cv2.Rodrigues(_rvec)
    _R_w = _R_c.transpose()
    _T_c = _tvec
    _T_w = -np.matmul(_R_w, _T_c)
    return _R_w, _T_w


def get_RT_trans_from_serial(_data_root, _left_serial, _right_serial):
    _prefix = 'left_' + _left_serial + '_right_' + _right_serial
    _stereo_root = os.path.join(_data_root, 'extrinsic_images', _prefix)
    _rvec_cam = np.load(os.path.join(_stereo_root, 'left_' + _left_serial + '_right_' + _right_serial+'_rvec.npy'))
    _tvec_cam = np.load(os.path.join(_stereo_root, 'left_' + _left_serial + '_right_' + _right_serial+'_tvec.npy'))

    R_trans_world_left_to_right, T_trans_world_left_to_right = get_RT_world_from_rtvec(_rvec_cam, _tvec_cam)

    return R_trans_world_left_to_right, T_trans_world_left_to_right


def from_world_to_cam_coord(_R_w, _T_w):
    _R_cam = _R_w.transpose()
    _T_cam = -np.matmul(_R_cam, _T_w)
    _r_scipy = scipy_R.from_matrix(_R_cam)
    _rvec = _r_scipy.as_rotvec().reshape(3,1)
    _tvec = _T_cam.reshape(3,1)
    return _rvec, _tvec