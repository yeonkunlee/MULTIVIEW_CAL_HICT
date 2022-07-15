import pickle
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R
from typing import Union


def get_RT_world_from_rtvec(_rvec=[.0,.0,.0],
                            _tvec=[.0,.0,.0]) -> Union[np.ndarray, np.ndarray]:
    _R_c, _ = cv2.Rodrigues(_rvec)
    _R_w = _R_c.transpose()
    _T_c = _tvec
    _T_w = -np.matmul(_R_w, _T_c)
    return _R_w, _T_w


def get_RT_trans_from_serial(_data_root: str,
                             _left_serial: str,
                             _right_serial: str) -> Union[np.ndarray, np.ndarray]:
    _prefix = 'left_' + _left_serial + '_right_' + _right_serial
    _stereo_root = os.path.join(_data_root, 'extrinsic_images', _prefix)
    _rvec_cam = np.load(os.path.join(_stereo_root, 'left_' + _left_serial + '_right_' + _right_serial+'_rvec.npy'))
    _tvec_cam = np.load(os.path.join(_stereo_root, 'left_' + _left_serial + '_right_' + _right_serial+'_tvec.npy'))

    R_trans_world_left_to_right, T_trans_world_left_to_right = get_RT_world_from_rtvec(_rvec_cam, _tvec_cam)

    return R_trans_world_left_to_right, T_trans_world_left_to_right


def load_intrinsics(data_root: str,
                    cam_serial_list: list) -> Union[list, list]:
    mtx_list = []
    dist_list = []
    for cam_serial in cam_serial_list:
        _npz_file = np.load(os.path.join(data_root, "intrinsic_images", cam_serial + '_intrinsic_result.npz'))
        _mtx = _npz_file['mtx']
        _dist = _npz_file['dist']
        mtx_list.append(_mtx)
        dist_list.append(_dist)

    return mtx_list, dist_list


def from_world_to_cam_coord(_R_w: np.ndarray,
                            _T_w: np.ndarray) -> Union[np.ndarray, np.ndarray]:
    _R_cam = _R_w.transpose()
    _T_cam = -np.matmul(_R_cam, _T_w)
    _r_scipy = scipy_R.from_matrix(_R_cam)
    _rvec = _r_scipy.as_rotvec().reshape(3,1)
    _tvec = _T_cam.reshape(3,1)
    return _rvec, _tvec


def calculate_Tr(rvec_np: np.ndarray,
                 tvec_np: np.ndarray) -> np.ndarray:
    _R_cv2bcam = np.array([[1,0,0],
                       [0,-1,0],
                       [0,0,-1]])
    R, _ = cv2.Rodrigues(rvec_np)
    t = tvec_np/100. #(mm to M)? scale problem exist.
    R_bcam= np.matmul(_R_cv2bcam, R)
    T_bcam = np.matmul(_R_cv2bcam, t)
    R_world = R_bcam.transpose()
    T_world = -np.matmul(R_world, T_bcam)

    R_world = R_world.reshape(3,3)
    T_world = T_world.reshape(3,1)

    Tr = np.identity(4)
    Tr[:3,:3] = R_world
    Tr[:3,3:] = T_world

    return Tr


def get_K(_params):
    K = np.zeros((3,3))
    K[0,0] = _params[0]
    K[1,1] = _params[1]
    K[0,2] = _params[2]
    K[1,2] = _params[3]
    K[2,2] = 1.
    return K


def distance_calculator(_rvec_cam_left: np.ndarray,
                        _tvec_cam_left: np.ndarray,
                        _rvec_cam_right: np.ndarray,
                        _tvec_cam_right: np.ndarray,
                        _april_objpoints: list,
                        _left_april_imgpoints: list,
                        _right_april_imgpoints: list,
                        _left_mtx: np.ndarray,
                        _left_dist: np.ndarray,
                        _right_mtx: np.ndarray,
                        _right_dist: np.ndarray) -> float:
    distance_holder = []
    for i in range(len(_april_objpoints)):
        left_success, left_rvec, left_tvec = cv2.solvePnP(_april_objpoints[i], _left_april_imgpoints[i], _left_mtx, _left_dist, 0)
        left_R = cv2.Rodrigues(left_rvec)[0]
        right_success, right_rvec, right_tvec = cv2.solvePnP(_april_objpoints[i], _right_april_imgpoints[i], _right_mtx, _right_dist, 0)
        right_R = cv2.Rodrigues(right_rvec)[0]

        # left_detected_imgpoints = _left_april_imgpoints[i].reshape(-1, 2)
        # right_detected_imgpoints = _right_april_imgpoints[i].reshape(-1, 2)

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


def fun(parameters,
        _april_objpoints,
        _left_april_imgpoints,
        _right_april_imgpoints):
    rvec_cam_left = parameters[:3].reshape(3, 1)
    tvec_cam_left = parameters[3:6].reshape(3, 1)
    rvec_cam_right = parameters[6:9].reshape(3, 1)
    tvec_cam_right = parameters[9:12].reshape(3, 1)
    left_mtx = get_K(parameters[12:16])
    left_dist = parameters[16:21].reshape(1, 5)
    right_mtx = get_K(parameters[21:25])
    right_dist = parameters[25:30].reshape(1, 5)

    return distance_calculator(rvec_cam_left, tvec_cam_left, rvec_cam_right, tvec_cam_right, _april_objpoints,
                               _left_april_imgpoints, _right_april_imgpoints, left_mtx, left_dist, right_mtx,
                               right_dist)


def load_point3d_structure(data_root: str,
                           _left_serial: str,
                           _right_serial: str) -> list:
    _prefix = 'left_' + _left_serial + '_right_' + _right_serial
    _stereo_root = os.path.join(data_root, 'extrinsic_images', _prefix)
    point3d_structure = pickle.load(open(os.path.join(_stereo_root, 'point3d_structure.pkl'), 'rb'))
    return point3d_structure


def make_x0(_rvec_cam_list: list,
            _tvec_cam_list: list,
            _mtx_list: list,
            _dist_list: list) -> Union[np.ndarray, int]:
    assert (len(_rvec_cam_list) == len(_tvec_cam_list) == len(_mtx_list) == len(
        _dist_list)), 'length of rvec_cam_list, tvec_cam_list, mtx_list, dist_list should be equal'
    num_cams = len(_rvec_cam_list)
    x0 = np.concatenate([_rvec_cam_list, _tvec_cam_list], axis=1)
    x0 = x0.reshape(-1)

    for i in range(num_cams):
        # x0 = np.concatenate([x0, _mtx_list[i].flatten(), _dist_list[i].flatten()])
        x0 = np.concatenate(
            [x0, np.array([_mtx_list[i][0, 0], _mtx_list[i][1, 1], _mtx_list[i][0, 2], _mtx_list[i][1, 2]]).flatten()])
        x0 = np.concatenate([x0, _dist_list[i].flatten()])

    return x0, num_cams


def fun_all_camera(parameters: np.ndarray,
                   _april_objpoints_all: list,
                   _left_april_imgpoints_all: list,
                   _right_april_imgpoints_all: list,
                   no_distortion: bool = False) -> float:
    # no_distortion = True

    rvec_cam_0 = parameters[:3].reshape(3, 1)
    tvec_cam_0 = parameters[3:6].reshape(3, 1)

    rvec_cam_0 = np.zeros([3, 1])
    tvec_cam_0 = np.zeros([3, 1])  # FIX FIRST CAMERA IDENTITY

    rvec_cam_1 = parameters[6:9].reshape(3, 1)
    tvec_cam_1 = parameters[9:12].reshape(3, 1)
    rvec_cam_2 = parameters[12:15].reshape(3, 1)
    tvec_cam_2 = parameters[15:18].reshape(3, 1)
    rvec_cam_3 = parameters[18:21].reshape(3, 1)
    tvec_cam_3 = parameters[21:24].reshape(3, 1)
    rvec_cam_4 = parameters[24:27].reshape(3, 1)
    tvec_cam_4 = parameters[27:30].reshape(3, 1)
    rvec_cam_5 = parameters[30:33].reshape(3, 1)
    tvec_cam_5 = parameters[33:36].reshape(3, 1)
    mtx_0 = get_K(parameters[36:40])
    dist_0 = parameters[40:45].reshape(1, 5)
    mtx_1 = get_K(parameters[45:49])
    dist_1 = parameters[49:54].reshape(1, 5)
    mtx_2 = get_K(parameters[54:58])
    dist_2 = parameters[58:63].reshape(1, 5)
    mtx_3 = get_K(parameters[63:67])
    dist_3 = parameters[67:72].reshape(1, 5)
    dist_3 = np.zeros([1, 5])
    mtx_4 = get_K(parameters[72:76])
    dist_4 = parameters[76:81].reshape(1, 5)
    dist_4 = np.zeros([1, 5])
    mtx_5 = get_K(parameters[81:85])
    dist_5 = parameters[85:90].reshape(1, 5)
    dist_5 = np.zeros([1, 5])

    if no_distortion:
        dist_0 = np.zeros([1, 5])
        dist_1 = np.zeros([1, 5])
        dist_2 = np.zeros([1, 5])
        dist_3 = np.zeros([1, 5])
        dist_4 = np.zeros([1, 5])
        dist_5 = np.zeros([1, 5])

    distance_0_1 = distance_calculator(rvec_cam_0, tvec_cam_0, rvec_cam_1, tvec_cam_1, _april_objpoints_all[0],
                                       _left_april_imgpoints_all[0], _right_april_imgpoints_all[0], mtx_0, dist_0,
                                       mtx_1, dist_1)
    distance_1_2 = distance_calculator(rvec_cam_1, tvec_cam_1, rvec_cam_2, tvec_cam_2, _april_objpoints_all[1],
                                       _left_april_imgpoints_all[1], _right_april_imgpoints_all[1], mtx_1, dist_1,
                                       mtx_2, dist_2)
    distance_2_3 = distance_calculator(rvec_cam_2, tvec_cam_2, rvec_cam_3, tvec_cam_3, _april_objpoints_all[2],
                                       _left_april_imgpoints_all[2], _right_april_imgpoints_all[2], mtx_2, dist_2,
                                       mtx_3, dist_3)
    distance_3_4 = distance_calculator(rvec_cam_3, tvec_cam_3, rvec_cam_4, tvec_cam_4, _april_objpoints_all[3],
                                       _left_april_imgpoints_all[3], _right_april_imgpoints_all[3], mtx_3, dist_3,
                                       mtx_4, dist_4)
    distance_4_5 = distance_calculator(rvec_cam_4, tvec_cam_4, rvec_cam_5, tvec_cam_5, _april_objpoints_all[4],
                                       _left_april_imgpoints_all[4], _right_april_imgpoints_all[4], mtx_4, dist_4,
                                       mtx_5, dist_5)
    distance_5_0 = distance_calculator(rvec_cam_5, tvec_cam_5, rvec_cam_0, tvec_cam_0, _april_objpoints_all[5],
                                       _left_april_imgpoints_all[5], _right_april_imgpoints_all[5], mtx_5, dist_5,
                                       mtx_0, dist_0)

    distance = distance_0_1 + distance_1_2 + distance_2_3 + distance_3_4 + distance_4_5 + distance_5_0
    avg_distance = distance / 6.

    return avg_distance


def flatten_3d_structure(point_3d_structure: list) -> Union[list, list, list, list, list]:
    obj_pts = []
    left_img_list, right_img_list = [], []
    left_img_pts, right_img_pts = [], []
    for i in range(len(point_3d_structure)):
        left_img_name_list = []
        right_img_name_list = []
        april_objpoints_list = []
        left_april_imgpoints_list = []
        right_april_imgpoints_list = []
        for smaple_i in range(len(point_3d_structure[i])):
            left_img_name = point_3d_structure[i][smaple_i]['left_img_name']
            right_img_name = point_3d_structure[i][smaple_i]['right_img_name']
            april_objpoints = point_3d_structure[i][smaple_i]['april_objpoints']
            left_detected_imgpoints = point_3d_structure[i][smaple_i]['left_detected_points']
            right_detected_imgpoints = point_3d_structure[i][smaple_i]['right_detected_points']
            left_img_name_list.append(left_img_name)
            right_img_name_list.append(right_img_name)
            april_objpoints_list.append(april_objpoints)
            left_april_imgpoints_list.append(left_detected_imgpoints)
            right_april_imgpoints_list.append(right_detected_imgpoints)
        left_img_list.append(left_img_name_list)
        right_img_list.append(right_img_name_list)
        obj_pts.append(april_objpoints_list)
        left_img_pts.append(left_april_imgpoints_list)
        right_img_pts.append(right_april_imgpoints_list)

    return left_img_list, right_img_list, obj_pts, left_img_pts, right_img_pts

