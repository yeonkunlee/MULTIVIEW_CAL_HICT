import os
import cv2
import numpy as np
import apriltag
import pickle
from glob import glob
from tqdm import tqdm
from typing import Union
from stereo_calibration_utils import load_intrinsic_results
from scipy.spatial.transform import Rotation as scipy_R


class StereoCalibration:
    def __init__(self,
                 cam_serials: list,
                 april_side_len: int,
                 april_tag_root: str) -> None:
        self.cam_serials = cam_serials
        self.left_top_corners = [[64, 228 * 2], [0, 228], [64, 0], [64, 100], [64, 228], [64, 288]]
        self.apriltag_one_side_length = april_side_len # 203
        self.april_tag_root = april_tag_root
        self.create_detector_base_result()

    def create_detector_base_result(self) -> None:
        detector = apriltag.Detector()
        img = cv2.imread(self.april_tag_root, cv2.IMREAD_GRAYSCALE)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.base_result = detector.detect(img)

    @staticmethod
    def init_stereo_calibration(res: tuple) -> Union[tuple, int, tuple]:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        res = res
        return criteria, stereocalibration_flags, res

    @staticmethod
    def create_detector() -> apriltag.Detector:
        return apriltag.Detector()

    def load_stereo_images(self,
                           left_serial: str,
                           right_serial: str,
                           prefix: str) -> Union[list, list, list, list]:
        stereo_root = os.path.join(data_root, 'extrinsic_images', prefix)
        left_image_dir = os.path.join(stereo_root, 'left_' + left_serial)
        left_image_list = glob(os.path.join(left_image_dir, '*.png'))
        left_image_list.sort()
        right_image_dir = os.path.join(stereo_root, 'right_' + right_serial)
        right_image_list = glob(os.path.join(right_image_dir, '*.png'))
        right_image_list.sort()

        left_basename_list = [os.path.basename(x) for x in left_image_list]
        right_basename_list = [os.path.basename(x) for x in right_image_list]
        return left_image_list, right_image_list, left_basename_list, right_basename_list

    def get_2d_3d_pts(self,
                      left_image_list: list,
                      right_image_list: list):
        left_detector = self.create_detector()
        right_detector = self.create_detector()

        april_objpoints = []
        left_april_imgpoints, right_april_imgpoints = [], []
        april_imagename = []
        point3d_structures = []
        processing_counter = 0
        num_image = len(left_image_list)
        if len(left_image_list) != len(right_image_list):
            raise "Total number of left and right image is not equal!"
        for i in tqdm(range(num_image)):
            left_image = cv2.imread(left_image_list[i])
            right_image = cv2.imread(right_image_list[i])

            # convert to gray scale
            left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            # find apriltag
            left_result = left_detector.detect(left_image_gray)
            right_result = right_detector.detect(right_image_gray)
            num_left_tag = len(left_result)
            num_right_tag = len(right_result)

            left_tag_id_list = []
            right_tag_id_list = []
            for i_tag in range(num_left_tag):
                left_tag_id_list.append(left_result[i_tag].tag_id)
            for i_tag in range(num_right_tag):
                right_tag_id_list.append(right_result[i_tag].tag_id)

            duplicated_tag_id = list(set(left_tag_id_list) & set(right_tag_id_list))

            if len(duplicated_tag_id) < 6:
                print('skip image:', left_image_list[i])
                continue
            else:
                processing_counter += 1

            # synthesis objpoints and imgpoints
            left_imgpoints = []
            right_imgpoints = []
            apriltag_objp = []

            for i_tag in duplicated_tag_id:
                for _left_i in range(num_left_tag):
                    if left_result[_left_i].tag_id == i_tag:
                        left_imgpoints.append(left_result[_left_i].corners)
                        break

                for _right_i in range(num_right_tag):
                    if right_result[_right_i].tag_id == i_tag:
                        right_imgpoints.append(right_result[_right_i].corners)
                        break

                for _base_i in range(12):
                    if self.base_result[_base_i].tag_id == i_tag:
                        apriltag_objp.append(self.base_result[_base_i].corners)
                        break

            left_imgpoints = np.array(left_imgpoints)
            left_imgpoints = left_imgpoints.reshape(-1, 1, 2)
            left_imgpoints = left_imgpoints.astype(np.float32)
            left_april_imgpoints.append(left_imgpoints)
            right_imgpoints = np.array(right_imgpoints)
            right_imgpoints = right_imgpoints.reshape(-1, 1, 2)
            right_imgpoints = right_imgpoints.astype(np.float32)
            right_april_imgpoints.append(right_imgpoints)

            apriltag_objp = np.array(apriltag_objp)
            apriltag_objp = apriltag_objp.reshape(-1, 2)
            _zeros = np.zeros([apriltag_objp.shape[0], 1])
            apriltag_objp = np.concatenate([apriltag_objp, _zeros], axis=1)
            apriltag_objp *= self.apriltag_one_side_length / (597.00000089 - 27.99999955)
            april_objpoints.append(apriltag_objp.astype(np.float32))
            april_imagename.append([left_image_list[i], right_image_list[i]])
            point3d_structures.append({"left_img_name": left_image_list[i], "right_img_name": right_image_list[i],
                                       "april_objpoints": apriltag_objp.astype(np.float32),
                                       "left_detected_points": left_imgpoints,
                                       "right_detected_points": right_imgpoints})
        return april_objpoints, left_april_imgpoints, right_april_imgpoints, april_imagename, point3d_structures


if __name__ == '__main__':
    data_root = '/home/juni/project/dataset/multi_view_dataset/'
    april_tag_root = './ros-apriltag-board.png'
    april_side_len = 203
    res = (2048, 1536) # W, H
    cam_serials = ['22240376', '22206080', '22240368',
                   '22240389', '22222961', '22240382']
    sc = StereoCalibration(cam_serials=cam_serials,
                           april_side_len=april_side_len,
                           april_tag_root=april_tag_root)
    for i in range(len(cam_serials)):
        left_serial = cam_serials[i]
        if i == len(cam_serials)-1:
            right_serial = cam_serials[0]
        else:
            right_serial = cam_serials[i+1]

        prefix = 'left_' + left_serial + '_right_' + right_serial

        print(f"processing camera between {left_serial} and {right_serial}...")
        # Get stereo images
        left_image_list, right_image_list, left_basename_list, right_basename_lsit = sc.load_stereo_images(left_serial,
                                                                                                           right_serial,
                                                                                                           prefix)
        # Get 2d and 3d pts from stereo images
        obj_pts, img_pts_l, img_pts_r, img_names, pts3d_structures = sc.get_2d_3d_pts(left_image_list, right_image_list)
        # Save results
        with open(os.path.join(data_root, 'extrinsic_images', prefix, 'point3d_structure.pkl'), 'wb') as f:
            pickle.dump(pts3d_structures, f)
        criteria, flags, res = sc.init_stereo_calibration(res=res)
        left_intrinsic, right_intrinsic = load_intrinsic_results(data_root, left_serial, right_serial)
        K1, D1, K2, D2 = left_intrinsic['mtx'], left_intrinsic['dist'], right_intrinsic['mtx'], right_intrinsic['dist']
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(obj_pts, img_pts_l, img_pts_r,
                                                              K1, D1, K2, D2, res, criteria=criteria, flags=flags)
        # get rotation and translation vector w.r.t. camera coord.
        rvec_cam = scipy_R.from_matrix(R).as_rotvec().reshape(3, 1)
        tvec_cam = T.reshape(3, 1)
        stereo_root = os.path.join(data_root, 'extrinsic_images', prefix)
        # save relative R, t
        np.save(os.path.join(data_root, 'extrinsic_images', 'left_' + left_serial + '_right_' + right_serial + '_rvec.npy'), rvec_cam)
        np.save(os.path.join(data_root, 'extrinsic_images', 'left_' + left_serial + '_right_' + right_serial + '_tvec.npy'), tvec_cam)
