import os
import numpy as np
from typing import Union


def load_intrinsic_results(data_root: str,
                           left_serial: str,
                           right_serial: str) -> Union[np.ndarray, np.ndarray]:
    # data_root = '/home/juni/project/dataset/multi_view_dataset/'
    # prefix = 'left_' + left_serial + '_right_' + right_serial
    left_npz_file = np.load(os.path.join(data_root, 'intrinsic_images', left_serial + '_intrinsic_result.npz'))
    right_npz_file = np.load(os.path.join(data_root, 'intrinsic_images', right_serial + '_intrinsic_result.npz'))
    return left_npz_file, right_npz_file