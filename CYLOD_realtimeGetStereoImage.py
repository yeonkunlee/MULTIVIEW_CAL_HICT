import sys
import os
import PySpin
import matplotlib.pyplot as plt
import sys
import time
import cv2
import numpy as np
import copy
import argparse
import apriltag


def mp_imwrite_raw(im_path, im_data):
    im_data = im_data.GetNDArray()
    # im_data = cv2.cvtColor(im_data, cv2.COLOR_BayerBG2BGR)
#     fid=open(im_path, "bw")
#     im_data.tofile(fid)
#     fid.close()
    return im_data


def camera_initialization(_cam_list):
    for _i, cam in enumerate(_cam_list):
        device_serial_number = ''
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        node_device_serial_number = PySpin.CStringPtr(
            nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' %
                  device_serial_number)
        # Initialize camera
        cam.Init()
    #     cam.PixelFormat.SetIntValue(4)
        print(cam.PixelFormat.GetValue())


def draw_corners(_img, _tag, _rescale_factor):
    # draw the four lines with corners. corners are in (x, y). float type
    corners = _tag.corners
    corners = corners * _rescale_factor
    centers = _tag.center
    centers = centers * _rescale_factor
    cv2.line(_img, (int(corners[0, 0]), int(corners[0, 1])),
             (int(corners[1, 0]), int(corners[1, 1])), (0, 255, 0), 2)
    cv2.line(_img, (int(corners[1, 0]), int(corners[1, 1])),
             (int(corners[2, 0]), int(corners[2, 1])), (0, 255, 0), 2)
    cv2.line(_img, (int(corners[2, 0]), int(corners[2, 1])),
             (int(corners[3, 0]), int(corners[3, 1])), (0, 255, 0), 2)
    cv2.line(_img, (int(corners[3, 0]), int(corners[3, 1])),
             (int(corners[0, 0]), int(corners[0, 1])), (0, 255, 0), 2)

    cv2.circle(_img, (int(centers[0]), int(centers[1])),
               2, (0, 255, 0), 2)
    # draw the tag id. int type
    cv2.putText(_img, str(_tag.tag_id), (int(centers[0]), int(centers[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return _img


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_serial', type=str,
                        help='left camera serial number')
    parser.add_argument('--right_serial', type=str,
                        help='right camera serial number')
    args = parser.parse_args()
    # display camera image. (GUI)
    result = True
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' %
          (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    camera_initialization(cam_list)

    for i, cam in enumerate(cam_list):
        cam.BeginAcquisition()

    max_image_num = 4000
    start_time = time.time()
    display_time = 1.0
    counter = 0
    img_counter = 0

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    detector_rescale = 0.5

    # image writing speed is too fast. disable imwrite in some time
    imwrite_enable = False
    imwrite_time = time.time()
    imwrite_interval = 0.5

    number_of_images_confirmed = 0

    while True:

        # empty image buffer
        im_data_left_show = np.zeros((768, 1024, 3), dtype=np.uint8)
        im_data_right_show = np.zeros((768, 1024, 3), dtype=np.uint8)

        prefix = 'left_' + args.left_serial + '_right_' + args.right_serial
        image_dir = os.path.join('/home/hict/yeonkunlee/MULTIVIEW_CAL_HICT/cal_image/extrinsic_images',
                                 prefix)

        debug_dir = debug_filename = os.path.join(
                image_dir, 'debug')
        if not os.path.isdir(debug_dir):
            os.makedirs(debug_dir)
        
        right_image_dir = os.path.join(
                image_dir, 'right_'+args.right_serial)
        if not os.path.isdir(right_image_dir):
            os.makedirs(right_image_dir)
        left_image_dir = os.path.join(
                image_dir, 'left_'+args.left_serial)
        if not os.path.isdir(left_image_dir):
            os.makedirs(left_image_dir)

        left_tagname_list = []
        right_tagname_list = []
        for i, cam in enumerate(cam_list):
            # GET SERIAL NO.
            device_serial_number = ''
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(
                nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            if device_serial_number != args.left_serial and device_serial_number != args.right_serial:
                continue

            image_result = cam.GetNextImage(1000)

            

            if device_serial_number == args.left_serial:
                im_data_left = image_result.GetNDArray()
                im_data_left = cv2.cvtColor(
                    im_data_left, cv2.COLOR_BayerBG2BGR)

                im_data_show = copy.deepcopy(im_data_left)
                im_data_show = cv2.resize(
                    im_data_show, (0, 0), fx=detector_rescale, fy=detector_rescale)
                left_detector_result = detector.detect(
                    cv2.cvtColor(im_data_left, cv2.COLOR_BGR2GRAY))
                for tag in left_detector_result:
                    im_data_show = draw_corners(im_data_show, tag, detector_rescale)
                    left_tagname = tag.tag_id
                    left_tagname_list.append(left_tagname)
                im_data_left_show = im_data_show

                # number of detected tag
                num_detected_tag_left = len(left_detector_result)


            if device_serial_number == args.right_serial:
                im_data_right = image_result.GetNDArray()
                im_data_right = cv2.cvtColor(
                    im_data_right, cv2.COLOR_BayerBG2BGR)

                im_data_show = copy.deepcopy(im_data_right)
                im_data_show = cv2.resize(
                    im_data_show, (0, 0), fx=detector_rescale, fy=detector_rescale)
                right_detector_result = detector.detect(
                    cv2.cvtColor(im_data_right, cv2.COLOR_BGR2GRAY))
                for tag in right_detector_result:
                    im_data_show = draw_corners(im_data_show, tag, detector_rescale)
                    right_tagname = tag.tag_id
                    right_tagname_list.append(right_tagname)
                im_data_right_show = im_data_show

                # number of detected tag
                num_detected_tag_right = len(right_detector_result)
            
            image_result.Release()

        # concate left and right image
        im_data_show = np.concatenate(
            [im_data_left_show, im_data_right_show], axis=1)
        cv2.imshow('image', im_data_show)

        # duplicated tag name
        duplicated_tag_name_list = list(set(left_tagname_list) & set(right_tagname_list))
        # print(duplicated_tag_name_list)

        # if num_detected_tag_left == 12 and num_detected_tag_right == 12:
        if len(duplicated_tag_name_list) >= 6:
            print('both camera detected full tag. save image')
            debug_filename = os.path.join(
                debug_dir, '%04d.png' % (img_counter % max_image_num))
            if not os.path.isdir(image_dir):
                os.makedirs(os.path.dirname(image_dir)) # not working because of the permission issue
            right_filename = os.path.join(
                right_image_dir, '%04d.png' % (img_counter % max_image_num))
            left_filename = os.path.join(
                left_image_dir, '%04d.png' % (img_counter % max_image_num))
            if imwrite_enable:
                # puttext debug_filename number_of_images_confirmed
                cv2.putText(im_data_show, str(number_of_images_confirmed), (60, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)
                cv2.imwrite(debug_filename, im_data_show)
                cv2.imwrite(right_filename, im_data_right)
                cv2.imwrite(left_filename, im_data_left)
                imwrite_enable = False
                number_of_images_confirmed += 1

        img_counter += 1
        counter += 1
        if time.time() - start_time > display_time:
            print("FPS over time {} sec is {}".format(
                display_time, counter/(time.time() - start_time)))
            counter = 0
            start_time = time.time()

        # imwrite_enable
        if time.time() - imwrite_time > imwrite_interval:
            imwrite_enable = True
            imwrite_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # for i, cam in enumerate(cam_list):
    #     cam.ClearQueue()
    #     cam.EndAcquisition()
    #     cam.DeInit()

    # what():  Spinnaker: Can't clear a camera because something still holds a reference to the camera [-1004]

    cv2.destroyAllWindows()


# cam.EndAcquisition()
