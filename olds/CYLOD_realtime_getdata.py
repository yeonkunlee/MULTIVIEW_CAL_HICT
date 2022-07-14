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


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial_number', type=str,
                        help='camera serial number')
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

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    max_image_num = 2000
    start_time = time.time()
    display_time = 1.0
    counter = 0
    img_counter = 0
    detector_rescale_vis = 0.5

    # image writing speed is too fast. disable imwrite in some time
    imwrite_enable = False
    imwrite_time = time.time()
    imwrite_interval = 0.5

    number_of_images_confirmed = 0

    for i, cam in enumerate(cam_list):
        cam.BeginAcquisition()

    while True:
        for i, cam in enumerate(cam_list):

            # GET SERIAL NO.
            device_serial_number = ''
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(
                nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            image_dir = os.path.join('/home/hict/yeonkunlee/MULTIVIEW_CAL_HICT/cal_image/intrinsic_images',
                                    device_serial_number)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            filename = os.path.join(image_dir, '%04d.png' % (img_counter % max_image_num))

            image_result = cam.GetNextImage(1000)
            if device_serial_number == args.serial_number:
                im_data = image_result.GetNDArray()
                im_data = cv2.cvtColor(im_data, cv2.COLOR_BayerBG2BGR)

                im_data = im_data[228:228+1080, 64:64+1920, :]

                # cv2.imwrite(filename, im_data)
                im_data_show = copy.deepcopy(im_data)
                im_data_show = cv2.resize(
                    im_data_show, (0, 0), fx=detector_rescale_vis, fy=detector_rescale_vis)

                # apriltag detector appears to show error. check it
                try:
                    detector_result = detector.detect(
                        cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY))
                except:
                    print('Apriltag detector error. This appearence usually means that dark view image aquired.')
                    continue

                # detector_result = detector.detect(
                #     cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY))

                for tag in detector_result:
                    corners = tag.corners * detector_rescale_vis # shape is (4, 2)
                    centers = tag.center * detector_rescale_vis # shape is (2,)
                    # draw the four lines with corners. corners are in (x, y). float type
                    cv2.line(im_data_show, (int(corners[0, 0]), int(corners[0, 1])),
                             (int(corners[1, 0]), int(corners[1, 1])), (0, 255, 0), 2)
                    cv2.line(im_data_show, (int(corners[1, 0]), int(corners[1, 1])),
                             (int(corners[2, 0]), int(corners[2, 1])), (0, 255, 0), 2)
                    cv2.line(im_data_show, (int(corners[2, 0]), int(corners[2, 1])),
                             (int(corners[3, 0]), int(corners[3, 1])), (0, 255, 0), 2)
                    cv2.line(im_data_show, (int(corners[3, 0]), int(corners[3, 1])),
                             (int(corners[0, 0]), int(corners[0, 1])), (0, 255, 0), 2)

                    # cv2.rectangle(im_data_show, (int(corners[0, 0]), int(corners[0, 1])),
                    #               (int(corners[2, 0]), int(corners[2, 1])), (0, 255, 0), 2)
                    # draw the tag center point. float type
                    cv2.circle(im_data_show, (int(centers[0]), int(centers[1])),
                               2, (0, 255, 0), 2)
                    # draw the tag id. int type
                    cv2.putText(im_data_show, str(tag.tag_id), (int(centers[0]), int(centers[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # imsave if detector result length == 12
                
                if len(detector_result) == 12 and imwrite_enable:
                    cv2.imwrite(filename, im_data)
                    number_of_images_confirmed += 1
                    imwrite_enable = False

                # mirror image after all processing. for convenience
                # im_data_show = cv2.flip(im_data_show, 1)
                #plot number_of_images_confirmed to image corner
                # cv2.putText arguments are : (image, text, (x, y), font, fontScale, (B, G, R), thickness, lineType)
                cv2.putText(im_data_show, str(number_of_images_confirmed), (60, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)
                cv2.imshow('frame', im_data_show)
                
            image_result.Release()

        # imwrite_enable = True
        if time.time() - imwrite_time > imwrite_interval:
            imwrite_enable = True
            imwrite_time = time.time()

        img_counter += 1
        counter += 1
        if time.time() - start_time > display_time:
            print("FPS over time {} sec is {}".format(
                display_time, counter/(time.time() - start_time)))
            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # for i, cam in enumerate(cam_list):
    #     cam.ClearQueue()
    #     cam.EndAcquisition()
    #     cam.DeInit()

    # what():  Spinnaker: Can't clear a camera because something still holds a reference to the camera [-1004]

    cv2.destroyAllWindows()


# cam.EndAcquisition()
