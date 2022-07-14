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
    # im_data = im_data.GetNDArray()
    # im_data = cv2.cvtColor(im_data, cv2.COLOR_BayerBG2BGR)
    fid=open(im_path, "bw")
    im_data.tofile(fid)
    fid.close()
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

    max_image_num = 8000
    start_time = time.time()
    display_time = 1.0
    counter = 0
    img_counter = 0

    detector_rescale = 0.5

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    cwd = os.getcwd()
    root_dir = os.path.join(cwd, 'acquired_image_png')
    print(root_dir)

    while True:
        
        for i, cam in enumerate(cam_list):
            # GET SERIAL NO.
            device_serial_number = ''
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(
                nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            image_result = cam.GetNextImage(1000)
            im_data = image_result.GetNDArray()
            im_data = cv2.cvtColor(im_data, cv2.COLOR_BayerBG2BGR)

            im_data = im_data[228:228+1080, 64:64+1920, :]

            # save images to serial number directory
            if not os.path.isdir(os.path.join(root_dir, device_serial_number)):
                os.makedirs(os.path.join(root_dir, device_serial_number))
            # image_filename = os.path.join(root_dir, device_serial_number, '%04d.raw'%img_counter)
            # mp_imwrite_raw(image_filename, im_data)
            cv2.imwrite(os.path.join(root_dir, device_serial_number, '%04d.png'%img_counter), im_data)

            image_result.Release()

        img_counter += 1
        counter += 1
        if time.time() - start_time > display_time:
            print("FPS over time {} sec is {}".format(
                display_time, counter/(time.time() - start_time)))
            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break