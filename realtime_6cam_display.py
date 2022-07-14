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

# def mp_imwrite_raw(im_path, im_data):
#     # im_data = im_data.GetNDArray()
#     # im_data = cv2.cvtColor(im_data, cv2.COLOR_BayerBG2BGR)
#     fid=open(im_path, "bw")
#     im_data.tofile(fid)
#     fid.close()
#     return im_data

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

    vis_scale = 0.2

    cwd = os.getcwd()
    root_dir = os.path.join(cwd, 'acquired_image')
    print(root_dir)

    while True:

        image_list = []
        
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

            im_plot = cv2.resize(im_data, (0, 0), fx=vis_scale, fy=vis_scale)
            image_list.append(im_plot)


            image_result.Release()
        
        # display image
        im_0_1_2 = cv2.hconcat(image_list[:3])
        im_3_4_5 = cv2.hconcat(image_list[3:])
        im_all = cv2.vconcat([im_0_1_2, im_3_4_5])

        cv2.imshow('frame', im_all)

        img_counter += 1
        counter += 1
        if time.time() - start_time > display_time:
            print("FPS over time {} sec is {}".format(
                display_time, counter/(time.time() - start_time)))
            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break