import cv2
import numpy as np
#import TakePicture as 
import IsGrabbed as ig
import UR_TCP_IP
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from math import pi

import os
os.system('cls') # brisanje terminala (windows = cls, linux = clear)

#robo vid
from pygrabber.dshow_graph import FilterGraph

#**********************************Find the available cameras*************************************************

def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

cameras = get_available_cameras()
camera_port = -1

for i in range(len(cameras)):
    #if cameras[i] == 'C922 Pro Stream Webcam':
    if cameras[i] == 'HD Pro Webcam C920':      
        camera_port = i

#get camera name
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
print("capture done")

#************************************************************************************************************

#*****************************************Defining some variables********************************************
ukaz = ""

test_msg = 'ni ok'
zakljucen_gib = 'ni zakljucen'

home_pos = [0.39128, -0.23382, 0.32238, 1.611, 0.055, -0.019]
nad_paleto_varna = []
v_paleti = []

#************************************************************************************************************

print("Run the program on the teach pendant of the UR robot")
UR = UR_TCP_IP.TCP_IP()

#*****************************************Convert degrees to radians*****************************************

def degreeToEuler(angle):

    euler_value = angle * (pi/180)

    return euler_value

#************************************************************************************************************

#*****************************************Get new location of an detected object*****************************

def new_location(center, width, angle_z):

    #size of the image
    location = [0.57651, -0.38739, 0.027, 1.611, 0.055, -0.019] # location = [position_x, position_y, position_z, angle_x, angle_y, angle_z]
    print('center[0]: ', center[0])
    print('center[1]: ', center[1])

    z = 0.003 # fixed value
    x = center[0] + width
    y = center[1]
    
    print('x:', x)
    print('y:', y)

    #remap values from size of width to 0-28.4
    y = np.interp(y, [0, 580], [0, 0.34])
    x = np.interp(x, [0, 460], [0, 0.284])

    print('Interpolate x:', x)
    print('Interpolate y:', y)

    print('angle_z:', angle_z)

    location[0] += -x
    location[1] += y
    location[2] += 0
    #location = RotateTool(0, angle_z-1.611, 0, location)
    location[3] += 0
    location[4] += 0
    location[5] += angle_z

    return location

#************************************************************************************************************

def TakePicture(frame):
    
    # Display the image with mat plot lib
    # plt.figure()
    # plt.title('takepicture')
    # plt.imshow(frame)
    # plt.show()
    #camera callibration output
    mtx = np.array([[644.68813571,   0.0 ,        308.84858405],
                            [  0.0  ,       645.41401477, 238.27188475],
                            [  0.0 ,          0.0 ,          1.0      ]])
    dist = np.array([-3.19275846e-03,  1.84520855e-01, -6.35263471e-04, -2.54347190e-03, -8.08299110e-01])


        # Assuming you have the camera matrix (mtx) and distortion coefficients (dist)
    img = frame
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    #cv2.imwrite('calibresult.png', dst)

    frame = dst

    #crop image 
    print(frame.shape)
    frame = frame[:,20:600]

    # Display the image with mat plot lib
    plt.figure()
    plt.title('takepicture')
    plt.imshow(frame)
    plt.show()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert image to binary
    _, bw = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)


    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours):

        # Calculate the area of each contour
        area = cv2.contourArea(contour)

        # Ignore contours that are too small or too large
        if area < 600 or area > 800:
            continue

        # cv2.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        euler_value = degreeToEuler(angle)
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
                
        label = "  Angle: " + str(angle) + " deg"
        textbox = cv2.rectangle(frame, (center[0]-35, center[1]-25), 
            (center[0] + 295, center[1] + 10), (255,255,255), -1)
        cv2.putText(gray_frame, label, (center[0]-50, center[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
        
        # Display the image with mat plot lib
            
        location = new_location(center, width, euler_value)
        plt.figure()
        plt.title('final')
        plt.imshow(frame)
        plt.show()

        return location
    
#************************************************************************************************************
while True:
    print("We are in the WHILE loop")
    #Skos prejemaj ukaze in shranjuj v spremenljivke
    ret, frame = cap.read()

    ukaz = UR.GetRobotMessage()

    #check which key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('z'):
        ukaz = "ukaz_zajem_slike"
        print("zajem")
    elif key == ord('p'):
        ukaz = "preveri_prijem"
        print("prijem")
        

    if ukaz == "ukaz_zajem_slike":
        # Read a frame from the webcam
        print(ukaz)
        ret, frame = cap.read()
        #cv2.imshow('ukaz', frame)
        location = TakePicture(frame)

        print("location:", location)
        UR.SendRobotCoordinates(location)
        print("New location sent to the robot!")


    if ukaz == "preveri_prijem":
        # Read a frame from the webcam
        print(ukaz)
        ret, frame = cap.read()
        grabbed = ig.IsGrabbed(frame, 130) #frame, treshold
        if grabbed:
            print("send_OK_to_robot")
            
        else:
            #T1 = (0,0,0,1,0,0)
            #T2 = (0,0,0,1,0,0)
            #T3 = (0,0,0,1,0,0)
            print("send T1, T2, T3")
            home_pos[2] = home_pos[2]+0.01 # premik po z osi
            home_pos[5] = home_pos[5]+0.6# rot z
            UR.SendRobotCoordinates(home_pos)

    ukaz = ""


# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()