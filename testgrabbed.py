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
# Read the first frame
ret, frame = cap.read()
cv2.imshow('frame', frame)
# Read the next frame
while True:
    ret, frame = cap.read()
    grabbed = ig.IsGrabbed(frame, 130) #frame, treshold
    if grabbed:
        print("send_OK_to_robot")
    #Display the image with mat plot lib
    #cv2.imshow('frame', frame)
#     #camera callibration output
#     mtx = np.array([[644.68813571,   0.0 ,        308.84858405],
#                             [  0.0  ,       645.41401477, 238.27188475],
#                             [  0.0 ,          0.0 ,          1.0      ]])
#     dist = np.array([-3.19275846e-03,  1.84520855e-01, -6.35263471e-04, -2.54347190e-03, -8.08299110e-01])


#         # Assuming you have the camera matrix (mtx) and distortion coefficients (dist)
#     img = frame
#     h, w = img.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#     # undistort
#     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

#     # crop the image
#     #x, y, w, h = roi
#     #dst = dst[y:y+h, x:x+w]
#     #cv2.imwrite('calibresult.png', dst)

#     frame = dst

#     #crop image 
#     print(frame.shape)
#     frame = frame[:,80:630]


#     # Convert the frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Convert image to binary
#     _, bw = cv2.threshold(gray_frame, 210, 255, cv2.THRESH_BINARY)


#     # Find all the contours in the thresholded image
#     contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# #*****************************************************************************************************************
#     objects = 0
#     small_location = []
#     middle_location = []
#     large_location = []

# #*****************************************************************************************************************
#     for i, contour in enumerate(contours):

#         # Calculate the area of each contour
#         area = cv2.contourArea(contour)
#         print(area)

#         # Ignore contours that are too small or too large
#         if area < 600 or area > 800:
#             continue
#         # cv2.minAreaRect returns:
#         # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
        
#         # Retrieve the key parameters of the rotated bounding box
#         center = (int(rect[0][0]),int(rect[0][1])) 
#         width = int(rect[1][0])
#         height = int(rect[1][1])
#         angle = int(rect[2])

#         #euler_value = degreeToEuler(angle)
            
#         if width < height:
#             angle = 90 - angle
#         else:
#             angle = -angle
            
#         label = "  Angle: " + str(angle) + " deg"
#         textbox = cv2.rectangle(gray_frame, (center[0]-35, center[1]-25), 
#             (center[0] + 295, center[1] + 10), (255,255,255), -1)
#         cv2.putText(gray_frame, label, (center[0]-50, center[1]), 
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        
#     #*****************************************************************************************************************
#         # if area > 600 and area < 800:
#         #     cv2.drawContours(frame,[box],0,(255,0,0),2)
#         #     objects += 1
#         #     small_location.append = new_location(center, width, euler_value)
#         # elif area < 1:
#         #     cv2.drawContours(frame,[box],0,(0,255,0),2)
#         #     objects += 1
#         #     middle_location.append = new_location(center, width, euler_value)
#         # elif area < 2:
#         #     cv2.drawContours(frame,[box],0,(0,0,255),2)
#         #     objects += 1
#         #     large_location.append = new_location(center, width, euler_value)
#     #*****************************************************************************************************************

#         # Display the image with mat plot lib
            
#         #location.append = new_location(center, width, euler_value)
#         # plt.figure()
#         # plt.title('final')
    #     # plt.imshow(gray_frame)
    #     # plt.show()
    # # Display the frame
    # cv2.imshow('frame', gray_frame)
    # # Break the loop  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object
cap.release()
# Destroy all windows
cv2.destroyAllWindows()





