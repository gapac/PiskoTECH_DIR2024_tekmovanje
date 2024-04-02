import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np

#function that will get rotation of object based on the contour
def get_rotation(w, h, image):

    #Processing box:  1  Width:  23 Height:  66
    #Processing box:  2  Width:  67 Height:  22
    # Check if the contour is a square

    # remap values from 22-67 to 0-90
    angle = np.interp(w, [22, 76], [0, 90])

    return angle

def degreeToEuler(angle):

    euler_value = angle * (pi/180)

    return euler_value

def new_location(center, width, angle_z):
    print(width)

    #size of the image
    location = [0.57651, -0.38739, 0.027, 1.611, 0.027, -0.059] # location = [position_x, position_y, position_z, euler_x, euler_y, euler_z
    print('center[0]: ', center[0])
    print('center[1]: ', center[1])

    z = 0.003 # fixed value
    x = center[0] + width
    y = center[1]
    
    print('x:', x)
    print('y:', y)

    #remap values from size of width to 0-28.4
    y = np.interp(y, [0, 623], [0, 0.34])
    x = np.interp(x, [0, 462], [0, 0.284])

    print('Interpolate x:', x)
    print('Interpolate y:', y)

    location[0] += x
    location[1] += y
    location[2] += 0
    location[3] += 0
    location[4] += 0
    location[5] += 0 # location = [position_x, position_y, position_z, euler_x, euler_y, euler_z]

    return location


def TakePicture(frame):

    #show image
    cv2.imshow('takepicture', frame)
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
    #show image
    cv2.imshow('Undistorted frame', frame)

        # treshold the image
    #_, thresh_frame = cv2.threshold(frame, 240, 255, cv2.THRESH_BINARY)

    #frame = frame[0:480,130:480]

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert image to binary
    _, bw = cv2.threshold(gray_frame, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    print(bw.shape)
     #show image
    cv2.imshow('treshold', bw)

    #crop image
    bw = bw[0:480, 0:640]

    #show image
    cv2.imshow('croped treshold', bw)


    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours):

        # Calculate the area of each contour
        #area = cv2.contourArea(contour)
        #print(area)
        #area_min = (width_min + added_value) * (height_min + added_value)
        #area_max = (width_max + added_value) * (height_max + added_value)
        #area_min = width_min*height_min
        #area_max= width_max*height_max
        #print(area_min)
        #print(area_max)

        # Ignore contours that are too small or too large
        #if area < area_min or area > area_max:
            #continue
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
        location = new_location(center, width, euler_value)
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
                
        label = "  Angle: " + str(euler_value) + " rad"
        textbox = cv2.rectangle(gray_frame, (center[0]-35, center[1]-25), 
            (center[0] + 295, center[1] + 10), (255,255,255), -1)
        cv2.putText(gray_frame, label, (center[0]-50, center[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.drawContours(gray_frame,[box],0,(0,0,255),2)

    cv2.imshow('Output Image', gray_frame)

    return location


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''#treshold the image
    _, frame = cv2.threshold(dst, 200, 255, cv2.THRESH_BINARY)

    #print(frame.shape)
    #crop picture
    #frame = frame[0:480, 0:640]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours on every 10th frame

    for i, contour in enumerate(contours):
        # Get the rectangle that encloses the contour
        x, y, w, h = cv2.boundingRect(contour)

        angle = get_rotation(w, h, frame)
        #print on the last itteration

        
        print("Processing box: ", i , "Angle: " , angle , "Width: " , w , "Height: " , h)

        #if i= 0 
        #    sent to robot(x,y,angle)

        # Draw the rectangle on the original image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #wait for 2 seconds
        #cv2.waitKey(500)

        if h+w > 50:
            #show image on new window
            cv2.imshow('frame2', frame)
            return x, y, angle
        else:
            return False


    # Release the video capture object
    #frame.release()

    # Close all OpenCV windows
    #cv2.destroyAllWindows()'''
        



