import cv2 
from math import atan2, cos, sin, sqrt, pi
import numpy as np

def degreeToEuler(angle):

    euler_value = angle * (pi/180)

    return euler_value

def new_location(center, width, angle_z):

    z = 0 # fixed value
    location = [center[0] + width, center[1], z, 0, 0, angle_z] # location = [position_x, position_y, position_z, euler_x, euler_y, euler_z]

    return location

cap = cv2.VideoCapture(4) 
# Load the image
#img = cv2.imread("primer.jpg")

# PPI = 96
# width_min = 9.95 * (PPI / 25.4) # 9.95 mm
# width_max = 17 * (PPI / 25.4) # 17 mm
# height_min = 20 * (PPI / 25.4) # 20 mm
# height_max = 50 * (PPI / 25.4) # 50 mm
# added_value = 100 * (PPI / 25.4) # 50 mm
while cap.isOpened():

    # Read a frame from the webcam
    ret, frame = cap.read()
        
    # treshold the image
    #_, thresh_frame = cv2.threshold(frame, 240, 255, cv2.THRESH_BINARY)

    #frame = frame[0:480,130:480]

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert image to binary
    _, bw = cv2.threshold(gray_frame, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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
        print(location)
            
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

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()

 
