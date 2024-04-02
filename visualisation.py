import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():

    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert image to binary
    _, bw = cv2.threshold(gray_frame, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
                
        label = "  Angle: " + str(angle) + " deg"
        textbox = cv2.rectangle(gray_frame, (center[0]-35, center[1]-25), 
            (center[0] + 295, center[1] + 10), (255,255,255), -1)
        cv2.putText(gray_frame, label, (center[0]-50, center[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.drawContours(gray_frame,[box],0,(0,0,255),2)
 
    cv2.imshow('Visualisation', gray_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()