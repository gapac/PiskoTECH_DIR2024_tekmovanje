import cv2
import numpy as np
import matplotlib.pyplot as plt
# Open the saved video
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error opening video file")
# Open the webcam
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#function that will get rotation of object based on the contour
def get_rotation(w, h, image):
    #Processing box:  1  Width:  23 Height:  66
    #Processing box:  2  Width:  67 Height:  22
    # Check if the contour is a square

    # remap values from 22-67 to 0-90
    angle = np.interp(w, [22, 76], [0, 90])

    return angle

#function that will test a few pixels and return the color
def checkPixels(image, y, x, width, height):
    # Define the region of interest
    roi = image[y:y+height, x:x+width]

    # Get the intensity values in the ROI
    intensities = roi.reshape(-1)

    #average intensity
    avg = np.mean(intensities)


    return avg

def IsGrabbed(frame, treshold):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #treshold the image
    _, tresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    #print(frame.shape)
    #crop picture
    #frame = frame[0:480, 0:640]
    frame = frame[:,80:]

    targetx = 450
    targety = 0
    #red dot on the target
    cv2.circle(tresh, (int(targetx), int(targety)), 7, (0, 0, 255), -1)

    grabbed = checkPixels(tresh, targety, targetx, 10, 10)
    
    print("Grabbed ", grabbed)

    # Display the image
    cv2.imshow('frame', tresh)

    #display with matplot lib
    #plt.figure()
    #plt.imshow(tresh)
    #plt.show()
    
    if grabbed > treshold:
        print("Object is grabbed")
        return True
    else:
        print("Object is not grabbed")
        return False

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

































































'''
import cv2
import numpy as np
# Open the saved video
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error opening video file")
# Open the webcam
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#function that will get rotation of object based on the contour
def get_rotation(w, h, image):
    #Processing box:  1  Width:  23 Height:  66
    #Processing box:  2  Width:  67 Height:  22
    # Check if the contour is a square

    # remap values from 22-67 to 0-90
    angle = np.interp(w, [22, 76], [0, 90])

    return angle

#function that will test a few pixels and return the color
def isGrabed(image, y, x, width, height):
    # Define the region of interest
    roi = image[y:y+height, x:x+width]

    # Get the intensity values in the ROI
    intensities = roi.reshape(-1)

    #average intensity
    avg = np.mean(intensities)


    return avg


while True:
    #delay for 1 second
    cv2.waitKey(30)
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #treshold the image
    _, tresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    #print(frame.shape)
    #crop picture
    #frame = frame[0:480, 0:640]


    targetx = 450
    targety = 200
    #red dot on the target
    cv2.circle(tresh, (int(targetx), int(targety)), 7, (0, 0, 255), -1)

    grabbed = isGrabed(tresh, targety, targetx, 10, 10)
    
    print("Grabbed ", grabbed)



    # Display the image
    cv2.imshow('frame', tresh)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()'''