import cv2
import numpy as np
#camera callibration output
camera_matrix = np.array([[644.68813571,   0.0 ,        308.84858405],
                          [  0.0  ,       645.41401477, 238.27188475],
                          [  0.0 ,          0.0 ,          1.0      ]])
dist_coeffs = np.array([-3.19275846e-03,  1.84520855e-01, -6.35263471e-04, -2.54347190e-03, -8.08299110e-01])

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#function that will get rotation of object based on the contour
def get_rotation(w, h, image):
    #Processing box:  1  Width:  23 Height:  66
    #Processing box:  2  Width:  67 Height:  22
    # Check if the contour is a square

    # remap values from 22-67 to 0-90
    angle = np.interp(w, [22, 76], [0, 90])

    return angle



while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    print(frame.shape)

    #treshold the image
    _, frame = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)

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
    



    # Display the image
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()