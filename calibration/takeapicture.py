import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
#camera callibration output
#robo vid
from pygrabber.dshow_graph import FilterGraph

def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

cameras = get_available_cameras()
camera_port = -1

for i in range(len(cameras)):
    if cameras[i] == 'C922 Pro Stream Webcam':
        camera_port = i

#get camera name
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
print("capture done")
start_time = time.time()
frame_id = 0

    # Read a frame from the webcam
ret, frame = cap.read()
print(frame)

    # # Convert the frame to grayscale
    # #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if time.time() - start_time > 3:

    #     if ret:
    #         # Save the resulting frame
    #         cv2.imwrite('frame{}.png'.format(frame_id), frame)
    #         frame_id += 1

    #         # Display the resulting frame
    #         cv2.imshow('frame', frame)

    #     # Wait for 3 seconds
    #     start_time = time.time()

    #     # Break the loop if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    #print(frame.shape)
#crop image 
frame = frame[10:400,10:600]

# Display the image with mat plot lib
plt.figure()
plt.title('Webcam')
plt.imshow(frame)
plt.show()

    # Display the image
    # Break the loop if 'q' is pressed
# Release the video capture object

# Close all OpenCV windows
cv2.destroyAllWindows()