from pickle import TRUE
import cv2
import os

os.system('cls')

# def find_external_camea_index():

#     for i in range(100):
#         cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
#         if cap.isOpened():

#             backend_name = cap.getBackendName()
#             if "usb" in backend_name.lower():
#                 print(f"External camera found at indes {i}")
#                 cap.release()
#                 return i
#             cap.release()
#     print("Error: No external camera found")
#     return None

# external_camera_index = find_external_camea_index()
# if external_camera_index is None:
#     exit()

cap = cv2.VideoCapture(1)
while(TRUE):
    ret,frame =cap.read()
    cv2.imshow('object detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.relase()
cv2.destroyAllWindows()
