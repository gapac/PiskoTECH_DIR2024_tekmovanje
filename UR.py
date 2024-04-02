import cv2
import numpy as np
import TakePicture as tp
import IsGrabbed as ig
import UR_TCP_IP
import os
os.system('cls') # brisanje terminala (windows = cls, linux = clear)





# ________________________________________________MAIN______________________________________________________________
def main():    
    UR = UR_TCP_IP.TCP_IP()

    #Robots_coordinates = UR.GetRobotCoordinates()     # prejme koordinate vrha robota [X,Y,Z,Rx,Ry,Rz]

    #Robots_coordinates[0] = Robots_coordinates[0]+0.2 # premik po X osi
    #Robots_coordinates[1] = Robots_coordinates[1]-0.3 # premik po Y osi
    #Robots_coordinates[2] = Robots_coordinates[2]-0.03 # premik po z osi
    # Robots_coordinates = [] # lokalni koordinatni sistem

    # UR.SendRobotCoordinates(Robots_coordinates)       # poslji koordinate robotu [X,Y,Z,Rx,Ry,Rz]

    # #Home position
    # UR.SendRobotCoordinates(home_pos)
    # while zakljucen_gib != 'zakljucen':
    #    zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zaključen'

    #Pošlji ukaz za zajem slike
    ukaz_zajem_slike = UR.GetRobotMessage()
    print("ukaz")
    #ukaz_zajem_slike = 'ukaz_zajem_slike'
    while ukaz_zajem_slike != "ukaz_zajem_slike":
        ukaz_zajem_slike = UR.GetRobotMessage()
        print(ukaz_zajem_slike)
    #Kličemo ustrezno skripto
    print('slika zajeta')

    if ukaz_zajem_slike == "ukaz_zajem_slike":
        # Read a frame from the webcam
        print(ukaz_zajem_slike)
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        location = tp.TakePicture(frame)

        print("send T1, T2")
        home_pos[2] = home_pos[2]-0.01 # premik po z osi
        home_pos[5] = home_pos[5]-0.6# rot z
        UR.SendRobotCoordinates(location)

    ukaz_zajem_slike = ''


    # #Dobimo točke in robota tja pripeljemo
    # UR.SendRobotCoordinates()#Tu vstavi točko 1
    # while zakljucen_gib != 'zakljucen':
    #     zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zaključen'
    # UR.SendRobotCoordinates()#Tu vstavi točko 2
    # while zakljucen_gib != 'zakljucen':
    #     zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zakljucen'

    # #Kliči skripto za preverjanje smeri prijema
    # while ukaz_preveri_prijem != 'preveri_prijem':
    #     ukaz_preveri_prijem = UR.GetRobotMessage
    # ukaz_preveri_prijem = 'ne preveri'

    # #Dvigni obdelovanec
    # UR.SendRobotCoordinates(obdelovanec_dvig_varna)
    # while zakljucen_gib != 'zakljucen':
    #     zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zakljucen'

    # #odloži na paleti
    # UR.SendRobotCoordinates(nad_paleto_varna)
    # while zakljucen_gib != 'zakljucen':
    #     zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zakljucen'
    # UR.SendRobotCoordinates(v_paleti)
    # while zakljucen_gib != 'zakljucen':
    #     zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zakljucen'
    # UR.SendRobotCoordinates(nad_paleto_varna)
    # while zakljucen_gib != 'zakljucen':
    #     zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zakljucen'

    # #nazaj na zajem slike
    # UR.SendRobotCoordinates(home_pos)
    # while zakljucen_gib != 'zakljucen':
    #    zakljucen_gib = UR.GetRobotMessage
    # zakljucen_gib = 'ni zaključen'




#spremenljivke
ukaz_zajem_slike = ""
ukaz_preveri_prijem = ""
test_msg = 'ni ok'
ukaz_zajem_slike = 'ne zajemi'
zakljucen_gib = 'ni zakljucen'

home_pos = [0.39128, -0.23382, 0.32238, 1.611, 0.027, -0.059]
nad_paleto_varna = []
v_paleti = []

ukaz_zajem_slike = ''

# Open the webcam
print("capture")

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
#show picture
#while True:
    #ret, frame = cap.read()
   # #breakpoint()
   # cv2.imshow('frame', frame)

# Release the webcam and destroy all windows
#frame.release()
#cv2.destroyAllWindows()
# #testne točke
# nad_obdelovanec_varna = []
# obdelovanec_poravnava = []
# obdelovanec_prijem = []
# obdelovanec_dvig_varna = []


if __name__ == "__main__":  
    main()


