import pickle
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R


class Image():
  
    def Rotate(self,img,fi):
        #funkcija rotira sliko okoli njenega sredisca

        # Naložite sliko
        originalna_slika = cv.imread(img)

        # Določite središče slike (točka okoli katere se bo slika vrtela)
        visina, sirina = originalna_slika.shape[:2]
        sredisce = (sirina / 2, visina / 2)

        # Določite kot rotacije
        rotation_angle = fi  # nadomestite s kotom, ki ga želite uporabiti

        # Izračunajte rotacijsko matriko
        M = cv.getRotationMatrix2D(sredisce, rotation_angle, 1.0) # Brez spremembe velikosti

        # Izvedite rotacijo
        rotated_img = cv.warpAffine(originalna_slika, M, (sirina, visina))
    
        # Shranite zavrtano sliko
        cv.imwrite('image.bmp', rotated_img)


class Ur16():
    # Funkcija prejme kote zasuka Roll, Pitch, Yaw in pa koordinate vrha robota 
    # Vrne nove koordinate vrha robota
    def RotateTool(self,Roll, Pitch, Yaw,Robots_coordinates):
        Ori = [Roll,Pitch,Yaw]

        # Rotacijski vektor (rx, ry, rz)
        rotation_vector = np.array(Robots_coordinates[3:]).reshape((3, 1))# Rotacijski vektor (rx, ry, rz) orientacije vrha robota
        rpy_rotation = R.from_euler('xyz', Ori)                           # 
        rpy_rotation_matrix = rpy_rotation.as_matrix()                    # Pretvorba RPY rotacije v rotacijsko matriko
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)                # Pretvorba rotacijskega vektorja v rotacijsko matriko
        combined_rotation_matrix = rotation_matrix @ rpy_rotation_matrix  # Množenje z obstoječo rotacijsko matriko vrha robota
        final_rotation_vector, _ = cv.Rodrigues(combined_rotation_matrix) # Izračun globalnih koordinat rotirane točke z uporabo nove kombinirane rotacijske matrike
        final_orientation = final_rotation_vector.flatten()               # Končna orientacija točke v globalnem koordinatnem sistemu
        Robots_coordinates[3:] = final_orientation                        # Zapis orientacije 
        return Robots_coordinates
   
  
class Points():
    def __init__(self,fix,fiy,fiz) -> None:
        # Transformacija iz K.S. Kamere v K.S. Vrh robota 
        #///////////////////////////////////////////////////////////////
        # Rotacija po x osi
        self.R_x = np.array([[1, 0, 0, 0],
                        [0, np.cos(fix), -np.sin(fix), 0],
                        [0, np.sin(fix), np.cos(fix), 0],
                        [0, 0, 0, 1]])
        # Rotacija po y osi
        self.R_y = np.array([[np.cos(fiy), 0, np.sin(fiy), 0],
                [0, 1, 0, 0],
                [-np.sin(fiy), 0, np.cos(fiy), 0],
                [0, 0, 0, 1],])
        # Rotacija po z osi
        self.R_z = np.array([[np.cos(fiz), -np.sin(fiz), 0, 0],
                        [np.sin(fiz), np.cos(fiz), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        with open('calibration_data_poz11.pkl', 'rb') as f:
            loaded_calib_data = pickle.load(f)
        self.CameraMatrix = loaded_calib_data['camera_matrix']
        dist = loaded_calib_data['dist_coeff']
        rvecs = loaded_calib_data['rotation_vectors']
        tvecs = loaded_calib_data['translation_vectors']


        pass
    
    def Kam2Glob(self, Objects, Robots_coordinates,KalibracijaX,KalibracijaY):
        XYfi = []
        # for zanka gra skozi vse zaznane prikljucke in ponovno fotografira vsak prikljucek
        for object in Objects:
            # Transformacija iz K.S. kamere v vrh robota
            #////////////////////////////////////////////////////////////////////////////////////
            Point = np.array([0.0, 0.0, 0.0, 1])
            if len(object)>2:
                # Detekcija 1
                x, y, fi, fi_rad = object   # branje parametrov objekta
            else:
                # detekcija 2 ne vrne kotov
                x, y, = object
                fi = 0
                fi_rad = 0

            # img = cv.imread('image.bmp')
            # cv.circle(img, (int(x),int(y)), 5, (0,0,255), -1)
            # cv.namedWindow('frame',cv.WINDOW_NORMAL)  # incializacija okna slike
            # cv.resizeWindow('frame', 800,600)          # velikost okna za prikaz slike
            # cv.imshow('frame', img) 
            # cv.waitKey(0)


            Point[0] = int(x) - (1280/2) # odmik iz sredisca po x osi
            Point[1] = int(y) - (1024/2) # odmik iz sredisca po y osi
            Point[0] = (Point[0]*KalibracijaX)/1000 # /1000 pretvotba v metre
            Point[1] = (Point[1]*KalibracijaY)/1000   # /1000 pretvotba v metre

            
            local_point = self.R_x @ self.R_y @ self.R_z @ Point
            #//////////////////////////////////////////////////////////////////////////////////ž
         

            # Transformacija iz K.S. Vrha robota v globalni K.S. 
            #//////////////////////////////////////////////////////////////////////
            # Trenutni polozaj vrha robota 
            position = Robots_coordinates[:3]                                   # Položaj vrha robota (X_r, Y_r, Z_r)
            rotation_vector = np.array(Robots_coordinates[3:])                  # Rotacijski vektor (rx, ry, rz)
            rotation_vector = np.array(Robots_coordinates[3:]).reshape((3, 1))  # Rotation vector (rx, ry, rz)
            
            # Pretvorba rotacijskega vektorja v rotacijsko matriko
            rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        
            
            # Izračun globalnih koordinat točke z uporabo rotacijske matrike
            final_position = np.dot(rotation_matrix, local_point[:3]) + position
            XYfi.append([final_position[0],final_position[1],fi,fi_rad])
        return XYfi

    def Distance(self, point1, point2):
        # Funkcija prejme koordinate tock v 2D in vrne razdaljo med nijima
        Distance =(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        return Distance

    def Rotate(self, objects, img, rotation_angle):
        # funkcija rotira koordinate tock na sliki
        # funkcija se uporablja ker se pri 2. zaznavanjem sliko rotira in je zato potrebno potem tocke zarotirati nazaj
        points_before_rotation = []
        points = []

        for object in objects:
            height, width = img.shape[:2]                       # visina slike
            sredisce = (width / 2, height / 2)                  # sirina slike
            points_before_rotation = [object[0]+35,object[1],1] # zamik tocke po x osi
       
            # Izračun rotacijske matrike
            M = cv.getRotationMatrix2D(sredisce, rotation_angle, 1.0) 
            points.append(np.round(M @ points_before_rotation).astype(int)) # dodajanje tock v tabelo points

        return points

    def Sequence(self, list):
        # funkcija prejme seznam tock za varjenje in vrne zaporedje v obliki seznama indeksov 
        
        arr = []
        # for zanka natedi seznam ki vsebuje indekse tock
        for j in range(len(list)):
            arr.append(j)
        
        # v while zanki se v seznam sequence zapise zaporedje
        sequence = []
        i = 0
        while (3*len(arr)//4+i) < len(arr):
            if arr[i] not in sequence:
                k = arr[i]
                sequence.append(k)
                
            if arr[len(arr)//2+i] not in sequence:
                k = arr[len(arr)//2+i]
                sequence.append(k)
            
            if arr[len(arr)//4+i] not in sequence:
                k = arr[len(arr)//4+i]
                sequence.append(k)
                
            if arr[3*len(arr)//4+i] not in sequence:
                k = arr[3*len(arr)//4+i]
                sequence.append(k)

            i = i+1
        return sequence
    
    def WrapToPi(self,angle):
        """
        Normalizira kot na obseg od -pi do pi.

        :param angle: Kot v radianih, ki ga je treba normalizirati.
        :return: Normaliziran kot med -pi in pi.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi