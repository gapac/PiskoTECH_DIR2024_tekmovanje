import socket

class TCP_IP():
 
    def __init__(self): # vzpostavitev serverja
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 25000))                # ker je server je IP0.0.0.0, vrata 25000
        s.listen()
        self.conn, addr = s.accept()              # povezava vzpostavljena s klientom
        print(f"Connected by {addr}")
        data = self.conn.recv(1024)
        print(f"Client: {data.decode('ascii')}")  # naslov klienta

    def GetRobotCoordinates(self):                      # prejmi koordinate robota
        data = self.conn.recv(1024)                     # prejme podatke
        data = data.decode('ascii')                     # dekodira asci kodo v string
        print(f"Client: {data}")
        niz = data.replace("p[", "").replace("]", "")   # Odstrani nepotrebne znake
        return  [float(x) for x in niz.split(",")]      # seznam stevil

    def SendRobotCoordinates(self,koordinates): # poslje koordinate robotu
        msg = koordinates                       # [X,Y,Z,Rx,Ry,Rz]
        msg = str(msg)+'\n'                     # spremeni v string
        msg = msg.encode('utf-8')               # zakodiraj v utf-8
        self.conn.send(msg)                     # poslji
        # self.confirm()

    def GetRobotMessage(self):      # prejmi sporocilo robota
        data = self.conn.recv(1024) # prejme podatke
        data = data.decode('ascii') # dekodira asci kodo v string
        return  data


    def SendRobotMessage(self,msg): # poslje sporocilo robotu
        msg = msg.encode('utf-8')   # zakodiraj v utf-8
        self.conn.send(msg)         # poslji
        #self.confirm()


    def confirm(self):              # Potrditev robota
        UR_16_msg = self.GetRobotMessage()
        while UR_16_msg != "1":
            UR_16_msg = self.GetRobotMessage()
        UR_16_msg = 0
