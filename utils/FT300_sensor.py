import socket

HOST = "192.168.1.102"
PORT = 63351
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("Connecting to " + HOST)
    
s.connect((HOST, PORT))

while (1):
    data = str(s.recv(1024),"utf-8").replace("(","").replace(")","").split(",")
    data = [float(x) for x in data]
    print(data)    



