import socket               # Import socket module
from pyqtgraph.Qt import QtCore
import time

s_server = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 13254                # Reserve a port for your service.
connected=False


try:
    print ('Try Connecting to ', host, port)
    connected=True
    s_server.connect((host, port))
except ConnectionRefusedError:
    connected=False
    print("Connection DOOM, did you open the Client Yet?")

while connected:
  msg = s_server.recv(1024)
  print ('SERVER >> '+msg.decode())
  break
#s.close   
s_server.close()