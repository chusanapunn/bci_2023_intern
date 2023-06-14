__author__ = 'Kujira'
import socket               # Import socket module

s_server = socket.socket()         # Create a socket object
host="127.0.0.1"              # Bind Local
port = 13254                # Reserve a port for your service.
s_server.bind((host, port))        # Bind to the port

message='Thank you for connecting'

s_server.listen(5)                 # Now wait for client connection.

def sendMessage(message,c):
   c.send(message.encode())

while True:
   s_client, addr = s_server.accept()     # Establish connection with client.
   print('Got connection from', addr)
   sendMessage(message,s_client)
   # send_timer=QtCore.Qtimer()
   # send_timer.timeout.connect(sendMessage(message,c))
   # send_timer.start(2)
   break
s_server.close()



