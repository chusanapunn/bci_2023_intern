import socket               # Import socket module
import time
s_sender = socket.socket()         # Create a socket object
host = "127.0.0.1"              # Bind Local
port = 9030                # Reserve a port for your service.
connected=False
s_sender.bind((host,port))

command="yea"

s_sender.listen(5)

def sendMessage(cmd,c):
   c.send(cmd.encode())
   print("--Sent message--")

try:
    conn, addr =s_sender.accept()
    print ('Open Listening Server', host, port)
    connected=True
     # Establish connection with client.
    print('Got connection from', addr)
except s_sender.error :
    connected=False
    print("Connection DOOM, did you open the Client Yet?")

while connected:
    try:
        time.sleep(1)
        sendMessage(command,conn)
    except ConnectionAbortedError:
        print("Connection Doom` due to something wa")
        connected = False