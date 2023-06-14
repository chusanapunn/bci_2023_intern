import socket               # Import socket module

s_sender = socket.socket()         # Create a socket object
host = "127.0.0.1"              # Bind Local
port = 9030                # Reserve a port for your service.
connected=False
s_sender.bind((host,port))

command="yea"

s_sender.listen(5)

def sendMessage(cmd,c):
   c.send(cmd.encode())

try:
    print ('Open Listening Server', host, port)
    connected=True
    s_sender.accept()
except s_sender.error :
    connected=False
    print("Connection DOOM, did you open the Client Yet?")

while True:
    conn, addr = s_sender.accept()     # Establish connection with client.
    print('Got connection from', addr)
    sendMessage(command,conn)