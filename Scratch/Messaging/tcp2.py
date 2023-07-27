# Example client program in Python that receives the current server time

import socket

 

# IP address of the server

serverIP    = "Kujira"

 

# Port number of the server

serverPort  = 13254

# Create a tcp socket

tcpSocket   = socket.socket()

 

try:

    # Connect tcp client socket to the tcp server socket

    tcpSocket.connect((serverIP, serverPort))

 

    # Receive data from server (i.e., current server time)

    timeReceived = tcpSocket.recv(1024)

   

    # Print the data received

    print(timeReceived.decode())

except Exception as Ex:

    print("Exception Occurred: %s"%Ex)

 

    # Close the socket upon an exception

    tcpSocket.close()