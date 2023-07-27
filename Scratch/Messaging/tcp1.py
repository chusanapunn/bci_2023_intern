# Example server program in Python that serves clients

# with the current server time

 

# import the required Python modules

import socket

import datetime

 

# Create a TCP server socket

serverSocket = socket.socket()

 

# Bind the tcp socket to an IP and port

serverSocket.bind(("Kujira", 13254))

 

# Keep listening
serverSocket.listen()

 

while(True):

    # Keep accepting connections from clients

    (clientConnection, clientAddress) = serverSocket.accept()

 

    # Send current server time to the client

    serverTimeNow = "%s"%datetime.datetime.now()

    clientConnection.send(serverTimeNow.encode())

    print("Sent %s to %s"%(serverTimeNow, clientAddress))

 

    # Close the connection to the client

    clientConnection.close()