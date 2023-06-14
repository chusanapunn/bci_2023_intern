import socket


# client code
cli_s=None      
TCP_IP = '127.0.0.1'
TCP_PORT=58058
conn=False
def openConnection(TCP_PORT):
    BUFFER_SIZE = 1024
    global cli_s
    global conn
    cli_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        cli_s.connect((TCP_IP, TCP_PORT))
        print("Connection Successful")
        conn=True
    except ConnectionRefusedError:
        print("Connection Refused")
        conn=False
    # finally:
    #     print("Connection Closed")
    #     s.close()
    #     conn=False

def sendMessage(MESSAGE):
    if conn:
        print("Sending Message : "+ str(MESSAGE))
        cli_s.send(MESSAGE.encode())
        #s.close()

def sendCommand(command):
    if conn:
        print("Sending Command : "+str(command))
        cli_s.sendall(command.encode())

def receiveMessage():
    if conn:
        BUFFER_SIZE = 1024
        data = cli_s.recv(BUFFER_SIZE)
        print (u"client received data:", data.decode("utf-8")
)
openConnection(TCP_PORT)
sendCommand("XD")
receiveMessage()
    