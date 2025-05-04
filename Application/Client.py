from Application.Application import Application
from Networking.Message import *
from ExitCode import ExitCode
from Log.UserInput import *
from Log.Logger import *

import sys
import socket
import threading
import time

class ClientApp(Application):
    def __init__(self):
        super().__init__()
        self.client = None
    
    def run(self):
        LogInfo("Client app is created.")
        
        # Get IP Address from the user
        # self.ip_address = GetIPAddress("Enter the ip address of the server.")
        self.ip_address = GetLocalIP()

        # Create a client socket
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        success : bool = self.connect_to_server()
        if not success:
            sys.exit(ExitCode.CANNOT_CONNECT_TO_SERVER)
        
        # Wait for commands
        LogInfo("Recieving Commands...")
        scene_data = ReceiveCommand(self.client)
        LogInfo(f"Scene Data: {scene_data}")



    def connect_to_server(self):
        # The amount of failed connection requests before trying again
        max_connectrion_req = 10
        counter = max_connectrion_req

        while True:        
            LogInfo("Connecting to the server...")
            try:
                self.client.connect((self.ip_address, self.port))
                LogInfo(f"Connected to server at {self.ip_address}:{self.port}")
                return True
            
            except Exception as e:
                LogError(f"Failed to connect to server: {e}")
                counter -= 1
                if counter == 0:
                    user_input = GetChar(f"Failed to to connect to the server {max_connectrion_req} times. Continue connecting? Y/N: ")
                    user_input = user_input.lower()
                    if user_input == "y":
                        counter = max_connectrion_req
                        time.sleep(1.0)
                    else:
                        return False
