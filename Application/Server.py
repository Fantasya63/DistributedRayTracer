from Application.Application import Application
from Log.Logger import *
from Log.UserInput import *
from ExitCode import ExitCode
from Networking.Message import *

import socket
import threading
import time

from enum import Enum

class ServerThreadCommands(Enum):
    INVALID = -1
    SEND_SCENE_DATA = 0


class ServerApp(Application):
    def __init__(self):
        super().__init__()

        self.ip_address = GetLocalIP()
        self.num_clients : int = 0
        self.max_clients : int = 0
        self.clients = {}
        self.server = None
        self.connections = []
        self.lock = threading.Lock()
        self.accept_clients : bool = True
        self.has_command : bool = False
        self.thread_response = {}
        self.command_id : int = -1
        self.command_data = None

        self.scene_path : str = ""
        self.num_samples : int = ""


    def run(self):
        LogInfo("Server app is created.")
        LogInfo("")
        LogInfo("Render Properties:")
        # self.scene_path = GetPath("Enter the filepath to the scene you want to render: ")
        self.scene_path = "TestData/TestScene.cdscn"

        self.num_samples = GetPositiveInt("Enter the amount of samples to calculate: ")
        self.num_bounces = GetPositiveInt("Enter the amount of light bounces: ")

        LogInfo("")
        LogInfo("Network Properties:")
        self.max_clients = GetPositiveInt("What is the max client do you want: ")
        
        
        # Start a server at ip_address and port
        ip_and_port = (self.ip_address, self.port)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(ip_and_port)
        self.server.listen(self.max_clients)
        LogInfo(f"Server is started at IP: {self.ip_address}, PORT: {self.port}")
        LogInfo("Waiting for connections...")
        
        # Listen for Clients
        while self.num_clients < self.max_clients:
            conn, addr = self.server.accept()
            self.num_clients += 1
            thread = threading.Thread(target=self.handle_clients, args=(conn, addr))
            thread.start()
        

        LogInfo("Connections are full.")
        LogInfo("Sending Scene data.")
        scene_data = None

        # Read the Scene file
        with open(self.scene_path) as f:
            scene_data = f.read()

        # Notify the threads to send the scene data for each client
        with self.lock:
            self.command_id = ServerThreadCommands.SEND_SCENE_DATA
            self.has_command = True
            self.command_data = scene_data
        

        # Wait for each thread to finish
        while True:
            num_finished = 0
            for thread_id in self.thread_response:
                if self.thread_response[thread_id] == True:
                    num_finished += 1


            if num_finished >= self.max_clients:
                break

            time.sleep(0.1)


        # Clear the thread flags
        self.__clear_thread_flags()

        # Divide the workload to the available nodes
        num_samples_per_thread = self.__divide_work(self.num_samples, self.max_clients)

        # Send the Render Command

        # Wait for the responses






    def __clear_thread_flags(self):
        # Set the has_command flag to False
        with self.lock:
            self.has_command = False


        # Set the thread responses to false
        for thread_id in self.thread_response:
            self.thread_response[thread_id] = False


    def __divide_work(self, total_work, amount_of_workers):
        base = total_work // amount_of_workers
        remainder = total_work % amount_of_workers


        return [(base + 1 if i < remainder else base) for i in range(amount_of_workers)]





    def handle_clients(self, connection, address):
        thread_id = threading.get_ident()
        
        # Initilize this thread's Response flag to false
        self.thread_response[thread_id] = False

        LogInfo(f"Client {address} just connected.")
        LogInfo(f"Total clients: {self.num_clients}")


        parse_command : bool = False
        _command_id : int = ServerThreadCommands.INVALID
        _command_data = None

        # Busy wait for commands
        while True:
            parse_command = False
            _command_id = ServerThreadCommands.INVALID

            with self.lock:
                if self.has_command and self.thread_response[thread_id] == False:
                    parse_command = True
                    _command_id = self.command_id
                    _command_data = self.command_data
            
            if parse_command:
                self._parse_thread_command(connection, _command_id, _command_data=_command_data)
                parse_command = False
                _command_id = ServerThreadCommands.INVALID
                self.thread_response[thread_id] = True

                
            
            else:
                time.sleep(0.1)
        

    def _parse_thread_command(self, conn, _command_id, _command_data):
        if _command_id == ServerThreadCommands.SEND_SCENE_DATA:
            SendSceneFile(conn, _command_data)



# class Server:
#     port = 6376
#     ipAddress = socket.gethostbyname(socket.gethostname())
#     maxClients = 1
#     clients = {}

#     def __init__(self):
#         pass

    


# PORT = 6376
# IP_ADDRESS = socket.gethostbyname(socket.gethostname())
# IP_AND_PORT = (IP_ADDRESS, PORT)

# MAX_CLIENTS = 1
# clients = {}

# print(f"Server started at: IP: {IP_ADDRESS}, PORT: {PORT}")
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(IP_AND_PORT)


# # Runs for each client
# def handle_client(connection, address):
#     print(f"  Client {address} just connected.")

#     connected = True
#     while connected:
#         pass


# def start():
#     print_stats()
#     server.listen(MAX_CLIENTS)

#     num_clients = 0
#     b_Listen = num_clients < MAX_CLIENTS

#     while b_Listen:
#         connection, address = server.accept()
#         thread = threading.Thread(target=handle_client, args=(connection, address))
#         thread.start()

#         print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")

#     pass


# def print_stats():
#     print('Status: ')

#     if len(clients) == 0:
#         print("  - NO CLIENTS CONNECTED - ")
    
#     else:
#         print(clients)
