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
    SEND_RENDER_COMMAND = 0
    RECEIVE_FILM_DATA = 1


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
        self.scene_path = GetPath("Enter the filepath to the scene you want to render: ")
        # self.scene_path = "TestData/TestScene.cdscn"

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

        # # Notify the threads to send the scene data for each client
        # with self.lock:
        #     self.command_id = ServerThreadCommands.SEND_RENDER_COMMAND
        #     self.has_command = True
        #     self.command_data = scene_data
        

        # Wait for each thread to finish
        


        # Clear the thread flags
        # self.__clear_thread_flags()

        # Divide the workload to the available nodes
        num_samples_per_thread = self.__divide_work(self.num_samples, self.max_clients)
        thread_ids = list(self.thread_response.keys())

        

        # Send the Render Command
        with self.lock:
            self.command_id = ServerThreadCommands.SEND_RENDER_COMMAND
            self.has_command = True
            self.command_data = {}

            # TODO: Make sure the num of threads is equal to the thread_ids' length
            for i, thread_id in enumerate(thread_ids):
                self.command_data[thread_id] = (num_samples_per_thread[i], scene_data)



        # Wait for the responses
        self.__wait_for_threads_to_finish()

        # Clear thread flags
        self.__clear_thread_flags()

        


        # Receive the clients' film data

        # Used to hold the rendered film for each client
        film_data = {} # Key: Thread_ID, Value: Film Object


        # Issue the command to the threads
        with self.lock:
            self.command_id = ServerThreadCommands.RECEIVE_FILM_DATA

            for thread_id in self.thread_response.keys():
                self.command_data[thread_id] = None
                
            self.command_data = film_data
            self.has_command = True

        self.__wait_for_threads_to_finish()
        self.__clear_thread_flags()

        # Post process
        films = [film_data[thread_id] for thread_id in film_data.keys()]
        combined_film : Film = Renderer.IntegrateFilmsCPU(films)
        combined_film = Renderer.PostProcessFilm(combined_film)
        
        # Save the image to disk
        image = Image.fromarray(combined_film.data)
        image.save('output.png')


    def __wait_for_threads_to_finish(self):
        while True:
            num_finished = 0
            for thread_id in self.thread_response:
                if self.thread_response[thread_id] == True:
                    num_finished += 1


            if num_finished >= self.max_clients:
                break

            time.sleep(0.1)




    def __clear_thread_flags(self):
        # Set the has_command flag to False
        with self.lock:
            self.has_command = False
            self.command_data = None


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
                    _command_data = self.command_data[thread_id]
            
            if parse_command:
                self._parse_thread_command(connection, _command_id, _command_data=_command_data)
                parse_command = False
                _command_id = ServerThreadCommands.INVALID
                self.thread_response[thread_id] = True

                
            
            else:
                time.sleep(0.1)
        

    def _parse_thread_command(self, conn, _command_id, _command_data):
        
        if _command_id == ServerThreadCommands.SEND_RENDER_COMMAND:
            _num_samples, _scene_data = _command_data
            SendRenderCommand(conn, _scene_data, _num_samples, self.num_bounces)

        if _command_id == ServerThreadCommands.RECEIVE_FILM_DATA:
            
            # Wait for Receive FIlm Command
            film : Film = ReceiveCommand(conn)

            _command_data = film
