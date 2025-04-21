import socket
import threading


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
