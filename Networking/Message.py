import struct
import socket
from enum import Enum
from Log.Logger import *
from ExitCode import *
from Renderer.Renderer import Renderer
from Scene.Scene import Scene
from Scene.SceneSerializer import SceneSerializer
from Renderer.Film import Film
import numpy as np

MAX_PACKET_SIZE = 4096

def GetLocalIP():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]


class CommandHeaders(Enum):
    DISCONNECT = 0
    RENDER = 1
    FILM = 2


def ReceiveCommand(conn : socket.socket):
    header = conn.recv(1)
    header = int.from_bytes(header)
    LogInfo(f"Received Header: {header}")
    if not header:
        LogError("No command header received.")
        return
    
    elif header == CommandHeaders.RENDER.value:
        return ReceiveRenderCommand(conn)
    
    elif header == CommandHeaders.FILM.value:

        return ReceiveFilmCommand(conn)

    else:
        CoreLogError("Unknown Command Header is received.")
        CoreLogError("Exiting....")
        sys.exit(ExitCode.UNKNOWN_COMMAND_HEADER)


def SendRenderCommand(conn : socket.socket, scene_data : str, num_sample : int, num_bounces : int):
    LogInfo("Sending Render Command...")
    try:
        # Send Header
        header_bytes = struct.pack('B', CommandHeaders.RENDER.value)
        conn.send(header_bytes)

        # Send the amount of samples and bounces
        conn.send(struct.pack(">I", num_bounces))
        conn.send(struct.pack(">I", num_sample))


        # Send Scene data
        scene_bytes = scene_data.encode('utf-8')
        scene_size = len(scene_bytes)

        conn.send(struct.pack(">I", scene_size))

        # Send the actual data
        conn.sendall(scene_bytes)

        CoreLogInfo(f"Scene file sent! {scene_size} bytes used.")
        CoreLogInfo(f"Render Command sent! Num of Bounces: {num_bounces}, Num of samples: {num_sample}")
    
    except Exception as e:
        CoreLogError(f"Failed to send render command: {e}")

# UDP - PlayerPos(0) - 
def ReceiveRenderCommand(conn : socket.socket):
    try:
        num_bounces_data = conn.recv(4)
        if len(num_bounces_data) < 4:
            raise ValueError("Num bounces data is incomplete")
        num_bounces = struct.unpack(">I", num_bounces_data)[0]


        num_samples_data = conn.recv(4)
        if len(num_samples_data) < 4:
            raise ValueError("Num samples data is incomplete")
        num_samples = struct.unpack(">I", num_samples_data)[0]
        

        scene_size_data = conn.recv(4)
        if len(scene_size_data) < 4:
            raise ValueError("Scene size data is incomplete")
        

        #  Receive the scene file
        size = struct.unpack(">I", scene_size_data)[0]
        data = b''
        while len(data) < size:
            # Receive MAX_PACKET_SIZE bytes or the amount remaining bytes if the remaining bytes is lessthan MAX_PACKET_SIZE
            chunk = conn.recv(min(MAX_PACKET_SIZE, size - len(data)))
            if not chunk:
                raise ConnectionError("Disconnected during scene file receive.")
            
            data += chunk
        scene_text = data.decode('utf-8')
        CoreLogInfo("Received scene file")
        

        # Parse Scene Text into Scene Object
        scene : Scene = Scene()
        scene_serializer : SceneSerializer = SceneSerializer(scene)

        scene_serializer.DeserializeSceneRuntimeFromString(scene_text)




        # Get the film object from the scene's camera
        film : Film = scene.camera.film
        film.num_samples = num_samples


        # Init Renderer
        renderer : Renderer =  Renderer()

        # Render the scene
        renderer.Render(scene, film, num_samples, num_bounces)

        # Send the Film to the server
        SendFilmCommand(conn, film)


    except Exception as e:
        CoreLogInfo(f"Failed to receive render command: {e}")
        return None


def SendFilmCommand(conn : socket.socket, film : Film):
    try:
        # Send Header
        header_bytes = struct.pack('B', CommandHeaders.FILM.value)
        conn.send(header_bytes)

        # Send the Film width, height, and num_samples
        conn.send(struct.pack(">III", film.width, film.height, film.num_samples))


        # Send the flattened film data as bytes (float32)
        data_bytes = film.data.astype(np.float32).tobytes()
        conn.sendall(data_bytes)
    
    except Exception as e:
        CoreLogInfo(f"Failed to send film command: {e}")
        return None



def ReceiveFilmCommand(conn : socket.socket):
    try:
        width_height_samples_data = conn.recv(12)
        if len(width_height_samples_data) < 12:
            raise ValueError("Film width, height, and num_samples data is incomplete")
    
        width, height, num_samples = struct.unpack(">III", width_height_samples_data)

        expected_bytes = width * height * 3 * 4

        # Receive film data
        received = bytearray()
        while len(received) < expected_bytes:
            chunk = conn.recv(expected_bytes - len(received))
            if not chunk:
                raise ConnectionError("Disconnected during receiving film data")
            received.extend(chunk)

        # Convert bytes back into numpy array
        data = np.frombuffer(received, dtype=np.float32).reshape((height, width, 3))

        # Construct and return Film
        film = Film(width, height, num_samples)
        film.data = data
        return film

    except Exception as e:
        CoreLogInfo(f"Failed to receive film command: {e}")
        return None
