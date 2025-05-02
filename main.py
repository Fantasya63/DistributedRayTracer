
from Renderer.Renderer import Renderer
from Renderer.Film import Film
from Renderer.Camera import Camera
from Scene.Scene import Scene
from Scene.SceneSerializer import SceneSerializer
from Scene.RTPrimitives import Sphere
from Log.Logger import *

import numpy as np
import math

def main():    
    # Image dimensions
    film : Film = Film(width=1920, height=1080)

        
    scene : Scene = Scene()
    serializer : SceneSerializer = SceneSerializer(scene)
    serializer.DeserializeSceneRuntime("TestData/TestScene.cdscn")

    # scene.camera.set_rotation([0.0, 90.0, 0.0])
    scene.camera.set_FOV(90.0)

    rot_matrix = scene.camera.get_rot_matrix()
    LogInfo(f"Forward: {rot_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32) }")
    LogInfo(f"Right  : {rot_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float32)}")
    LogInfo(f"UP     : {rot_matrix @ np.array([0.0, 1.0, 0.0], dtype=np.float32)}")

    renderer = Renderer()
    renderer.Render(scene, film, 1024 * 4, 8)



if __name__ == "__main__":
    main()