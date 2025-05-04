from Scene.Scene import Scene
from Scene.RTPrimitives import *

from Log.Logger import LogInfo, LogError, CoreLogError, CoreLogInfo

from Renderer.Camera import Camera
from Renderer.Film import Film
from Renderer.Material import Material
from VectorUtils.Transformations import *


import sys
import yaml
from yaml.loader import SafeLoader

import numpy as np
import math
from pyglm import glm


class SceneSerializer:
    def __init__(self, scene : Scene):
        self.scene = scene

    def DeserializeSceneRuntime(self, _path : str):
        # Open the scene file
        try:
            with open(_path, 'r') as file_stream:
                scene_data = yaml.safe_load(file_stream)
                self.__LoadSceneFromSceneData(scene_data)


        except FileNotFoundError:
            CoreLogError(f"File not fount at: {_path}")
        
        except yaml.YAMLError as err:
            CoreLogError(f"Yaml error: {err}")

    def DeserializeSceneRuntimeFromString(self, scene_string : str):
        try:
            scene_data = yaml.safe_load(yaml_string)
            self.__LoadSceneFromSceneData(scene_data)

        except yaml.YAMLError as err:
            CoreLogError(f"Yaml error: {err}")
        
    # Private Methods
    def __LoadSceneFromSceneData(self, _sceneData):
        scene = self.scene
        scene.name = _sceneData['Scene']


        if not 'Camera' in _sceneData:

            CoreLogError(f"No camera is present in the scene.")
            sys.exit(2)    
        camData = _sceneData['Camera']


        filmData = _sceneData["Film"]
        film : Film = Film(filmData["Width"], filmData["Height"])

        camRot = np.degrees(camData["Rotation"])
        camRot[0] -= 90
        cam = Camera(camData["Position"], camRot, math.degrees(camData["FOV"]), camData["NearPlane"], camData["FarPlane"], film)
        scene.set_camera(cam)
        
        for material_data in _sceneData["Materials"]:
            # material_data = _sceneData["Materials"][i]
            material = Material(
                albedo=material_data["BaseColor"],
                roughness=material_data["Roughness"],
                metallic=material_data["Metallic"],
                emission= glm.pow(material_data["EmissionColor"], glm.vec3(2.2)) * glm.vec3(material_data["EmissionStrength"]))
            
            scene.materials.append(material)
        scene.materials = np.array(scene.materials)
        # scene.materials = np.array(_sceneData["Materials"])

        for entity in _sceneData['Entities']:            
            transform = entity["TransformComponent"]
            
            if "SphereComponent" in entity:
                sphereComponent = entity['SphereComponent']
                _sphere = Sphere(transform["Position"], sphereComponent["Radius"], entity["MaterialID"])
                scene.add_sphere(_sphere)
                continue

            
            if "PlaneComponent" in entity:
                planeComponent = entity['PlaneComponent']
                
                position = glm.vec3(transform["Position"])
                rotation = glm.vec3(transform["Rotation"])
                rot_matrix = EulerToMatrix(rotation)

                normal = (rot_matrix @ glm.vec4(0.0, 1.0, 0.0, 0.0)).xyz
                distance = glm.dot(normal, position)

                _plane = Plane(normal=normal, distance=distance, material_id=entity["MaterialID"])
                scene.add_plane(_plane)
                continue
            # pass