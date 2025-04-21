from Scene.Scene import Scene, Camera
from Log.Logger import LogInfo, LogError


import sys
import yaml
from yaml.loader import SafeLoader


class SceneSerializer:
    def __init__(self, scene : Scene):
        pass

    def DeserializeSceneRuntime(self, _path : str):
        # Open the scene file
        try:
            with open(_path, 'r') as file_stream:
                scene_data = yaml.safe_load(file_stream)
                self.__LoadSceneFromSceneData(scene_data)


        except FileNotFoundError:
            LogError(f"File not fount at: {_path}")
        
        except yaml.YAMLError as err:
            LogError(f"Yaml error: {err}")

    
    # Private Methods
    def __LoadSceneFromSceneData(self, _sceneData):
        scene = Scene()
        scene.name = _sceneData['Scene']
        
        camData = _sceneData['Camera']
        if not camData:
            LogError(f"No camera is present in the scene.")
            sys.exit(1)    

        cam = Camera()
        # ------------------- Continue here setip camera

        for entity in _sceneData['Entities']:            
            
            sphereComponent = entity['SphereComponent']
            if (sphereComponent):
                pass
        
            boxComponent = entity['BoxComponent']
            if boxComponent:
                pass

            planeComponent = entity['PlaneComponent']
            if planeComponent:
                pass
            
            pass