import numpy as np

class Sphere:
    def __init__(self, position, radius : float, material_id : int):
        self.position = np.array(position, dtype=np.float32)
        self.radius = radius
        self.material_id = material_id


class Cube:
    def __init__(self, position, size, material_id : int):
        self.position = np.array(position, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.material_id = material_id


class Plane:
    def __init__(self, normal, distance : float, material_id : int):
        self.normal = np.array(normal, dtype=np.float32) / np.linalg.norm(normal)
        self.distance = distance
        self.material_id = material_id