import numpy as np

class Sphere:
    def __init__(self, position, radius : float):
        self.position = np.array(position, dtype=np.float32)
        self.radius = radius


class Cube:
    def __init__(self, position, size):
        self.position = np.array(position, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)


class Plane:
    def __init__(self, normal, distance : float):
        self.normal = np.array(normal, dtype=np.float32) / np.linalg.norm(normal)
        self.distance = distance