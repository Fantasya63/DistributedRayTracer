import numpy as np

class Film:
    def __init__(self, width : int, height : int):
        self.width = width
        self.height = height
        self.data = np.zeros((height, width, 3), dtype=np.float32)