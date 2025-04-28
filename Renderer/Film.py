import numpy as np

class Film:
    def __init__(self, width : int, height : int, num_samples : int = 0):
        self.width = width
        self.height = height
        self.num_samples = num_samples
        self.data = np.zeros((height, width, 3), dtype=np.float32)