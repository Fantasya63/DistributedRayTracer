import cv2
import numpy as np
from numba import cuda

def load_hdr_to_device(path):
    hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Loads .hdr as float32
    if hdr is None:
        raise ValueError(f"Failed to load HDR image from '{path}'")

    # Convert from BGR to RGB
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)

    height, width, _ = hdr.shape
    d_hdr = cuda.to_device(hdr)

    return d_hdr, width, height
