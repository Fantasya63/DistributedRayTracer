from numba import cuda
from PIL import Image
import numpy as np

from Scene.Scene import Scene
from Renderer.Film import Film



@cuda.jit(device=True)
def ray_sphere_intersect(ray_origin, ray_dir, sphere_pos, sphere_radius):
    oc = ray_origin - sphere_pos
    a = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
    b = 2.0 * (oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2])
    c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return -1.0
    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    return t if t > 0 else -1.0



@cuda.jit
def generate_gradient(output, width, height):
    x, y = cuda.grid(2)

    if x >= width or y >= height:
        return
   
    
    u = x / (width - 1)  # Normalize x to [0,1]
    v = y / (height - 1)

    # RGB gradient from black (0,0,0) to white (255,255,255)
    output[y, x, 0] = int(u * 255)  # R
    output[y, x, 1] = int(v * 255)  # G
    output[y, x, 2] = int(0 * 255)  # B


class Renderer:
    def __init__(self):
        pass

    def Render(self, scene : Scene, film : Film):
        # Allocate output array on host and device
        output_host = np.zeros((film.height, film.width, 3), dtype=np.uint8)
        output_device = cuda.to_device(output_host)

        # Set up grid and block dimensions
        block_size = (16, 16)
        grid_size = ((film.width + block_size[0] - 1) // block_size[0],
                    (film.height + block_size[1] - 1) // block_size[1])

        # Launch kernel
        generate_gradient[grid_size, block_size](output_device, film.width, film.height)

        # Copy result back to host and save as image
        output_host = output_device.copy_to_host()
        image = Image.fromarray(output_host)
        image.save('gradient.png')