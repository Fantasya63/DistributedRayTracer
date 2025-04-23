from numba import cuda
from PIL import Image
import numpy as np
import math

from Scene.Scene import Scene
from Renderer.Film import Film
from Renderer.RTUtils import *



@cuda.jit(device=True)
def ray_sphere_intersect(ray_origin, ray_dir, sphere_pos, sphere_radius):
    oc = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        oc[i] = ray_origin[i] - sphere_pos[i]

    # oc = ray_origin - sphere_pos
    a = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
    b = 2.0 * (oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2])
    c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return -1.0
    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    return t if t > 0 else -1.0




@cuda.jit
def Trace(output, width, height, sphere, camPos, viewProj, invViewProj):
    x, y, = cuda.grid(2)

    if x >= width or y >= height:
        return
   
    aspect_ratio = width / height
    ndc_x = ((x + 0.5) / width) * 2.0 - 1.0
    ndc_y = ((y + 0.5) / height) * 2.0 - 1.0
   
    # Create ray direction in clip space
    ray_clip = cuda.local.array(4, dtype=np.float32)
    ray_clip[0] = ndc_x
    ray_clip[1] = ndc_y
    ray_clip[2] = -1.0  # Pointing into the scene
    ray_clip[3] = 1.0

    # Transform to world space (inverse view-projection)
    # Note: For simplicity, we're not computing the full inverse here
    ray_eye = cuda.local.array(4, dtype=np.float32)
    mat4x4_vec4_multiply(invViewProj, ray_clip, ray_eye)


    # Perspective Division
    for i in range(3):
        ray_eye[i] /= ray_eye[3]
    ray_eye[3] = 1.0

    # Convert to direction (w = 0 for direction vectors)
    ray_world = cuda.local.array(4, dtype=np.float32)
    for i in range(3):
        ray_world[i] = ray_eye[i] - camPos[i]
    
    ray_world[3] = 0.0


    # Normalize direction
    ray_dir = cuda.local.array(3, dtype=np.float32)
    normalize_vec3(ray_world[:3], ray_dir)
    
    sphere_center = sphere[:3]
    sphere_rad = sphere[3]

    ray_origin = camPos
    t = -1.0
    t = ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_rad)



    temp = cuda.local.array(3, dtype=np.float32)
    # temp[0] = x / width
    # temp[1] = y / height
    # temp[2] = 0.0

    temp[0] = ray_dir[0]
    temp[1] = ray_dir[1]
    temp[2] = ray_dir[2]

    if t != -1.0:
        temp[0] = 1.0
        temp[1] = 1.0
        temp[2] = 1.0



    rgbOut = cuda.local.array(3, dtype=np.uint8)
    Float3ToRGB(rgbOut, temp)

    for i in range(3):
        output[y, x, i] = rgbOut[i]


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
       
        camera = scene.camera
        cudaCamPos = cuda.to_device(camera.position)
        cudaViewProj = cuda.to_device(camera.viewProjection)
        cudaInvViewProj = cuda.to_device(np.linalg.inv(camera.viewProjection))


        numSphere = len(scene.spheres)
        # spherePos = np.array([])

        # Position Rad
        # sphere_arr = np.array([0.0, 0.0, 20.0, 1.0], dtype=np.float32)
        # cudaSphere = cuda.to_device(sphere_arr)

        # spherePos = np.array([sphere.position for sphere in scene.spheres])
        # sphereRad = np.array([sphere.radius for sphere in scene.spheres])
        # cudaSpherePos = cuda.to_device(spherePos)
        # cudaSphereRad = cuda.to_device(sphereRad)
        # cudaOutData = cuda.to_device(film.data)
        
        Trace[grid_size, block_size](output_device, film.width, film.height, cudaSphere, cudaCamPos, cudaViewProj, cudaInvViewProj)

        # Copy result back to host and save as image
        output_host = output_device.copy_to_host()
        image = Image.fromarray(output_host)
        image.save('gradient.png')