# renderer.py

from numba import cuda
from PIL import Image
import numpy as np
import math
import sys
from datetime import datetime
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32
)

from Scene.Scene import Scene
from Renderer.Film import Film
from Renderer.RTUtils import *
from Log.Logger import *
from numba.cuda.cudadrv.error import CudaSupportError
from Renderer.ImageLoader import *

@cuda.jit(device=True)
def ray_sphere_intersect(ray_origin, ray_dir, sphere_pos, sphere_radius, sphere_index, sphere_material_indices, materials, best_hit):
    oc = cuda.local.array(3, dtype=np.float32)
    
    for i in range(3):
        oc[i] = ray_origin[i] - sphere_pos[i]
    
    # oc = ray_origin - sphere_pos
    a = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
    b = 2.0 * (oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2])
    c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0.0:
        return

    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    
    best_distance = best_hit[2, 0]
    if t > 0.0 and t < best_distance:
        # Position
        pos = cuda.local.array(3, dtype=np.float32)
        pos[0] = ray_origin[0] + t * ray_dir[0]
        pos[1] = ray_origin[1] + t * ray_dir[1]
        pos[2] = ray_origin[2] + t * ray_dir[2]


        # Normal
        normal = cuda.local.array(3, dtype=np.float32)
        # vec3_sub(pos, sphere_pos, normal)
        # vec3_normalize(normal, normal)

        normal[0] = (pos[0] - sphere_pos[0]) / sphere_radius
        normal[1] = (pos[1] - sphere_pos[1]) / sphere_radius
        normal[2] = (pos[2] - sphere_pos[2]) / sphere_radius

        material_data = materials[sphere_material_indices[sphere_index]] 
        # MAterial Data
        for i in range(3):
            best_hit[0, i] = pos[i]
            best_hit[1, i] = normal[i]
            best_hit[3, i] = material_data[0, i]
            best_hit[4, i] = material_data[1, i]

        # Hit data:
        # [0] - position
        # [1] - normal
        # [2] - [0] distance, roughness[1], metallic[2]
        # [3] - albedo
        # [4] - emission

        # Distance
        best_hit[2, 0] = t # Distance
        best_hit[2, 1] = material_data[2, 0]
        best_hit[2, 2] = material_data[2, 1]



@cuda.jit(device=True)
def energy(vec3_color):
    return (vec3_color[0] * 1.0 / 3.0) + (vec3_color[1] * 1.0 / 3.0) + (vec3_color[2] * 1.0 / 3.0)


@cuda.jit(device=True)
def ray_plane_intersect(ray_origin, ray_dir, plane_normal, plane_dist, plane_index, plane_material_indices, materials, best_hit):
    point = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        point[i] = plane_normal[i] * plane_dist

    ray_origin_minus_point = cuda.local.array(3, dtype=np.float32)
    
    vec3_sub(point, ray_origin, ray_origin_minus_point)

    t = vec3_dot(plane_normal, ray_origin_minus_point)
    NdotD = vec3_dot(plane_normal, ray_dir)
    
    if abs(NdotD) > 1e-6:  # Only proceed if not near zero
        t /= NdotD
    else:
        return


    best_distance = best_hit[2, 0]
    if t > 0.0 and t < best_distance:
        # Position
        pos = cuda.local.array(3, dtype=np.float32)
        pos[0] = ray_dir[0] * t
        pos[1] = ray_dir[1] * t
        pos[2] = ray_dir[2] * t

        pos[0] += ray_origin[0]
        pos[1] += ray_origin[1]
        pos[2] += ray_origin[2]


        
        material_data = materials[plane_material_indices[plane_index]] 
        # MAterial Data
        for i in range(3):
            best_hit[0, i] = pos[i]
            best_hit[1, i] = plane_normal[i]
            # best_hit[1, i] = plane_material_indices[i]
            best_hit[3, i] = material_data[0, i]
            best_hit[4, i] = material_data[1, i]

        # Hit data:
        # [0] - position
        # [1] - normal
        # [2] - [0] distance, roughness[1], metallic[2]
        # [3] - albedo
        # [4] - emission

        # Distance
        best_hit[2, 0] = t # Distance
        best_hit[2, 1] = material_data[2, 0]
        best_hit[2, 2] = material_data[2, 1]



@cuda.jit(device=True)
def generate_camera_rays(ndc_x, ndc_y, inv_view_proj, cam_pos, output):
    # Create ray direction in clip space
    ray_clip = cuda.local.array(4, dtype=np.float32)
    ray_clip[0] = ndc_x
    ray_clip[1] = ndc_y
    ray_clip[2] = 1.0  # Pointing into the scene
    ray_clip[3] = 1.0


    # Transform to world space (inverse view-projection)
    # Note: For simplicity, we're not computing the full inverse here
    ray_eye = cuda.local.array(4, dtype=np.float32)
    mat4x4_vec4_multiply(inv_view_proj, ray_clip, ray_eye)

    # Perspective Division
    for i in range(3):
        ray_eye[i] /= ray_eye[3]
    ray_eye[3] = 1.0



    for i in range(3):
        output[i] = ray_eye[i] - cam_pos[i]

    vec3_normalize(output, output)


@cuda.jit(device=True)
def IntersectScene(ray_origin, ray_dir, sphere_position, sphere_radius, sphere_material_indices, num_spheres, plane_normal, plane_dist, plane_material_indices, num_planes, materials, hit_data):
    # Hit data:
    # [0] - position
    # [1] - normal
    # [2] - [0] distance, roughness[1], metallic[2]
    # [3] - albedo
    # [4] - emission


    # Intersect Spheres
    for i in range(num_spheres):
        sphere_center = sphere_position[i][:3]
        sphere_rad = sphere_radius[i]
        ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_rad, i, sphere_material_indices, materials, hit_data)

    # Intersect Planes
    for i in range(num_planes):
        _plane_normal = plane_normal[i][:3]
        _plane_dist = plane_dist[i]

        ray_plane_intersect(ray_origin, ray_dir, _plane_normal, _plane_dist, i, plane_material_indices, materials, hit_data)


@cuda.jit(device=True)
def Shade_Test(ray_origin, ray_dir, ray_energy, hit_data, sky_hdr, sky_width, sky_height, rng_states, thread_id, output):
    distance = hit_data[2, 0]

    light_dir = cuda.local.array(3, dtype=np.float32)
    light_dir[0] = 1.0
    light_dir[1] = 1.0
    light_dir[2] = -1.0
    vec3_normalize(light_dir, light_dir)

    if distance < math.inf:
        # hit_specular = 1.0 - hit_data[2, 1]
        
        hit_specular = cuda.local.array(3, dtype=np.float32)
        for i in range(3):
            hit_specular[i] = 0.6


        hit_pos = hit_data[0]
        hit_normal = hit_data[1, :3]
        hit_color = hit_data[3, :3]
        hit_emission = hit_data[4]

        ray_origin[0] = hit_pos[0] + (hit_normal[0] * 0.001)
        ray_origin[1] = hit_pos[1] + (hit_normal[1] * 0.001)
        ray_origin[2] = hit_pos[2] + (hit_normal[2] * 0.001)

        diff = cuda.local.array(3, dtype=np.float32)
        l_dot_n = vec3_dot(hit_normal, light_dir)

        for i in range(3):
            diff[i] = max(0.0, l_dot_n) * hit_color[i]

        ray_dir_ref_hit_norm = cuda.local.array(3, dtype=np.float32)
        vec3_reflect(ray_dir, hit_normal, ray_dir_ref_hit_norm)
        vec3_normalize(ray_dir_ref_hit_norm, ray_dir)


        for i in range(3):
            ray_energy[i] *= hit_specular[i]
        
        # output[0] = hit_emission[0]
        # output[1] = hit_emission[1]
        # output[2] = hit_emission[2]
        
    else:
        for i in range(3):
            ray_energy[i] = 0.0

        sample_hdr(ray_dir, sky_hdr, sky_width, sky_height, output)
        

@cuda.jit(device=True)
def Shade(ray_origin, ray_dir, ray_energy, hit_data, sky_hdr, sky_width, sky_height, rng_states, thread_id, output):
    # Hit data:
    # [0] - position
    # [1] - normal
    # [2] - [0] distance, roughness[1], metallic[2]
    # [3] - albedo
    # [4] - emission


    distance = hit_data[2, 0]
    if distance < math.inf:
        hit_smoothness = 1.0 - hit_data[2, 1]
        hit_specular = 0.04
        hit_pos = hit_data[0]
        hit_normal = hit_data[1, :3]
        
        hit_color = cuda.local.array(3, np.float32)

        for i in range(3):
            hit_color[i] = min(1.0 - hit_specular, hit_data[3, i])
        
        spec_chance = hit_specular
        diff_chance = energy(hit_color)

        ray_origin[0] = hit_pos[0] + hit_normal[0] * 0.001
        ray_origin[1] = hit_pos[1] + hit_normal[1] * 0.001
        ray_origin[2] = hit_pos[2] + hit_normal[2] * 0.001

        roulette = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if roulette < spec_chance:
            ray_origin[0] = hit_pos[0] + hit_normal[0] * 0.001
            ray_origin[1] = hit_pos[1] + hit_normal[1] * 0.001
            ray_origin[2] = hit_pos[2] + hit_normal[2] * 0.001


            # Specular reflection
            alpha = smoothness_to_phong_alpha(hit_smoothness)
            
            ray_dir_ref_hit_norm = cuda.local.array(3, dtype=np.float32)
            vec3_reflect(ray_dir, hit_normal, ray_dir_ref_hit_norm)
            # vec3_normalize(ray_dir_ref_hit_norm, ray_dir_ref_hit_norm)
            sample_hemisphere_test(ray_dir_ref_hit_norm, alpha, rng_states, thread_id, ray_dir)
            vec3_normalize(ray_dir, ray_dir)

            f = (alpha + 2) / (alpha + 1)
            
            for i in range(3):
                ray_energy[i] *= (1.0 / spec_chance) * hit_specular * sdot_fast(hit_normal, ray_dir, f)
        
        elif diff_chance > 0.0 and roulette < spec_chance + diff_chance:
            # Diffuse reflection
            ray_origin[0] = hit_pos[0] + hit_normal[0] * 0.001
            ray_origin[1] = hit_pos[1] + hit_normal[1] * 0.001
            ray_origin[2] = hit_pos[2] + hit_normal[2] * 0.001

            sample_hemisphere_test(hit_normal, 1.0, rng_states, thread_id, ray_dir)
            vec3_normalize(ray_dir, ray_dir)

            for i in range(3):
                ray_energy[i] *= (1.0 / diff_chance) * hit_color[i]
        else:
            ray_energy[0] = 0.0
            ray_energy[1] = 0.0
            ray_energy[2] = 0.0

        hit_emission = hit_data[4]

        output[0] = hit_emission[0]
        output[1] = hit_emission[1]
        output[2] = hit_emission[2]
    else:
        for i in range(3):
            ray_energy[i] = 0.0
            output[i] = 0.0
        
        sample_hdr(ray_dir, sky_hdr, sky_width, sky_height, output)

@cuda.jit
def Trace(output, width, height, sphere_position, sphere_radius, sphere_material_indices, num_spheres, 
    plane_normal, plane_dist, plane_material_indices, num_planes,
    materials, camPos, viewProj, inv_view_proj, num_samples, num_bounces,
    sky_hdr, sky_width, sky_height,
    rng_states):
    
    x, y, = cuda.grid(2)
    thread_id = y * height + x
    if x >= width or y >= height:
        return

    
    aspect_ratio = width / height
    

    ray_dir = cuda.local.array(3, dtype=np.float32)
    ray_origin = cuda.local.array(3, dtype=np.float32)


    out_color = cuda.local.array(3, dtype=np.float32)
    out_color[0] = 0.0
    out_color[1] = 0.0
    out_color[2] = 0.0


    # Hit data:
    # [0] - position
    # [1] - normal
    # [2] - [0] distance, roughness[1], metallic[2]
    # [3] - albedo
    # [4] - emission
    for i in range(num_samples):
        pixel_offset_x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        pixel_offset_y = xoroshiro128p_uniform_float32(rng_states, thread_id)

        ndc_x = ((x + pixel_offset_x) / width) * 2.0 - 1.0
        ndc_y = ((y + pixel_offset_y) / height) * 2.0 - 1.0
        ndc_y *= -1.0

        
        # # Normalize direction
        generate_camera_rays(ndc_x, ndc_y, inv_view_proj, camPos, ray_dir)
        
        for i in range(3):
            ray_origin[i] = camPos[i]

        ray_energy = cuda.local.array(3, dtype=np.float32)
        ray_energy[0] = 1.0
        ray_energy[1] = 1.0
        ray_energy[2] = 1.0


        
        hit_data = cuda.local.array(shape=(5, 3), dtype=np.float32)
        for j in range(5):
            for k in range(3):
                hit_data[j, k] = 0.0

        for i in range(num_bounces):

          
            hit_data[2, 0] = math.inf
            hit_data[2, 1] = math.inf
            hit_data[2, 2] = math.inf


            IntersectScene(ray_origin, ray_dir, sphere_position, sphere_radius, sphere_material_indices, num_spheres, plane_normal, plane_dist, plane_material_indices, num_planes, materials, hit_data)

            temp = cuda.local.array(3, dtype=np.float32)
            
            curr_energy = cuda.local.array(3, dtype=np.float32)
            for i in range(3):
                curr_energy[i] = ray_energy[i]

            Shade(ray_origin, ray_dir, ray_energy, hit_data, sky_hdr, sky_width, sky_height, rng_states, thread_id, temp)
            
           
            out_color[0] += curr_energy[0] * temp[0]        
            out_color[1] += curr_energy[1] * temp[1]        
            out_color[2] += curr_energy[2] * temp[2]        



            # Test
            # if hit_data[2, 0] > 0.0 and hit_data[2, 0] < math.inf:
            #     for i in range(3):
            #         out_color[i] += hit_data[1, i] * 0.5 + 0.5
            # else:
            #     for i in range(3):
            #         out_color[i] += 0.0
        
            # TODO: Remove this after testing
            # if (vec3_dot(ray_energy, ray_energy) < 0.001):
            #     break


    out_color[0] /= float(num_samples)
    out_color[1] /= float(num_samples)
    out_color[2] /= float(num_samples)

    
    # filmic_tonemap(out_color, out_color)
    # Float3ToRGB(output[y, x, :3], out_color)
    output[y, x, :3] = out_color



class Renderer:
    def __init__(self):
        CoreLogInfo("Initializing Renderer...")
        if cuda.is_available():
            try:
                gpus = list(cuda.gpus)
                CoreLogInfo(f"Cuda is available. {len(gpus)} devices found:")
                for i, gpu in enumerate(gpus):
                    CoreLogInfo(f"  [{i}] {gpu.name}")

            except CudaSupportError as e:
                CoreLogError(f"  Cuda Support error: {str(e)}")
        else:
            CoreLogError(f"Cuda is not available on this machine")
            CoreLogError("Exiting...")
            sys.exit(1)

    def aces_tonemap_numpy(color: np.ndarray) -> np.ndarray:
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        tonemapped = (color * (a * color + b)) / (color * (c * color + d) + e)
        return np.clip(tonemapped, 0.0, 1.0)

    def PostProcessFilm(film : Film) -> Film:
        result = Film(film.width, film.height, film.num_samples)
        result.data = aces_tonemap_numpy(film.data)
        return result


    def IntegrateFilmsCPU(films : list[Film]) -> Film:
        if not films:
            CoreLogError("Film List is empty")
            raise ValueError("Film List is empty")

        width = films[0].width
        height = films[0].height

        # Check all films have the same dimensions
        for film in films:
            if film.width != width or film.height != height:
                raise ValueError("All films must have the same dimensions.")

        total_samples = sum(f.num_samples for f in films)
        if total_samples == 0:
            raise ValueError("Total number of samples is zero.")

        # Create output film
        combined_film = Film(width, height, total_samples)

        # Accumulate weighted sum
        for film in films:
            if film.num_samples > 0:
                weight = film.num_samples / total_samples
                combined_film.data += film.data * weight

        return combined_film



    def Render(self, scene : Scene, film : Film, num_samples : int = 128, num_bounces : int = 16):
        # Allocate output array on host and device
        # output_host = np.zeros((film.height, film.width, 3), dtype=np.uint8)
        # output_host = np.zeros((film.height, film.width, 3), dtype=np.uint8)

        output_device = cuda.to_device(film.data)

        # Set up grid and block dimensions
        block_size = (16, 16)
        grid_size = ((film.width + block_size[0] - 1) // block_size[0],
                    (film.height + block_size[1] - 1) // block_size[1])

        # Random Number Generation
        num_threads = block_size[0] * grid_size[0] * block_size[1] * grid_size[1]
        seed = int(datetime.now().timestamp())
        rng_states = create_xoroshiro128p_states(num_threads, seed)

        # Launch kernel
        camera = scene.camera
        cuda_cam_pos = cuda.to_device(camera.position)
        cuda_view_proj = cuda.to_device(camera.view_projection)
        cuda_inv_view_proj = cuda.to_device(np.linalg.inv(camera.view_projection))


        numSpheres = len(scene.spheres)
        sphere_positions = np.array([sphere.position for sphere in scene.spheres])
        sphere_radius = np.array([sphere.radius for sphere in scene.spheres])
        sphere_materials = np.array([sphere.material_id for sphere in scene.spheres])
        cuda_sphere_positions = cuda.to_device(sphere_positions)
        cuda_sphere_radius = cuda.to_device(sphere_radius)
        cuda_sphere_material_indices = cuda.to_device(sphere_materials)

        num_planes = len(scene.planes)
        plane_normal = np.array([plane.normal for plane in scene.planes])
        plane_distance = np.array([plane.distance for plane in scene.planes])
        plane_material_indices = np.array([plane.material_id for plane in scene.planes])

        cuda_plane_normal = cuda.to_device(plane_normal)
        cuda_plane_distance = cuda.to_device(plane_distance)
        cuda_plane_material_indices = cuda.to_device(plane_material_indices)

        num_materials = len(scene.materials)
        materials = np.zeros((num_materials, 3, 3), dtype=np.float32)
        for i in range(num_materials):
            for j in range(3):
                materials[i, 0, j] = scene.materials[i].albedo[j]
                materials[i, 1, j] = scene.materials[i].emission[j]
            materials[i, 2, 0] = scene.materials[i].roughness
            materials[i, 2, 1] = scene.materials[i].metallic
            

        cuda_materials = cuda.to_device(materials)
        cuda_hdr, hdr_width, hdr_height = load_hdr_to_device("TestData/sky.hdr")


        # Time the render
        start = cuda.event()
        end = cuda.event()


        start.record()

        Trace[grid_size, block_size](output_device, film.width, film.height, 
            cuda_sphere_positions, cuda_sphere_radius, cuda_sphere_material_indices, numSpheres,
            cuda_plane_normal, cuda_plane_distance, cuda_plane_material_indices, num_planes,
            cuda_materials,
            cuda_cam_pos, cuda_view_proj, cuda_inv_view_proj, num_samples, num_bounces,
            cuda_hdr, hdr_width, hdr_height,
            rng_states)

        end.record()
        end.synchronize()
        elapsed_time = cuda.event_elapsed_time(start, end)


        # Copy result back to host and save as image
        film.data = output_device.copy_to_host()
        LogInfo(f"Render finished with total time of {elapsed_time * 0.001 : 0.4f} seconds")
        # image = Image.fromarray(output_host)
        # image.save('gradient.png')