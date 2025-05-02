# RTUtils.py

from numba import cuda
import math
import numpy as np
from numba.cuda.random import (
    xoroshiro128p_uniform_float32
)


@cuda.jit(device=True)
def vec3_normalize(vec, result):
    """Normalize a 3D vector, storing result in provided array."""
    length = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    if length > 0:
        for i in range(3):
            result[i] = vec[i] / length
    else:
        for i in range(3):
            result[i] = vec[i]


@cuda.jit(device=True)
def vec3_dot(a, b):
    result = 0.0
    for i in range(3):
        result += a[i] * b[i]

    return result



@cuda.jit(device=True)
def vec3_cross(a, b, result):
    """ calculates the cross product of A x B and returns the result in result arr """
    result[0] = (a[1] * b[2]) - (a[2] * b[1])
    result[1] = (a[2] * b[0]) - (a[0] * b[2])
    result[2] = (a[0] * b[1]) - (a[1] * b[0])


@cuda.jit(device=True)
def vec3_add(a, b, result):
    """ Subtracts b from a and returns the result in result """
    for i in range(3):
        result[i] = a[i] - b[i]


@cuda.jit(device=True)
def vec3_sub(a, b, result):
    """ Subtracts b from a and returns the result in result """
    for i in range(3):
        result[i] = a[i] - b[i]


@cuda.jit(device=True)
def vec3_mul(a, b, result):
    for i in range(3):
        result[i] = a[i] * b[i]


@cuda.jit(device=True)
def vec3_div(a, b, result):
    for i in range(3):
        result[i] = a[i] / b[i]


@cuda.jit(device=True)
def mat4x4_vec4_multiply(mat, vec, result):
    """Multiply 4x4 matrix by 4D vector, storing result in provided array."""
    for i in range(4):
        sum_val = 0.0
        for j in range(4):
            sum_val += mat[i, j] * vec[j]
        result[i] = sum_val


@cuda.jit(device=True)
def mat3x3_vec4_multiply(mat, vec, result):
    """Multiply 4x4 matrix by 4D vector, storing result in provided array."""
    for i in range(3):
        sum_val = 0.0
        for j in range(3):
            sum_val += mat[i, j] * vec[j]
        result[i] = sum_val


@cuda.jit(device=True)
def Float3ToRGB(output, _input):
    for i in range(3):
        temp = max(_input[i], 0.0)
        temp = min(temp, 1.0)

        output[i] = int(temp * 255)

@cuda.jit(device=True)
def reinhard_tonemap(input, output):
    """
    Applies Reinhard tonemapping to a vec3 color.
    
    Parameters:
        input_color: float32[3] - Input HDR color
        output_color: float32[3] - Output LDR color
    """
    output[0] = input[0] / (1.0 + input[0])
    output[1] = input[1] / (1.0 + input[1])
    output[2] = input[2] / (1.0 + input[2])


@cuda.jit(device=True)
def filmic_tonemap(input_color, output_color):
    """
    Approximates Blender's Filmic tonemapping using Hable's curve.
    
    Parameters:
        input_color: float32[3] - HDR input color
        output_color: float32[3] - LDR output color
    """
    for i in range(3):
        x = input_color[i]
        
        # Hable tonemapping parameters
        A = 0.22
        B = 0.30
        C = 0.10
        D = 0.20
        E = 0.01
        F = 0.30
        
        # Hable curve
        tonemapped = ((x*(A*x+C*B)+D*E) / (x*(A*x+B)+D*F)) - E/F

        # Normalize to [0,1]
        output_color[i] = max(0.0, min(tonemapped, 1.0))



@cuda.jit(device=True)
def aces_tonemap(input, output):
    # RRT and ODT fit approximation (ACES filmic curve)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    for i in range(3):  # process R, G, B channels
        x = input[i]
        tonemapped = (x * (a * x + b)) / (x * (c * x + d) + e)
        output[i] = min(max(tonemapped, 0.0), 1.0)


@cuda.jit(device=True)
def get_tangent_space_matrix(normal, output):
    helper = cuda.local.array(3, dtype=np.float32)
    
    if math.fabs(normal[0]) > 0.99:
        helper[0] = 0.0
        helper[1] = 0.0
        helper[2] = 1.0
    else:
        helper[0] = 1.0
        helper[1] = 0.0
        helper[2] = 0.0


    # Generate vectors
    tangent = cuda.local.array(3, dtype=np.float32)
    binormal = cuda.local.array(3, dtype=np.float32)
    temp = cuda.local.array(3, dtype=np.float32)
   
    # Tangent = norm(Normal x helper)
    vec3_cross(helper, normal, temp)
    vec3_normalize(temp, tangent)

    # Binormal = norm(Normal x Tangent)
    vec3_cross(tangent, normal, temp)
    vec3_normalize(temp, binormal)
    
    for i in range(3):
        output[0, i] = tangent[i]
        output[1, i] = binormal[i]
        output[2, i] = normal[i]


@cuda.jit(device=True)
def sample_hemisphere(normal, alpha, rng_states, thread_id, output):
    rand_cos = xoroshiro128p_uniform_float32(rng_states, thread_id)
    cos_theta = math.pow(rand_cos, 1.0 / (alpha + 1.0))
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    phi = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, thread_id)
    tangent_space_dir = cuda.local.array(3, dtype=np.float32)

    tangent_space_dir[0] = math.cos(phi) * sin_theta
    tangent_space_dir[1] = math.sin(phi) * sin_theta
    tangent_space_dir[2] = cos_theta

    tangent_matrix = cuda.local.array((3, 3), dtype=np.float32)
    get_tangent_space_matrix(normal, tangent_matrix)
    mat3x3_vec4_multiply(tangent_matrix, tangent_space_dir, output)



@cuda.jit(device=True)
def smoothness_to_phong_alpha(s):
    return pow(1000.0, s * s)



@cuda.jit(device=True)
def vec3_reflect(incident, normal, result):
    """
    Reflect vector Incident around normal Normal, result = I - 2 * dot(I, N) * N
    """
    dot_IN = vec3_dot(incident, normal)
    for i in range(3):
        result[i] = incident[i] - ((2.0 * dot_IN) * normal[i])



@cuda.jit(device=True)
def sdot_fast(x, y, f=1.0):
    d = vec3_dot(x, y) * f
    d = 0.5 * (d + abs(d))  # remove negative values (clamp at 0)
    return min(d, 1.0)      # clamp upper at 1


@cuda.jit(device=True)
def sample_hdr(ray_dir, hdr_image, width, height, output):
    x, y, z = ray_dir[0], ray_dir[1], ray_dir[2]
    
    # len_inv = 1.0 / math.sqrt(x*x, y*y, z*z)

    # Convert to UV coordinates using equirectangular projection
    u = (math.atan2(x, -z) / (2.0 * math.pi)) + 0.5
    v = math.acos(y) / math.pi

    # Convert UV to pixel coordinates
    px = int(u * width) % width
    py = int(v * height) % height

    output[0] = hdr_image[py, px, 0]
    output[1] = hdr_image[py, px, 1]
    output[2] = hdr_image[py, px, 2]
