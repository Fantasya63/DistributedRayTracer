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
    result[1] = (a[2] * b[0]) - (a[1] * b[2])
    result[2] = (a[0] * b[1]) - (a[2] * b[0])


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
    vec3_cross(normal, helper, temp)
    vec3_normalize(temp, tangent)

    # Binormal = norm(Normal x Tangent)
    vec3_cross(normal, tangent, temp)
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
    # for i in range(3):
    #     incident[i] *= -1.0
    vec3_normalize(incident, incident)
    vec3_normalize(normal, normal)
    dot_IN = vec3_dot(incident, normal)
    for i in range(3):
        result[i] = incident[i] - 2.0 * dot_IN * normal[i]



@cuda.jit(device=True)
def sdot_fast(x, y, f=1.0):
    d = vec3_dot(x, y) * f
    d = 0.5 * (d + abs(d))  # remove negative values (clamp at 0)
    return min(d, 1.0)      # clamp upper at 1