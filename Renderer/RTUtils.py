from numba import cuda
import math

@cuda.jit(device=True)
def normalize_vec3(vec, result):
    """Normalize a 3D vector, storing result in provided array."""
    length = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    if length > 0:
        for i in range(3):
            result[i] = vec[i] / length
    else:
        for i in range(3):
            result[i] = vec[i]


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
def Float3ToRGB(output, _input):
    for i in range(3):
        temp = max(_input[i], 0.0)

        output[i] = int(temp * 255)