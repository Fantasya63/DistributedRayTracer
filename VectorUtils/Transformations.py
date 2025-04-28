from pyglm import glm

def EulerToMatrix(euler : glm.vec3) -> glm.vec4:
    rot_x = glm.rotate(glm.radians(euler.x), glm.vec3(1.0, 0.0, 0.0))
    rot_y = glm.rotate(glm.radians(euler.y), glm.vec3(0.0, 1.0, 0.0))
    rot_z = glm.rotate(glm.radians(euler.z), glm.vec3(0.0, 0.0, 1.0))

    return rot_z @ rot_x @ rot_y