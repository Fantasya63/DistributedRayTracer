from pyglm import glm;

class Material():
    def __init__(self, albedo : glm.vec3, roughness : float, metallic : float = 0.0, emission : glm.vec3 = glm.vec3(0.0)):
        self.albedo = albedo
        self.emission = emission
        self.roughness = roughness
        self.metallic = metallic