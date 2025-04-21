from Scene.RTPrimitives import Sphere, Cube, Plane

from Renderer.Camera import Camera

class Scene:
    def __init__(self):
        self.camera : Camera = None
        self.spheres = []
        self.cubes = []
        self.planes = []

    def set_camera(self, cam : Camera):
        self.camera = cam

    def add_sphere(self, sphere : Sphere):
        self.spheres.append(sphere)

    def add_cube(self, cube : Cube):
        self.cubes.append(cube)

    def add_plane(self, plane : Plane):
        self.planes.append(plane)
