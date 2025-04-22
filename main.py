
from Renderer.Renderer import Renderer
from Renderer.Film import Film
from Renderer.Camera import Camera
from Scene.Scene import Scene
from Scene.RTPrimitives import Sphere

def main():    
    # Image dimensions
    film : Film = Film(width=1920, height=1080)

        
    scene : Scene = Scene()
    mySphere : Sphere = Sphere(position=[0.0, 0.0, 0.0], radius=1.0)
    camera : Camera = Camera(pos=[0.0, 0.0, 5.0], rot=[0.0, 0.0, 0.0], fov=90.0, near=0.1, far=10.0, film=film)

    scene.add_sphere(mySphere)
    scene.set_camera(camera)

    renderer = Renderer()
    renderer.Render(scene, film)



if __name__ == "__main__":
    main()