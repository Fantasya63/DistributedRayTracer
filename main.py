
from Renderer.Renderer import Renderer
from Renderer.Film import Film
from Scene.Scene import Scene
from Scene.RTPrimitives import Sphere

def main():    
    # Image dimensions
    film : Film = Film(width=1920, height=1080)

        
    scene : Scene = Scene()
    mySphere : Sphere = Sphere

    renderer = Renderer()
    renderer.Render(scene, film)



if __name__ == "__main__":
    main()