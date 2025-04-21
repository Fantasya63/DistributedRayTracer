import numpy as np
import math

from Renderer import Film


class Camera:
    # TODO: Add a change aspect method
    def __init__(self, pos, rot, fov, near, far, film : Film):
        self.position = np.array(pos, dtype=np.float32)
        self.rotation = np.array(rot, dtype=np.float32)
        self.fov = math.radians(fov)
        self.aspect = film.width / film.height
        self.near = near
        self.far = far
        self.viewProjection = self._compute_view_proj()
        self.film : Film = film
   

    def _create_view_matrix(self):
        # Convert Euler angles (degrees) to radians
        rx, ry, rz = np.radians(self.rotation)
        
        # Rotation matrices for X, Y, Z (Blender's XYZ Euler order)
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        
        rot_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], dtype=np.float32)
        
        rot_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=np.float32)
        
        rot_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combined rotation: R = Rz * Ry * Rx
        rot = rot_z @ rot_y @ rot_x
        
        # In Blender, +Y is forward, +Z is up, so forward is [0, 1, 0]
        forward = rot @ np.array([0, 1, 0], dtype=np.float32)
        up = rot @ np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(up, forward)  # Ensure orthogonality
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = forward
        view[2, :3] = up
        view[0, 3] = -np.dot(right, self.position)
        view[1, 3] = -np.dot(forward, self.position)
        view[2, 3] = -np.dot(up, self.position)
        return view


    def _create_proj_matrix(self):
        f = 1.0 / math.tan(self.fov / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        return proj


    def _compute_view_proj(self):
        return self._create_proj_matrix() @ self._create_view_matrix()