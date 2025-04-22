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
   

    def _compute_view_proj(self):
        
        # Rotation Matrix
        pitch, yaw, roll = self.rotation
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_r, sin_r = np.cos(roll), np.sin(roll)

        rot_matrix_3x3 = np.array([
            [
                cos_y * cos_r + sin_y * sin_p * sin_r,
                sin_y * cos_p,
                cos_y * sin_r - sin_y * sin_p * cos_r
            ],
            [
                -sin_y * cos_r + cos_y * sin_p * sin_r,
                cos_y * cos_p,
                -sin_y * sin_r - cos_y * sin_p * cos_r
            ],
            [
                -cos_p * sin_r,
                sin_p,
                cos_p * cos_r
            ]
        ], dtype=np.float32)
        rot_matrix = np.eye(4, dtype=np.float32)
        rot_matrix[:3, :3] = rot_matrix_3x3

        # Translation Matrix
        trans_matrix = np.eye(4, dtype=np.float32)
        trans_matrix[:3, 3] = -self.position

        # View Matrix
        view_matrix = np.dot(rot_matrix, trans_matrix)

        # Create projection matrix
        tan_half_fov = np.tan(self.fov / 2)
        proj_matrix = np.zeros((4, 4), dtype=np.float32)
        proj_matrix[0, 0] = 1.0 / (self.aspect * tan_half_fov)
        proj_matrix[1, 1] = 1.0 / tan_half_fov
        proj_matrix[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj_matrix[2, 3] = -(2 * self.far * self.near) / (self.far - self.near)
        proj_matrix[3, 2] = -1.0

        # Combine view and projection matrices
        view_proj_matrix = np.dot(proj_matrix, view_matrix)
        
        return view_proj_matrix