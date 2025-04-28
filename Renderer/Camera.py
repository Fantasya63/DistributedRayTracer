import numpy as np
import math

class Camera:
    def __init__(self, pos, rot, fov, near, far, film):
        self.position = np.array(pos, dtype=np.float32)
        self.rotation = np.radians(rot).astype(np.float32)  # (yaw, pitch, roll) in degrees
        self.fov = math.radians(fov)
        self.aspect = film.width / film.height
        self.near = near
        self.far = far
        self.film = film

        self.view_matrix = self._compute_view_matrix()
        self.projection_matrix = self._compute_projection_matrix()
        self.view_projection = self.projection_matrix @ self.view_matrix

    def set_rotation(self, rot):
        self.rotation = np.radians(rot).astype(np.float32)
        # self.rotation = np.array(rot, dtype=np.float32)
        self.view_matrix = self._compute_view_matrix()
        self.projection_matrix = self._compute_projection_matrix()
        self.view_projection = self.projection_matrix @ self.view_matrix

    def set_FOV(self, fov):
        self.fov = math.radians(fov)
        self.projection_matrix = self._compute_projection_matrix()
        self.view_projection = self.projection_matrix @ self.view_matrix


    def _rotation_matrix_yaw_pitch_roll(self, pitch, yaw, roll):
        # Yaw (Y-axis)
        cy, sy = math.cos(yaw), math.sin(yaw)
        # Pitch (X-axis)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Roll (Z-axis)
        cr, sr = math.cos(roll), math.sin(roll)

        # Combine rotations (Z * X * Y)
        rot = np.array([
            [cy * cr + sy * sp * sr, sr * cp, -sy * cr + cy * sp * sr],
            [-cy * sr + sy * sp * cr, cr * cp, sr * sy + cy * sp * cr],
            [sy * cp, -sp, cy * cp]
        ], dtype=np.float32)

        return rot
    

    def get_rot_matrix(self):
        return self._rotation_matrix_yaw_pitch_roll(*self.rotation)


    def _compute_view_matrix(self):
        # Calculate rotation matrix from yaw, pitch, roll
        rot = self._rotation_matrix_yaw_pitch_roll(*self.rotation)

        # Basis vectors
        right = rot[:, 0]
        up = rot[:, 1]
        forward = -rot[:, 2]  # -Z is forward in RH

        # Build view matrix (right-handed)
        mat = np.identity(4, dtype=np.float32)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = forward
        mat[0, 3] = -np.dot(right, self.position)
        mat[1, 3] = -np.dot(up, self.position)
        mat[2, 3] = -np.dot(forward, self.position)

        return mat

    def _compute_projection_matrix(self):
        f = 1.0 / math.tan(self.fov / 2.0)
        z_range = self.near - self.far

        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / self.aspect
        mat[1, 1] = f
        mat[2, 2] = (self.far + self.near) / z_range
        mat[2, 3] = -1.0
        mat[3, 2] = (2 * self.near * self.far) / z_range

        return mat
