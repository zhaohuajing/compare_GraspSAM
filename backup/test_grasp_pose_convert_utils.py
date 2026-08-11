import numpy as np
from data.utils.grasp_pose_convert_utils import (
    CameraIntrinsics,
    rectangle_to_pose_topdown
)

# -----------------------------
# Fake / sample grasp
# -----------------------------
class DummyGrasp:
    def __init__(self, center, angle, width):
        self.center = center      # (u, v) pixels
        self.angle = angle        # radians
        self.width = width        # pixels (full opening)

# Example grasp near image center
grasp = DummyGrasp(
    center=(512, 512),
    angle=np.deg2rad(-15),
    width=32,   # px
)

# -----------------------------
# Fake depth image
# -----------------------------
depth_img = np.ones((1024, 1024), dtype=np.float32) * 0.6  # 60 cm flat plane

# -----------------------------
# Camera intrinsics (scaled!)
# -----------------------------
fx = 554.3827128226441 * (1024 / 640)
fy = 554.3827128226441 * (1024 / 480)
cx = 320.0 * (1024 / 640)
cy = 240.0 * (1024 / 480)

intrinsics = CameraIntrinsics(fx, fy, cx, cy)

# -----------------------------
# Convert
# -----------------------------
pos, quat, width_m = rectangle_to_pose_topdown(
    grasp,
    depth_img,
    intrinsics,
)

print("Position (camera frame):", pos)
print("Quaternion (x,y,z,w):", quat)
print("Gripper width (m):", width_m)
