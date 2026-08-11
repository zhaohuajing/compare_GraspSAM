# data/utils/grasp_pose_convert_utils.py

import numpy as np
from scipy.spatial.transform import Rotation as R


# IMPORTANT:
# In this GraspSAM branch, grasp.center behaves effectively as (u_like, v_like)
# for the detected rectangle, but the correct depth lookup for our converted
# Jacquard-like images is depth_img[u, v], not depth_img[v, u].
# Do not change this unless the upstream Grasp.center convention changes.


class CameraIntrinsics:
    """
    Minimal pinhole camera model
    """
    def __init__(self, fx=554.3827128226441, fy=554.3827128226441, cx=320.0, cy=240.0):


        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # #  RGB-D images in GraspSAM are currently 1024×1024, but the camera info (gazebo sim) is 640×480: will convert in eval.py

        # scale_x = 1024 / 640
        # scale_y = 1024 / 480

        # scale_x = 1024 / 480
        # scale_y = 1024 / 270

        # self.fx *= scale_x
        # self.fy *= scale_y
        # self.cx *= scale_x
        # self.cy *= scale_y



def rectangle_to_pose_topdown(
    grasp,
    depth_img,
    intrinsics: CameraIntrinsics,
    grasp_height_offset=0.15,
):
    """
    Convert a planar grasp rectangle to a 6D grasp pose assuming:

    - camera is approximately top-down
    - gripper approaches along camera +Z axis
    - rectangle angle defines yaw in image plane

    IMPORTANT:
    The detected grasp center corresponds to a pixel on the object surface.
    Use that surface depth for x/y back-projection. Apply the height offset
    only to the returned z target; otherwise x/y will be scaled toward the
    camera optical center.
    """

    u, v = grasp.center
    u = int(round(u))
    v = int(round(v))

    H, W = depth_img.shape[:2]
    if u < 0 or u >= H or v < 0 or v >= W:
        raise ValueError(
            f"Grasp center out of depth bounds: center={grasp.center}, "
            f"rounded=(u={u}, v={v}), depth_shape=({H}, {W})"
        )

    # In this branch, u is row-like and v is column-like.
    z_surface = float(depth_img[u, v])
    print(f"[grasp_pose_convert_utils]: grasp.center={grasp.center}, "
          f"u(row)={u}, v(col)={v}, z_surface={z_surface}")

    if z_surface <= 0 or np.isnan(z_surface):
        raise ValueError("Invalid depth at grasp center")

    # Back-project the object surface point.
    x = (v - intrinsics.cx) * z_surface / intrinsics.fx
    y = (u - intrinsics.cy) * z_surface / intrinsics.fy

    # EE/TCP target depth. Do not use this reduced depth for x/y.
    z_target = z_surface - float(grasp_height_offset)
    z_target = max(z_target, 1e-4)

    position = np.array([x, y, z_target], dtype=np.float64)

    yaw = grasp.angle
    rot = R.from_euler("z", yaw)
    quaternion = rot.as_quat()  # (x, y, z, w)

    # Metric gripper width is measured at the observed object surface.
    width_m = (float(grasp.width) * z_surface) / intrinsics.fx

    print(f"[grasp_pose_convert_utils]: position={position}, width_m={width_m}")

    return position, quaternion, width_m
