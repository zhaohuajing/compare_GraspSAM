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

        # #  RGB-D images in GraspSAM are currently 1024×1024, but the camera info is 640×480: will convert in eval.py

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
    grasp_height_offset=0.05,
):
    """
    Convert a planar grasp rectangle to a 6D grasp pose assuming:

    - camera is approximately top-down
    - gripper approaches along camera +Z axis
    - rectangle angle defines yaw in image plane

    Parameters
    ----------
    grasp:
        Grasp object (from grasp_utils)
        - grasp.center = (u, v)
        - grasp.angle  = radians
        - grasp.width  = pixels (full opening)
    depth_img:
        HxW depth image (meters)
    intrinsics:
        CameraIntrinsics
    grasp_height_offset:
        Optional offset added to depth (meters)

    Returns
    -------
    position : np.ndarray (3,)
        [x, y, z] in camera frame
    quaternion : np.ndarray (4,)
        [qx, qy, qz, qw]
    width_m : float
        Gripper opening in meters
    """

    u, v = grasp.center
    # v, u = grasp.center # test swaping x and y - x using center[1], y using center[0]
    u = int(round(u))
    v = int(round(v))

    # z = depth_img[v, u]
    z = depth_img[u,v] # this is correct!
    print(f"[grasp_pose_convert_utils]: grasp.center = {grasp.center}, z = {z}")
    if z < 0 or np.isnan(z):
        raise ValueError("Invalid depth at grasp center")

    # z = z + grasp_height_offset
    z = z - 2*grasp_height_offset
    # z = max(float(z) - 2 * float(grasp_height_offset), 1e-4) # consider adding a lower bound later

    # Pixel -> camera coordinates
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy

    # y = (u - intrinsics.cx) * z / intrinsics.fy
    # x = (v - intrinsics.cy) * z / intrinsics.fx

    x = (v - intrinsics.cx) * z / intrinsics.fx
    y = (u - intrinsics.cy) * z / intrinsics.fy

    position = np.array([x, y, z])
    # position = np.array([y, x, z])

    # Orientation:
    #   - yaw from grasp angle
    #   - flip gripper to point downward
    yaw = grasp.angle
    rot = R.from_euler("z", yaw) # * R.from_euler("x", np.pi)
    quaternion = rot.as_quat()  # (x, y, z, w)

    # Metric gripper width
    width_m = (grasp.width * z) / intrinsics.fx

    return position, quaternion, width_m
