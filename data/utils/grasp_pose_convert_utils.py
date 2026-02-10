# data/utils/grasp_pose_convert_utils.py

import numpy as np
from scipy.spatial.transform import Rotation as R


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

        # self.fx *= scale_x
        # self.fy *= scale_y
        # self.cx *= scale_x
        # self.cy *= scale_y



def rectangle_to_pose_topdown(
    grasp,
    depth_img,
    intrinsics: CameraIntrinsics,
    grasp_height_offset=0.0,
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
    u = int(round(u))
    v = int(round(v))

    z = depth_img[v, u]
    if z <= 0 or np.isnan(z):
        raise ValueError("Invalid depth at grasp center")

    z = z + grasp_height_offset

    # Pixel -> camera coordinates
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy

    position = np.array([x, y, z])

    # Orientation:
    #   - yaw from grasp angle
    #   - flip gripper to point downward
    yaw = grasp.angle
    rot = R.from_euler("z", yaw) * R.from_euler("x", np.pi)
    quaternion = rot.as_quat()  # (x, y, z, w)

    # Metric gripper width
    width_m = (grasp.width * z) / intrinsics.fx

    return position, quaternion, width_m
