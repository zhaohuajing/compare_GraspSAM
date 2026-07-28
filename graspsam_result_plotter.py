#!/usr/bin/env python3
"""
graspsam_result_plotter.py

Visualize GraspSAM 6D poses on a point cloud reconstructed from the same
Jacquard-like RGB-D sample used by eval_functional_jac_loader.py.

Inputs expected:
  <sample_id>_RGB.png
  <sample_id>_perfect_depth.tiff
  sample_0_grasps.json  (or any JSON containing a list of grasps)

The JSON grasp entries should contain:
  pos:  [x, y, z]
  quat: [qx, qy, qz, qw]
  width_m: optional
  score: optional

By default, this visualizes everything in the camera optical frame, because:
  - the point cloud reconstructed from image intrinsics is in optical convention
  - eval_functional_jac_loader.py's rectangle_to_pose_topdown() output is also
    in the same camera frame before the server applies optical->ROS conversion.

If you want to visualize the same ROS-camera-link convention used by the server,
pass --frame_mode ros_cam. This applies the standard optical->camera_link
axis conversion to BOTH the point cloud and grasp poses.
"""

"""
Example:

python3 graspsam_result_plotter.py   
    --rgb ./rgbd2jacquard/Kinova_Gen3_real_YCB/sample2_mnet_scene/0_from_rgbd_RGB.png   
    --depth ./rgbd2jacquard/Kinova_Gen3_real_YCB/sample2_mnet_scene/0_from_rgbd_perfect_depth.tiff   
    --grasps ./grasp_outputs/run_20260628_020526_realYCB_mnet_6DPose/sample_0_grasps.json   
    --top_k 5   --stride 2   --frame_mode optical


python3 graspsam_result_plotter.py   --rgb ./rgbd2jacquard/realtime/0_from_rgbd_RGB.png   
    --depth ./rgbd2jacquard/realtime/0_from_rgbd_perfect_depth.tiff   
    --grasps ./grasp_outputs/run_20260/sample_0_grasps.json   --top_k 10   --stride 2   --frame_mode optical



"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit(
        "Open3D is required for this plotter. Install with:\n"
        "  pip install open3d\n"
        f"Original import error: {e}"
    )


def load_depth_tiff(path: str) -> np.ndarray:
    depth = np.array(Image.open(path)).astype(np.float32)
    return depth


def load_rgb(path: str) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    return rgb


def make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_intrinsics(args, width: int, height: int) -> np.ndarray:
    """
    Scale the original CameraInfo intrinsics to the current image resolution.

    This matches eval_functional_jac_loader.py's current style:
      sx = out_size / intr_w
      sy = out_size / intr_h
      fx = fx0 * sx, fy = fy0 * sy, cx = cx0 * sx, cy = cy0 * sy

    For the current Jacquard-like images, width=height=1024 normally.
    """
    sx = float(width) / float(args.intr_w)
    sy = float(height) / float(args.intr_h)
    return make_K(args.fx * sx, args.fy * sy, args.cx * sx, args.cy * sy)


def depth_rgb_to_point_cloud(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray,
                             stride: int = 2, max_depth: float = 3.0,
                             min_depth: float = 1e-6):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[0:H:stride, 0:W:stride]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)

    x = xs[valid].astype(np.float64)
    y = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    pts = np.stack([X, Y, Z], axis=1)

    colors = rgb[0:H:stride, 0:W:stride, :][valid].astype(np.float64) / 255.0
    return pts, colors


def optical_to_ros_matrix() -> np.ndarray:
    """
    Standard camera optical -> ROS camera_link rotation.

    Optical:     x right, y down, z forward
    ROS camera:  x forward, y left, z up

    p_ros = R * p_opt = [z, -x, -y]
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    return T


def quat_pos_to_T(pos, quat) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def T_to_pose(T):
    return T[:3, 3].copy(), R.from_matrix(T[:3, :3]).as_quat()


def load_grasps_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "grasps" in data:
            grasps = data["grasps"]
        else:
            # Allow server-style dictionaries later if needed.
            grasps = data.get("results", [])
    elif isinstance(data, list):
        grasps = data
    else:
        raise ValueError(f"Unsupported grasp JSON type: {type(data)}")

    valid = []
    for i, g in enumerate(grasps):
        pos = g.get("pos", g.get("pos_cam", None))
        quat = g.get("quat", g.get("quat_cam", None))
        if pos is None or quat is None:
            print(f"[WARN] grasp {i} skipped: no pos/quat")
            continue
        if len(pos) != 3 or len(quat) != 4:
            print(f"[WARN] grasp {i} skipped: malformed pos/quat")
            continue
        if not np.all(np.isfinite(np.asarray(pos, dtype=float))) or not np.all(np.isfinite(np.asarray(quat, dtype=float))):
            print(f"[WARN] grasp {i} skipped: non-finite pos/quat")
            continue
        valid.append(g)
    return valid


def make_gripper_lines(T: np.ndarray, width_m: float, finger_len: float = 0.055,
                       palm_depth: float = 0.025) -> "o3d.geometry.LineSet":
    """
    Create a small gripper-like wireframe attached to the grasp pose.

    Local convention used only for visualization:
      - local X: jaw/opening direction
      - local Z: approach direction / forward axis of the pose frame
      - local Y: finger length direction

    If your end-effector convention differs, the coordinate frame axes are the
    most reliable orientation indicator; this wireframe is just visual context.
    """
    w = max(float(width_m), 0.02)
    half_w = w / 2.0

    # points in local grasp frame
    pts_local = np.array([
        [-half_w, 0.0, 0.0],
        [ half_w, 0.0, 0.0],
        [-half_w, finger_len, 0.0],
        [ half_w, finger_len, 0.0],
        [-half_w, 0.0, -palm_depth],
        [ half_w, 0.0, -palm_depth],
    ], dtype=np.float64)

    pts_h = np.c_[pts_local, np.ones((pts_local.shape[0], 1))]
    pts_world = (T @ pts_h.T).T[:, :3]

    lines = np.array([
        [0, 1],  # palm opening
        [0, 2],  # left finger
        [1, 3],  # right finger
        [0, 4],  # small depth cue
        [1, 5],
        [4, 5],
    ], dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lines), 1)))
    return line_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, help="Path to *_RGB.png")
    ap.add_argument("--depth", required=True, help="Path to *_perfect_depth.tiff")
    ap.add_argument("--grasps", required=True, help="Path to sample_0_grasps.json")
    ap.add_argument("--save_image", default="", help="Optional screenshot path")

    # Intrinsics as used by eval_functional_jac_loader.py
    ap.add_argument("--fx", type=float, default=554.3827128226441)
    ap.add_argument("--fy", type=float, default=554.3827128226441)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--intr_w", type=int, default=640)
    ap.add_argument("--intr_h", type=int, default=480)
    # ap.add_argument("--intr_w", type=int, default=480) # kinova gen3 camera depth
    # ap.add_argument("--intr_h", type=int, default=270) # kinova gen3 camera depth

    ap.add_argument("--frame_mode", choices=["optical", "ros_cam"], default="optical",
                    help="optical: plot raw eval poses/cloud. ros_cam: apply optical->ROS camera conversion to both.")
    ap.add_argument("--stride", type=int, default=2, help="Point cloud decimation stride")
    ap.add_argument("--max_depth", type=float, default=2.0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--frame_size", type=float, default=0.05)
    ap.add_argument("--finger_len", type=float, default=0.055)
    ap.add_argument("--crop", type=int, default=0, choices=[0, 1])
    ap.add_argument("--crop_center", type=float, nargs=3, default=[0.0, 0.0, 0.65])
    ap.add_argument("--crop_size", type=float, nargs=3, default=[0.8, 0.8, 0.8])
    args = ap.parse_args()

    rgb = load_rgb(args.rgb)
    depth = load_depth_tiff(args.depth)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: rgb={rgb.shape[:2]}, depth={depth.shape}")

    K = scale_intrinsics(args, width=depth.shape[1], height=depth.shape[0])
    print("[INFO] K used for point cloud:\n", K)

    pts, colors = depth_rgb_to_point_cloud(depth, rgb, K, stride=args.stride, max_depth=args.max_depth)

    T_frame = np.eye(4)
    if args.frame_mode == "ros_cam":
        T_frame = optical_to_ros_matrix()
        pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
        pts = (T_frame @ pts_h.T).T[:, :3]
        print("[INFO] Applied optical->ROS camera conversion to point cloud and grasps.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if args.crop == 1:
        center = np.asarray(args.crop_center, dtype=np.float64)
        size = np.asarray(args.crop_size, dtype=np.float64)
        bbox = o3d.geometry.AxisAlignedBoundingBox(center - size / 2.0, center + size / 2.0)
        pcd = pcd.crop(bbox)
        print(f"[INFO] Cropped point cloud to {len(pcd.points)} points")

    geometries = [pcd]

    # Add camera/world origin frame
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frame_size * 1.5)
    geometries.append(origin_frame)

    grasps = load_grasps_json(args.grasps)
    print(f"[INFO] Loaded valid grasps: {len(grasps)}")

    # Sort if score exists; otherwise keep JSON order.
    def score_of(g):
        return float(g.get("score", 0.0))
    grasps = sorted(grasps, key=score_of, reverse=True)[:args.top_k]

    for i, g in enumerate(grasps):
        T = quat_pos_to_T(g.get("pos", g.get("pos_cam")), g.get("quat", g.get("quat_cam")))
        T = T_frame @ T

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frame_size)
        frame.transform(T)
        geometries.append(frame)

        width_m = float(g.get("width_m", 0.04))
        geometries.append(make_gripper_lines(T, width_m=width_m, finger_len=args.finger_len))

        pos, quat = T_to_pose(T)
        print(f"[grasp {i}] pos={pos.round(4).tolist()} quat={quat.round(4).tolist()} width_m={width_m:.4f}")

    if args.save_image:
        # Interactive window + one screenshot after rendering.
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="GraspSAM 6D pose visualization", width=1280, height=900, visible=True)
        for geom in geometries:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save_image, do_render=True)
        print(f"[INFO] Saved screenshot: {args.save_image}")
        vis.run()
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries(geometries, window_name="GraspSAM 6D pose visualization")


if __name__ == "__main__":
    main()
