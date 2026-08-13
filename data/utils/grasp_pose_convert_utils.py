# data/utils/grasp_pose_convert_utils.py

import numpy as np
from scipy.spatial.transform import Rotation as R


# IMPORTANT:
# In this GraspSAM branch, grasp.center behaves effectively as (row-like, col-like)
# for the detected rectangle, so the correct depth lookup for the converted
# Jacquard-like images is depth_img[row, col].


class CameraIntrinsics:
    """Minimal pinhole camera model."""
    def __init__(self, fx=554.3827128226441, fy=554.3827128226441, cx=320.0, cy=240.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def _nearest_valid_depth(depth_img,
                         row: int,
                         col: int,
                         mask_img=None,
                         radii=(0, 5, 10, 20, 40, 80, 120),
                         min_depth=1e-6,
                         max_depth=3.0,
                         nearest_count=20):
    """Return a robust local depth estimate near (row, col).

    The Kinova registered depth sometimes has holes on reflective / thin / angled
    objects.  The 2D grasp can still be good, but depth_img[row, col] may be 0.
    This function first checks the center pixel, then searches expanding windows.
    If a mask is provided, it prefers valid depth pixels on the selected instance;
    otherwise it falls back to nearby valid depth pixels.
    """
    depth = np.asarray(depth_img, dtype=np.float32)
    h, w = depth.shape[:2]

    row = int(np.clip(row, 0, h - 1))
    col = int(np.clip(col, 0, w - 1))

    def valid_depth(z):
        return np.isfinite(z) and (float(z) > float(min_depth)) and (float(z) < float(max_depth))

    z0 = float(depth[row, col])
    if valid_depth(z0):
        return z0, f"center row={row}, col={col}"

    mask = None
    if mask_img is not None:
        mask = np.asarray(mask_img)
        if mask.shape[:2] != depth.shape[:2]:
            # Shape mismatch means the mask is unsafe to use for indexing.
            mask = None
        else:
            mask = mask > 0

    def search_window(radius, use_mask=True):
        r0 = max(0, row - radius)
        r1 = min(h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(w, col + radius + 1)

        win = depth[r0:r1, c0:c1]
        valid = np.isfinite(win) & (win > float(min_depth)) & (win < float(max_depth))
        if use_mask and mask is not None:
            valid &= mask[r0:r1, c0:c1]

        if not np.any(valid):
            return None

        rr, cc = np.nonzero(valid)
        rr_abs = rr + r0
        cc_abs = cc + c0
        d2 = (rr_abs - row) ** 2 + (cc_abs - col) ** 2
        order = np.argsort(d2)
        n = min(int(nearest_count), len(order))
        z_vals = win[rr[order[:n]], cc[order[:n]]].astype(np.float64)
        # Median of a few nearest pixels is more stable than a single noisy pixel.
        return float(np.median(z_vals))

    # Prefer same selected instance mask when available.
    if mask is not None:
        for radius in radii:
            z = search_window(int(radius), use_mask=True)
            if z is not None:
                return z, f"mask fallback radius={radius}, row={row}, col={col}, center_z={z0}"

    # Last resort: use nearby valid depth even without the instance mask.
    for radius in radii:
        z = search_window(int(radius), use_mask=False)
        if z is not None:
            return z, f"unmasked fallback radius={radius}, row={row}, col={col}, center_z={z0}"

    raise ValueError(
        "Invalid depth at grasp center and no nearby valid fallback depth found: "
        f"row={row}, col={col}, center_z={z0}, depth_shape={depth.shape}"
    )


def rectangle_to_pose_topdown(
    grasp,
    depth_img,
    intrinsics: CameraIntrinsics,
    grasp_height_offset=0.15,
    mask_img=None,
    depth_fallback_radii=(0, 5, 10, 20, 40, 80, 120),
    min_depth=1e-6,
    max_depth=3.0,
):
    """Convert a planar grasp rectangle to a 6D grasp pose.

    Convention used in this branch:
      - grasp.center[0] behaves as row-like index for depth lookup.
      - grasp.center[1] behaves as col-like index for depth lookup.
      - camera-frame x uses the column offset from cx.
      - camera-frame y uses the row offset from cy.

    x/y are back-projected using the object surface depth.  The optional
    grasp_height_offset is applied only to the returned z target, so the lateral
    position is not pulled toward the camera optical center.
    """
    row_like, col_like = grasp.center
    row = int(round(row_like))
    col = int(round(col_like))

    z_surface, depth_source = _nearest_valid_depth(
        depth_img,
        row=row,
        col=col,
        mask_img=mask_img,
        radii=depth_fallback_radii,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    print(
        "[grasp_pose_convert_utils]: "
        f"grasp.center={grasp.center}, row={row}, col={col}, "
        f"z_surface={z_surface}, depth_source={depth_source}"
    )

    # Pixel -> camera coordinates, using surface depth for lateral projection.
    x = (col - intrinsics.cx) * z_surface / intrinsics.fx
    y = (row - intrinsics.cy) * z_surface / intrinsics.fy

    z_target = z_surface - float(grasp_height_offset)
    z_target = max(float(z_target), 1e-4)

    position = np.array([x, y, z_target], dtype=np.float64)

    yaw = grasp.angle
    rot = R.from_euler("z", yaw)
    quaternion = rot.as_quat()  # (x, y, z, w)

    # Width is observed at the object surface, not at the offset target depth.
    width_m = (float(grasp.width) * z_surface) / intrinsics.fx

    return position, quaternion, width_m
