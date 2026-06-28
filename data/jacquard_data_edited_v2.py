import glob
import os
import numpy as np
from PIL import Image

from .base_grasp_data import BaseGraspDataset
from .utils import image_utils as iu
from .utils import grasp_utils as gu


class JacquardDataset(BaseGraspDataset):
    """
    Jacquard dataset loader with optional direct RGB-D inference mode.

    Normal mode (default):
        - root points to Jacquard dataset
        - samples defined by *_grasps.txt

    RGB-D direct mode:
        - pass rgbd_pairs=[(rgb_path, depth_path[, mask_path[, grasp_path]])]
        - set has_gt=False for pure inference
    """

    def __init__(
        self,
        root,
        start=0.0,
        end=1.0,
        ds_rotate=0,
        rgbd_pairs=None,
        has_gt=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.has_gt = has_gt
        self.rgbd_pairs = rgbd_pairs

        # ---------------------------------------------------------
        # RGB-D DIRECT MODE (bypasses Jacquard folder structure)
        # ---------------------------------------------------------
        if rgbd_pairs is not None:
            self.rgb_files = [p[0] for p in rgbd_pairs]
            self.depth_files = [p[1] for p in rgbd_pairs]

            self.mask_files = (
                [p[2] for p in rgbd_pairs]
                if len(rgbd_pairs[0]) >= 3
                else [None] * len(rgbd_pairs)
            )

            self.grasp_files = (
                [p[3] for p in rgbd_pairs]
                if len(rgbd_pairs[0]) >= 4
                else [None] * len(rgbd_pairs)
            )

            self.length = len(self.rgb_files)
            return

        # ---------------------------------------------------------
        # ORIGINAL JACQUARD MODE (UNCHANGED)
        # ---------------------------------------------------------
        grasp_files = glob.glob(
            os.path.join(root, "**", "*_grasps.txt"), recursive=True
        )
        grasp_files.sort()

        l = len(grasp_files)
        start = int(l * start)
        end = int(l * end)

        self.grasp_files = grasp_files[start:end]

        self.rgb_files = [
            f.replace("_grasps.txt", "_RGB.png") for f in self.grasp_files
        ]
        self.depth_files = [
            f.replace("_grasps.txt", "_perfect_depth.tiff") for f in self.grasp_files
        ]
        self.mask_files = [
            f.replace("_grasps.txt", "_mask.png") for f in self.grasp_files
        ]

        self.length = len(self.grasp_files)

    # ------------------------------------------------------------------
    # REQUIRED HOOKS CALLED BY BaseGraspDataset.__getitem__()
    # ------------------------------------------------------------------

    def get_rgb(self, idx, rot=0, zoom=1.0):
        rgb_path = self.rgb_files[idx]
        rgb_img = iu.Image.from_file(rgb_path)
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        return rgb_img.img

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_path = self.depth_files[idx]

        # Jacquard TIFF (original behavior)
        if depth_path.endswith(".tiff") or depth_path.endswith(".tif"):
            depth_img = iu.DepthImage.from_tiff(depth_path)
            depth_img.rotate(rot)
            depth_img.normalise()
            depth_img.zoom(zoom)
            depth_img.resize((self.output_size, self.output_size))
            return depth_img.img

        # PNG or other image depth
        if depth_path.endswith(".png"):
            depth = np.array(Image.open(depth_path)).astype(np.float32)
        # NPY depth (Contact-GraspNet style)
        elif depth_path.endswith(".npy"):
            depth = np.load(depth_path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")

        # Normalize / clean
        depth[depth <= 0] = -1.0

        # Apply same spatial ops as Jacquard
        depth_img = iu.DepthImage(depth)
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_mask(self, idx, rot=0, zoom=1.0):
        mask_path = self.mask_files[idx]

        # No mask provided → allow everything
        if mask_path is None:
            return np.ones((self.output_size, self.output_size), dtype=np.float32)

        mask_img = iu.Mask.from_file(mask_path)
        mask_img.rotate(rot)
        mask_img.zoom(zoom)
        mask_img.resize((self.output_size, self.output_size))
        return mask_img.img

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        if (not self.has_gt) or (self.grasp_files[idx] is None):
            return gu.GraspRectangles([])

        gtbbs = gu.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # return gu.GraspRectangles.load_from_jacquard_file(
        #     self.grasp_files[idx],
        #     rot=rot,
        #     zoom=zoom,
        #     output_size=self.output_size
        # )

        return gtbbs

    def get_jname(self, idx):
        if self.grasp_files[idx] is None:
            return f"rgbd_{idx}"
        return os.path.basename(self.grasp_files[idx]).replace("_grasps.txt", "")
