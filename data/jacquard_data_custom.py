import os
import glob
import pickle

import numpy as np

from .base_grasp_data import BaseGraspDataset
from .utils import grasp_utils as gu
from .utils import image_utils as iu


class JacquardDataset(BaseGraspDataset):
    """
    Dataset wrapper for Jacquard-format data, with an added custom-RGBD mode.

    Normal Jacquard mode:
      - discovers samples from *_grasps.txt files
      - uses the corresponding *_RGB.png, *_perfect_depth.tiff, *_mask.png

    Custom RGB-D / UOC mode (custom_no_gt=True):
      - discovers samples from *_RGB.png files instead of *_grasps.txt
      - does NOT require a meaningful *_grasps.txt
      - get_gtbb() returns an empty GraspRectangles object when no grasp file exists
      - can use either the union mask or a specific instance mask:
            mask_id=0  -> <sample_id>_mask.png
            mask_id=N  -> <sample_id>_mask_instance_N.png
    """

    def __init__(
        self,
        root,
        start=0.0,
        end=1.0,
        ds_rotate=0,
        custom_no_gt=False,
        mask_id=0,
        sample_id=None,
        recursive_custom=False,
        **kwargs,
    ):
        super(JacquardDataset, self).__init__(**kwargs)

        self.root = root
        self.custom_no_gt = bool(custom_no_gt)
        self.mask_id = int(mask_id) if mask_id is not None else 0
        self.sample_id = sample_id

        if self.custom_no_gt:
            self._init_custom_rgbd(
                root=root,
                mask_id=self.mask_id,
                sample_id=sample_id,
                recursive_custom=recursive_custom,
            )
        else:
            self._init_jacquard_gt(root=root, start=start, end=end, ds_rotate=ds_rotate)

    # ------------------------------------------------------------------
    # Initializers
    # ------------------------------------------------------------------
    def _init_jacquard_gt(self, root, start=0.0, end=1.0, ds_rotate=0):
        graspf = glob.glob(os.path.join(root, '**', '*_grasps.txt'), recursive=True)
        graspf.sort()
        l = len(graspf)
        print("len jacquard grasp files:", l)

        if l == 0:
            raise FileNotFoundError('No *_grasps.txt files found. Check path: {}'.format(root))

        if ds_rotate:
            graspf = graspf[int(l * ds_rotate):] + graspf[:int(l * ds_rotate)]

        # Preserve your current behavior: no start/end split filtering.
        self.grasp_files = graspf
        self.depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
        self.rgb_files = [f.replace('perfect_depth.tiff', 'RGB.png') for f in self.depth_files]
        self.mask_files = [f.replace('perfect_depth.tiff', 'mask.png') for f in self.depth_files]

        self._validate_files(require_grasp=True)

    def _init_custom_rgbd(self, root, mask_id=0, sample_id=None, recursive_custom=False):
        pattern_root = os.path.join(root, '**', '*_RGB.png') if recursive_custom else os.path.join(root, '*_RGB.png')
        rgbf = glob.glob(pattern_root, recursive=recursive_custom)
        rgbf.sort()

        # Optional: restrict to one known sample id, e.g., 0_from_rgbd
        if sample_id:
            rgbf = [p for p in rgbf if os.path.basename(p) == f'{sample_id}_RGB.png']

        if len(rgbf) == 0:
            raise FileNotFoundError(
                'No *_RGB.png files found for custom RGB-D mode. Check path: {}'.format(root)
            )

        depthf = []
        maskf = []
        graspf = []

        for rgb_path in rgbf:
            folder = os.path.dirname(rgb_path)
            sid = self._sample_id_from_rgb(rgb_path)

            depth_path = os.path.join(folder, f'{sid}_perfect_depth.tiff')
            mask_path = self._pick_mask_path(folder, sid, mask_id)
            grasp_path = os.path.join(folder, f'{sid}_grasps.txt')

            depthf.append(depth_path)
            maskf.append(mask_path)
            # Keep a list with the same length as rgb_files so BaseGraspDataset.__len__
            # works even if it checks grasp_files.
            graspf.append(grasp_path if os.path.exists(grasp_path) else None)

        self.rgb_files = rgbf
        self.depth_files = depthf
        self.mask_files = maskf
        self.grasp_files = graspf

        print(f"len custom RGB-D samples: {len(self.rgb_files)} | mask_id={mask_id} | custom_no_gt=True")
        self._validate_files(require_grasp=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_id_from_rgb(rgb_path):
        name = os.path.basename(rgb_path)
        if not name.endswith('_RGB.png'):
            raise ValueError(f'Expected *_RGB.png, got: {rgb_path}')
        return name[:-len('_RGB.png')]

    @staticmethod
    def _pick_mask_path(folder, sid, mask_id):
        if mask_id is None or int(mask_id) == 0:
            return os.path.join(folder, f'{sid}_mask.png')

        instance_path = os.path.join(folder, f'{sid}_mask_instance_{int(mask_id)}.png')
        if os.path.exists(instance_path):
            return instance_path

        raise FileNotFoundError(
            f'Requested mask_id={mask_id}, but instance mask does not exist: {instance_path}'
        )

    def _validate_files(self, require_grasp=False):
        for p in self.rgb_files:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        for p in self.depth_files:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        for p in self.mask_files:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        if require_grasp:
            for p in self.grasp_files:
                if not p or not os.path.exists(p):
                    raise FileNotFoundError(str(p))

    # ------------------------------------------------------------------
    # Required dataset API
    # ------------------------------------------------------------------
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        grasp_path = self.grasp_files[idx] if idx < len(self.grasp_files) else None

        # Custom RGB-D mode: no meaningful GT grasp rectangles are required.
        # Return an empty container so BaseGraspDataset can still draw zero target maps.
        if self.custom_no_gt or grasp_path is None or not os.path.exists(grasp_path):
            return gu.GraspRectangles([])

        gtbbs = gu.GraspRectangles.load_from_jacquard_file(
            grasp_path,
            scale=self.output_size / 1024.0,
        )
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = iu.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = iu.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_mask(self, idx, rot=0, zoom=1.0):
        mask_image = iu.Mask.from_file(self.mask_files[idx])
        mask_image.rotate(rot)
        mask_image.zoom(zoom)
        mask_image.resize((self.output_size, self.output_size))
        mask_image.normalise()
        return mask_image.img

    def get_jname(self, idx):
        grasp_path = self.grasp_files[idx] if idx < len(self.grasp_files) else None
        if grasp_path:
            return '_'.join(os.path.basename(grasp_path).split('_')[:-1])
        return self._sample_id_from_rgb(self.rgb_files[idx])
