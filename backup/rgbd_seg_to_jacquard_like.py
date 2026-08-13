#!/usr/bin/env python3
"""
Convert an RGB image, a depth image (uint16 PNG), and an instance-label image (.npy)
into a Jacquard-like sample triplet:

  <sample_id>_RGB.png
  <sample_id>_mask.png                 (union of all instances, binary 0/255)
  <sample_id>_perfect_depth.tiff        (float32 meters, uncompressed)
  <sample_id>_mask_instance_<id>.png    (one per instance id)

Resizing policy:
  - pad to square (constant 0)
  - resize to out_size x out_size (default 1024)
  - masks use nearest-neighbor; RGB/depth use bilinear.

Depth policy:
  - input depth is uint16 (typically millimeters)
  - output TIFF stores float32 depth in meters: depth_m = depth_u16 * depth_scale_to_m
"""

"""
Usage:
python3 rgbd_seg_to_jacquard_like.py \
  --rgb /path/to/from_rgbd-color.png \
  --depth /path/to/from_rgbd-depth.png \
  --labels /path/to/im_label.npy \
  --out_dir /path/to/out \
  --sample_id 0_1a9fa4c269cfcc1b738e43095496b061 \
  --out_size 1024 \
  --depth_scale_to_m 0.001
"""

"""
Example:
python rgbd_seg_to_jacquard_like.py --rgb ~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/Kinova_Gen3_real/sample23_physical_YCB_mnet_scene/input/from_rgbd-color.png \
    --depth ~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/Kinova_Gen3_real/sample23_physical_YCB_mnet_scene/input/from_rgbd-depth.png 
    --labels ~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/Kinova_Gen3_real/sample23_physical_YCB_mnet_scene/output/segmentation_from_rgbd/im_label.npy \
    --sample_id 0_from_rgbd --out_size 1024 --depth_scale_to_m 0.001 --out_dir out/
"""

import argparse
import json
import os
import numpy as np
from PIL import Image


def to_square_and_resize(arr: np.ndarray, out_size: int = 1024, is_mask: bool = False) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w = arr.shape[:2]

    side = max(h, w)
    pad_y1 = (side - h) // 2
    pad_y2 = side - h - pad_y1
    pad_x1 = (side - w) // 2
    pad_x2 = side - w - pad_x1

    if arr.ndim == 2:
        padded = np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode="constant", constant_values=0)
        img = Image.fromarray(padded)
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize((out_size, out_size), resample=resample)
        return np.array(img)
    else:
        padded = np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode="constant", constant_values=0)
        img = Image.fromarray(padded)
        img = img.resize((out_size, out_size), resample=Image.BILINEAR)
        return np.array(img)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--rgb", required=True, help="RGB image path (png/jpg)", default="~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/input/from_rgbd-color.png")
    ap.add_argument("--depth", required=True, help="Depth image path (uint16 PNG)", default="~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/input/from_rgbd-depth.png")
    ap.add_argument("--labels", required=True, help="Instance label path (.npy), values 0..N", default = "~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/output/segmentation_from_rgbd/im_label.npy")
    ap.add_argument("--out_dir", required=True, help="Output directory", default="rgbd2jacquard/temp")
    ap.add_argument("--sample_id", default="0_from_rgbd", help="Output sample id prefix")
    ap.add_argument("--out_size", type=int, default=1024, help="Output square size (Jacquard commonly uses 1024)")
    ap.add_argument("--depth_scale_to_m", type=float, default=0.001,
                    help="Meters per depth unit in the input uint16 PNG (0.001 for mm)")
    
    # ap.add_argument("--input_base_path", required=False, help="path to UOC", default="~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd/")
    # ap.add_argument("--input_figure_path", required=False, help="path to RGB and depth images (png/jpg)", default="~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd/input")
    # ap.add_argument("--input_label_path", required=False, help="path to im_label", default="~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd/output/segmentation_from_rgbd")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rgb = np.array(Image.open(args.rgb).convert("RGB"))
    depth_u16 = np.array(Image.open(args.depth))
    labels = np.load(args.labels)

    if labels.shape != depth_u16.shape or labels.shape != rgb.shape[:2]:
        raise ValueError(f"Shape mismatch: rgb {rgb.shape[:2]}, depth {depth_u16.shape}, labels {labels.shape}")

    instance_ids = [int(x) for x in np.unique(labels) if int(x) != 0]

    union = (labels != 0).astype(np.uint8) * 255
    per_instance = {iid: (labels == iid).astype(np.uint8) * 255 for iid in instance_ids}

    depth_m = depth_u16.astype(np.float32) * float(args.depth_scale_to_m)

    rgb_out = to_square_and_resize(rgb, out_size=args.out_size, is_mask=False)
    union_out = to_square_and_resize(union, out_size=args.out_size, is_mask=True)
    depth_out = to_square_and_resize(depth_m, out_size=args.out_size, is_mask=False).astype(np.float32)
    per_instance_out = {iid: to_square_and_resize(m, out_size=args.out_size, is_mask=True) for iid, m in per_instance.items()}

    out_rgb_path = os.path.join(args.out_dir, f"{args.sample_id}_RGB.png")
    out_mask_path = os.path.join(args.out_dir, f"{args.sample_id}_mask.png")
    out_depth_path = os.path.join(args.out_dir, f"{args.sample_id}_perfect_depth.tiff")

    Image.fromarray(rgb_out).save(out_rgb_path)
    Image.fromarray(union_out).convert("L").save(out_mask_path)

    # Save float32 depth without compression to avoid requiring extra TIFF codecs
    Image.fromarray(depth_out, mode="F").save(out_depth_path, compression="raw")

    instance_files = []
    for iid, m in per_instance_out.items():
        p = os.path.join(args.out_dir, f"{args.sample_id}_mask_instance_{iid}.png")
        Image.fromarray(m).convert("L").save(p)
        instance_files.append(os.path.basename(p))

    areas = {int(iid): int(np.sum(labels == iid)) for iid in instance_ids}

    meta = {
        "sample_id": args.sample_id,
        "input": {"rgb": os.path.abspath(args.rgb), "depth": os.path.abspath(args.depth), "labels": os.path.abspath(args.labels)},
        "output": {
            "RGB": os.path.basename(out_rgb_path),
            "mask_union": os.path.basename(out_mask_path),
            "perfect_depth_tiff": os.path.basename(out_depth_path),
            "mask_instances": instance_files,
        },
        "instance_ids": instance_ids,
        "instance_pixel_areas_input_res": areas,
        "conversion": {
            "input_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "output_resolution": [int(args.out_size), int(args.out_size)],
            "resize_method": "pad-to-square then resize",
            "depth_scale_to_m": float(args.depth_scale_to_m),
        },
        "note_about_eval_py": (
            "This produces Jacquard-like RGB/mask/depth files. "
            "eval.py (JacquardDataset.get_gtbb) will ALSO require Jacquard-format grasp-annotation files; "
            "those are not generated here because their format depends on the dataset implementation."
        ),
    }

    meta_path = os.path.join(args.out_dir, f"{args.sample_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print(" ", out_rgb_path)
    print(" ", out_mask_path)
    print(" ", out_depth_path)
    print(" ", meta_path)
    for fn in instance_files:
        print(" ", os.path.join(args.out_dir, fn))


if __name__ == "__main__":
    main()
