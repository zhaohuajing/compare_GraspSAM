#!/usr/bin/env python3
"""
Convert either:
A) RGB (png/jpg) + depth (uint16 PNG in mm) + labels (.npy)  OR
B) UOC outputs: segmentation.mat (contains rgb + label/label_refined) + sample.npz (contains depth[0,2,:,:] depth in meters)

into Jacquard-like files:
  <sample_id>_RGB.png
  <sample_id>_mask.png
  <sample_id>_perfect_depth.tiff   (float32 meters, uncompressed)
  <sample_id>_mask_instance_<id>.png

Resizing policy: pad to square then resize to out_size (default 1024).
"""


"""

python3 uoc_or_rgbd_to_jacquard_like.py uoc_mat_npz \
  --segmentation_mat segmentation.mat \
  --sample_npz sample.npz \
  --out_dir ./out \
  --sample_id 0_<your_scene_id> \
  --out_size 1024


python3 uoc_or_rgbd_to_jacquard_like.py rgbd_labels   --out_dir ./out   --sample_id 1_from_rgbd   --out_size 1024 --rgb from_rgbd-color.png --depth from_rgbd-depth.png --labels im_label.npy
python3 uoc_or_rgbd_to_jacquard_like.py uoc_mat_npz   --segmentation_mat segmentation.mat   --sample_npz sample.npz   --out_dir ./UOC_sample_scene/   --sample_id uoc_0   --out_size 1024


"""
import argparse
import json
import os
import numpy as np
from PIL import Image

try:
    import scipy.io as sio
except Exception as e:
    sio = None


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
        return np.array(img.resize((out_size, out_size), resample=resample))
    else:
        padded = np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode="constant", constant_values=0)
        img = Image.fromarray(padded)
        return np.array(img.resize((out_size, out_size), resample=Image.BILINEAR))


def save_outputs(rgb: np.ndarray, depth_m: np.ndarray, labels: np.ndarray, out_dir: str, sample_id: str, out_size: int,
                 meta_extra: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, f"{sample_id}"), exist_ok=True)

    if labels.ndim == 3:
        labels = labels[..., 0]
    labels = labels.astype(np.int32)

    instance_ids = [int(x) for x in np.unique(labels) if int(x) != 0]
    union = (labels != 0).astype(np.uint8) * 255
    per_instance = {iid: (labels == iid).astype(np.uint8) * 255 for iid in instance_ids}

    rgb_out = to_square_and_resize(rgb, out_size=out_size, is_mask=False)
    union_out = to_square_and_resize(union, out_size=out_size, is_mask=True)
    depth_out = to_square_and_resize(depth_m.astype(np.float32), out_size=out_size, is_mask=False).astype(np.float32)
    per_instance_out = {iid: to_square_and_resize(m, out_size=out_size, is_mask=True) for iid, m in per_instance.items()}

    out_rgb_path = os.path.join(out_dir, f"{sample_id}/{sample_id}_RGB.png")
    out_mask_path = os.path.join(out_dir, f"{sample_id}/{sample_id}_mask.png")
    out_depth_path = os.path.join(out_dir, f"{sample_id}/{sample_id}_perfect_depth.tiff")

    Image.fromarray(rgb_out).save(out_rgb_path)
    Image.fromarray(union_out).convert("L").save(out_mask_path)
    Image.fromarray(depth_out, mode="F").save(out_depth_path, compression="raw")

    inst_files = []
    for iid, m in per_instance_out.items():
        p = os.path.join(out_dir, f"{sample_id}/{sample_id}_mask_instance_{iid}.png")
        Image.fromarray(m).convert("L").save(p)
        inst_files.append(os.path.basename(p))

    meta = {
        "sample_id": sample_id,
        "output": {
            "RGB": os.path.basename(out_rgb_path),
            "mask_union": os.path.basename(out_mask_path),
            "perfect_depth_tiff": os.path.basename(out_depth_path),
            "mask_instances": inst_files,
        },
        "instance_ids": instance_ids,
        "conversion": {
            "input_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "output_resolution": [int(out_size), int(out_size)],
            "resize_method": "pad-to-square then resize",
            "depth_units": "meters (float32 TIFF)",
        },
        **meta_extra,
    }

    meta_path = os.path.join(out_dir, f"{sample_id}/{sample_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"rgb": out_rgb_path, "mask": out_mask_path, "depth": out_depth_path, "meta": meta_path, "instance_masks": inst_files}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    # Mode A
    ap_a = sub.add_parser("rgbd_labels", help="RGB + depth(uint16 png) + labels(.npy)")
    ap_a.add_argument("--path", required=False, 
        default = "/home/csrobot/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/datasets/sample_scene_ucn/demo_sample/YCB_scene/", 
        help="Path to folde with rgbd image + segmentation")
    ap_a.add_argument("--rgb", required=True)
    ap_a.add_argument("--depth", required=True, help="uint16 PNG (usually mm)")
    ap_a.add_argument("--labels", required=True)
    ap_a.add_argument("--depth_scale_to_m", type=float, default=0.001, help="meters per uint16 unit (0.001 if mm)")
    ap_a.add_argument("--out_dir", required=True)
    ap_a.add_argument("--sample_id", required=True)
    ap_a.add_argument("--out_size", type=int, default=1024)

    # Mode B
    ap_b = sub.add_parser("uoc_mat_npz", help="UOC segmentation.mat + sample.npz")
    ap_b.add_argument("--path", required=False, 
        default = "/home/csrobot/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/datasets/sample_scene_ucn/demo_sample/uoc_sample_scene/segmentation_0", 
        help="Path to folde with rgbd image + segmentation")
    ap_b.add_argument("--segmentation_mat", required=True, help="MAT file with keys rgb + label/label_refined")
    ap_b.add_argument("--sample_npz", required=True, help="NPZ with key 'depth' shaped (1,3,H,W); uses channel 2 as Z")
    ap_b.add_argument("--out_dir", required=False, default = "./output")
    ap_b.add_argument("--sample_id", required=True)
    ap_b.add_argument("--out_size", type=int, default=1024)

    args = ap.parse_args()

    if args.mode == "rgbd_labels":
        rgb_path = os.path.join(args.path, args.rgb)
        depth_path = os.path.join(args.path, args.depth)
        labels_path = os.path.join(args.path, args.labels)
        

        # rgb = np.array(Image.open(args.rgb).convert("RGB"))
        # depth_u16 = np.array(Image.open(args.depth))
        # labels = np.load(args.labels)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth_u16 = np.array(Image.open(depth_path))
        labels = np.load(labels_path)

        depth_m = depth_u16.astype(np.float32) * float(args.depth_scale_to_m)

        if labels.shape[:2] != depth_u16.shape or rgb.shape[:2] != depth_u16.shape:
            raise ValueError(f"Shape mismatch: rgb {rgb.shape[:2]}, depth {depth_u16.shape}, labels {labels.shape}")

        # meta_extra = {"inputs": {"rgb": os.path.abspath(args.rgb), "depth": os.path.abspath(args.depth), "labels": os.path.abspath(args.labels)},
        meta_extra = {"inputs": {"rgb": os.path.abspath(rgb_path), "depth": os.path.abspath(depth_path), "labels": os.path.abspath(labels_path)},
                      "depth_source": f"uint16 PNG scaled by {args.depth_scale_to_m} m/unit"}
        out = save_outputs(rgb, depth_m, labels, args.out_dir, args.sample_id, args.out_size, meta_extra)

    else:
        mat_path = os.path.join(args.path, args.segmentation_mat)
        npz_path = os.path.join(args.path, args.sample_npz)
        print(F"mat_path= {mat_path}")

        if sio is None:
            raise ImportError("scipy is required for reading .mat files (pip install scipy)")
        # mat = sio.loadmat(args.segmentation_mat)
        mat = sio.loadmat(mat_path)
        if "rgb" not in mat:
            raise ValueError("segmentation.mat must contain key 'rgb'")
        rgb = mat["rgb"].astype(np.uint8)
        labels = mat.get("label_refined", mat.get("label"))
        if labels is None:
            raise ValueError("segmentation.mat must contain key 'label' or 'label_refined'")
        labels = labels.astype(np.int32)

        npz = np.load(npz_path, allow_pickle=True)
        if "depth" not in npz:
            raise ValueError("sample.npz must contain key 'depth'")
        depth = npz["depth"]
        if depth.ndim != 4 or depth.shape[1] != 3:
            raise ValueError(f"Expected depth shape (1,3,H,W); got {depth.shape}")
        depth_m = depth[0, 2].astype(np.float32)

        if rgb.shape[:2] != labels.shape or labels.shape != depth_m.shape:
            raise ValueError(f"Shape mismatch: rgb {rgb.shape}, labels {labels.shape}, depth_m {depth_m.shape}")

        meta_extra = {"inputs": {"segmentation_mat": os.path.abspath(args.segmentation_mat), "sample_npz": os.path.abspath(args.sample_npz)},
                      "depth_source": "sample.npz depth[0,2,:,:] (Z channel, meters)"}
        out = save_outputs(rgb, depth_m, labels, args.out_dir, args.sample_id, args.out_size, meta_extra)

    print("Wrote:")
    print("  RGB :", out["rgb"])
    print("  Mask:", out["mask"])
    print("  Depth:", out["depth"])
    print("  Meta:", out["meta"])
    print("  Instance masks:", ", ".join(out["instance_masks"]))


if __name__ == "__main__":
    main()
