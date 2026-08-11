import os
import time
import argparse
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# from data.jacquard_data import JacquardDataset
from data.jacquard_data_custom import JacquardDataset
from data.grasp_anything_data import GraspAnythingDataset

from model.planar_grasp_sam import PlanarGraspSAM

import matplotlib.pyplot as plt
from skimage.filters import gaussian
from data.utils.grasp_utils import *


"""

Sample input: 

python eval.py   --root ./datasets/YCB_scene_Jac_form/mask_1  --ckp_path ./trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-04-40-49/epoch54.pth   \
        --sam-encoder-type vit_t   --no-grasps 5   --remove_background 0 --apply_mask_to_q 1

python eval.py --root ./rgbd2jacquard/Kinova_Gen3_real_YCB/sample2_mnet_scene   --custom_no_gt 1   --sample-id 0_from_rgbd  \
         --ckp_path ./trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-04-40-49/epoch54.pth   --sam-encoder-type vit_t   --no-grasps 5   \
         --remove_background 0   --apply_mask_to_q 1 --mask_id 4


python eval.py --ckp_path ./trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-04-40-49/epoch54.pth \
        --sam-encoder-type vit_t --root ./datasets/CGN_scene_Jac_form/UOC_sample_scene/ --no-grasps 20
"""

# ------------------------------------------------------------------
# Added: Fix for old Ultralytics checkpoints expecting `ultralytics.yolo`
# ------------------------------------------------------------------
import sys
import types 

try:
    import ultralytics
    import ultralytics.models.yolo
    sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
except Exception as e:
    print("Warning: Ultralytics aliasing failed:", e)


# import random, numpy as np
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

import cv2 

from data.utils.grasp_pose_convert_utils import (
    CameraIntrinsics,
    rectangle_to_pose_topdown
)



# ------------------------------------------------------------------
# Added part ends
# ------------------------------------------------------------------


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    if len(q_img.shape) == 3:
        q_img = F.interpolate(q_img.unsqueeze(0), size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img.unsqueeze(0), size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img.unsqueeze(0), size=(1024, 1024))[0]
        width_img = F.interpolate(width_img.unsqueeze(0), size=(1024, 1024))[0]
        
    elif len(q_img.shape) == 4:
        q_img = F.interpolate(q_img, size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img, size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img, size=(1024, 1024))[0]
        width_img = F.interpolate(width_img, size=(1024, 1024))[0]

    width_scale = 512 # default : 512
    q_img = q_img.data.detach().cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).data.detach().cpu().numpy().squeeze()
    width_img = width_img.data.detach().cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, gs=None): #added gs input
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs


    if gs == None:
        gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)

    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False


def setup_model(model_type, sam_encoder_type):
    if model_type == "bs_grasp_sam":
        model = PlanarGraspSAM(sam_encoder_type=sam_encoder_type)

    else:
        raise("please input correct model type")

    return model


def grasp_pred_to_dict(grasp_pred):
    """
    Normalize model.total_forward() grasp output into a dict with keys:
    pos, cos, sin, width.

    This lets custom_no_gt inference avoid relying on model.compute_loss(), which
    was originally designed for Jacquard GT supervision/evaluation.
    """
    if isinstance(grasp_pred, dict):
        # common variants
        if all(k in grasp_pred for k in ['pos', 'cos', 'sin', 'width']):
            return grasp_pred
        if 'pred' in grasp_pred and isinstance(grasp_pred['pred'], dict):
            return grasp_pred['pred']

    if isinstance(grasp_pred, (list, tuple)) and len(grasp_pred) >= 4:
        return {
            'pos': grasp_pred[0],
            'cos': grasp_pred[1],
            'sin': grasp_pred[2],
            'width': grasp_pred[3],
        }

    raise TypeError(f"Unsupported grasp_pred type/structure: {type(grasp_pred)}")


#-----------------------------
# Add function for data loading from prior custom defined rgbd loader, pending confirmation
#----------------------------

def load_sample(root: str, sample_id: str = "0_from_rgbd"):
    """
    Load RGB and float32-meter depth from a Jacquard-like custom RGB-D folder.

    This is used only for rectangle->6D pose conversion. The actual neural-network
    input still comes from JacquardDataset.
    """
    rgb_path = os.path.join(root, f'{sample_id}_RGB.png')
    depth_path = os.path.join(root, f'{sample_id}_perfect_depth.tiff')

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(rgb_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)

    rgb = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    H, W = rgb.shape[:2]
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    return rgb, depth



#-----------------------------
# Add function for data saving
#----------------------------

def grasp_to_dict(g):
    """
    Convert a GraspSAM grasp (object OR tensor/array) into JSON-safe dict
    """
    # Case 1: Grasp object
    if hasattr(g, "center"):
        return {
            "x": float(g.center[0]),
            "y": float(g.center[1]),
            "angle": float(g.angle),
            "width": float(g.width),
            "score": float(getattr(g, "score", 0.0)),
        }

    # Case 2: Tensor or ndarray [x, y, angle, width, score?]
    import numpy as np
    if hasattr(g, "detach"):
        g = g.detach().cpu().numpy()

    if isinstance(g, (list, tuple, np.ndarray)):
        return {
            "x": float(g[0]),
            "y": float(g[1]),
            "angle": float(g[2]),
            "width": float(g[3]),
            "score": float(g[4]) if len(g) > 4 else 0.0,
        }

    raise TypeError(f"Unsupported grasp type: {type(g)}")

def grasp_to_dict_with_pose(g):
    d = {
        'x': float(g.center[1]),
        'y': float(g.center[0]),
        # 'x': float(g.center[0]),
        # 'y': float(g.center[1]),
        'angle': float(g.angle),
        'width_px': float(g.width),
        'length_px': float(getattr(g, 'length', 0.0)),
        'score': float(getattr(g, 'score', 0.0)),
    }
    if getattr(g, 'pos', None) is not None:
        d['pos'] = [float(v) for v in np.asarray(g.pos).reshape(-1)[:3]]
    if getattr(g, 'quat', None) is not None:
        d['quat'] = [float(v) for v in np.asarray(g.quat).reshape(-1)[:4]]
    if getattr(g, 'width_m', None) is not None:
        d['width_m'] = float(g.width_m)
    return d


#-----------------------------
# Added function ends
#----------------------------

def main(args, i=0):
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    # no_grasps = 10
    no_grasps = args.no_grasps



    #-----------------------------
    # Add to save info to log file
    #----------------------------


    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"grasp_outputs/run_{run_id}"

    # out_dir = "grasp_outputs/sample4"
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "note.txt")
    log_f = open(log_path, "a")   # append mode

    log_f.write("\n" + "="*60 + "\n")
    log_f.write(f"New run | encoder={args.sam_encoder_type} | no_grasps={no_grasps}\n")
    log_f.write("="*60 + "\n")

    # Add to save info to log file
    def log_print(*args): 
        msg = " ".join(str(a) for a in args)
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()   # ensure it's written immediately

    #-----------------------------
    # Added part end
    #----------------------------
    
    if args.dataset_name == "jacquard":
        
        jac_kwargs = dict(
            root=args.root,
            crop_size=1024,
            include_mask=True,
            random_rotate=False,
            random_zoom=False,
            custom_no_gt=bool(args.custom_no_gt),
            mask_id=args.mask_id,
            sample_id=args.sample_id,
        )

        train_dataset = JacquardDataset(start=0.0, end=0.9, seen=True, **jac_kwargs)
        test_dataset = JacquardDataset(start=0.9, end=1.0, seen=False, **jac_kwargs)

        log_print(
            f"JacquardDataset loaded | custom_no_gt={args.custom_no_gt} | "
            f"mask_id={args.mask_id} | sample_id={args.sample_id}"
        )

    elif args.dataset_name == "grasp_anything":
        
        train_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                             random_rotate=False, random_zoom=False,
                                             start=0.0, end=0.9, seen=True)
        test_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                            random_rotate=False, random_zoom=False,
                                            start=0.9, end=1.0, seen=False)

    if args.seen_set:
        indices = list(range(test_dataset.__len__()))
        split = int(np.floor(args.split * train_dataset.__len__()))

        test_indices = indices[split:]
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
        #                                           num_workers=4, shuffle=False, sampler=test_sampler)

        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, pin_memory=False,
                                                  num_workers=0, shuffle=False, sampler=test_sampler) # modified num_workers: 4->0
        
        log_print("test_dataset size : {}".format(len(test_indices)))
    else:
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=False, 
                                                  # num_workers=4, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=False, 
                                                  num_workers=0, shuffle=False) # modified num_workers: 4->0

    
    # model = setup_model(model_type="bs_grasp_sam", sam_encoder_type="eff_vit_t_w_ad") # commented out to avoid hardcoding encoder type to (non-existing) efficent sam cpk
    # model = setup_model(model_type="bs_grasp_sam",sam_encoder_type="vit_h") #added 
    model = setup_model(model_type="bs_grasp_sam",sam_encoder_type=args.sam_encoder_type)

    model   = model.to(args.device)

    ckp_path = args.ckp_path

    '''
    print("loading checkpoint from : ", ckp_path.split("/")[-1])
    
    # state_dict = torch.load(ckp_path, map_location=args.device) # commented out
    state_dict = torch.load(ckp_path, map_location=args.device, weights_only=True) #added


    if "module." in list(state_dict["model"].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
                
        state_dict = new_state_dict

    model.load_state_dict(state_dict["model"], strict=False) # commented out
    '''

    #--------------------
    # Added check point loading session
    #--------------------

    log_print("loading checkpoint from : ", os.path.basename(ckp_path))

    ckpt = torch.load(ckp_path, map_location="cpu")  # CPU load is safer for big pickles

    # Common patterns:
    # 1) {"model": state_dict, ...}
    # 2) {"state_dict": state_dict, ...}
    # 3) directly a state_dict
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt  # might already be a state_dict-like dict
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip DataParallel "module." prefix if present
    if len(sd) > 0:
        k0 = next(iter(sd.keys()))
        if k0.startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    log_print("loaded. missing:", len(missing), "unexpected:", len(unexpected))

    #--------------------
    # Added part ends
    #--------------------

    
    log_print("-"*80) 
     
    ld = len(test_loader)
    results = {"correct": 0, "failed": 0, 
               "g_loss":0, 
               "g_losses":{
                "p_loss": 0,
                "cos_loss": 0,
                "sin_loss": 0,
                "width_loss": 0,
                },}
    
    model.eval()
    with torch.no_grad():
        
        for idx, data in enumerate(test_loader):

            torch.cuda.empty_cache() # temporally added


            images, masks, grasps, didx, rot, zoom_factor = data
            images = images.to(args.device)    
            masks = masks.to(args.device)
            grasps = [g.to(args.device) for g in grasps]
            # grasps = grasps.to(args.device)

            if args.remove_background == 1:
                m = masks.float()
                m = (m > 0).float()   # non-zero -> 1.0, zero -> 0.0
                m3 = m.repeat(1,3,1,1)  # expand [B,1,H,W] -> [B,3,H,W]
                images = images * m3  #+ args.bg_value * (1 - m3)
        
            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps    

            if idx == 0: # added for cuda usage inspection
                log_print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                log_print("Reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
                log_print("Total    :", torch.cuda.get_device_properties(0).total_memory / 1024**2, "MB")

                # input()
                
            grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)    
            
            # Standard Jacquard evaluation uses compute_loss() because GT maps exist.
            # Custom RGB-D inference may have no meaningful GT grasps, so fall back to
            # direct prediction unpacking if compute_loss() fails.
            try:
                lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)

                loss = lossd["g_loss"]
                results['g_loss'] += loss.item() / ld
                for ln, l in lossd['g_losses'].items():
                    if ln not in results['g_losses']:
                        results['g_losses'][ln] = 0
                    results['g_losses'][ln] += l.item() / ld

                pred_dict = lossd['pred']
            except Exception as e:
                if not args.custom_no_gt:
                    raise
                log_print("[custom_no_gt] compute_loss failed; using raw grasp_pred instead:", repr(e))
                pred_dict = grasp_pred_to_dict(grasp_pred)

            q_out, ang_out, w_out = post_process_output(pred_dict['pos'], pred_dict['cos'],
                                                        pred_dict['sin'], pred_dict['width'])

            

            if args.seen_set:
                success = calculate_iou_match(q_out, ang_out, 
                                          train_dataset.get_gtbb(didx, rot, zoom_factor), 
                                          no_grasps=1, 
                                          grasp_width=w_out)
            else:

                #------------------------------
                # Added to force scalar rot/zoom_factor/didx before calling get_gtbb
                #------------------------------

                # ---- make dataloader outputs scalar (batch_size=1 expected) ----
                

                def _scalar(x):
                    # tensor -> python scalar
                    if isinstance(x, torch.Tensor):
                        return x.item() if x.numel() == 1 else x[0].item()
                    # list/tuple -> first element
                    if isinstance(x, (list, tuple)):
                        return _scalar(x[0])
                    # numpy scalar -> python scalar
                    try:
                        import numpy as np
                        if isinstance(x, np.ndarray):
                            return x.item() if x.size == 1 else x.reshape(-1)[0].item()
                    except Exception:
                        pass
                    return x

                didx_s = int(_scalar(didx))
                rot_s = float(_scalar(rot))
                zoom_s = float(_scalar(zoom_factor))

                # then use these when GT exists. For custom RGB-D inference, no GT boxes are meaningful.
                if args.custom_no_gt:
                    gtbb = None
                else:
                    gtbb = test_dataset.get_gtbb(didx_s, rot_s, zoom_s)


                #------------------------------
                # Add a one-off visualization for one sample
                #------------------------------

                # out_dir = "grasp_outputs"
                os.makedirs(out_dir, exist_ok=True)    

                from data.utils.grasp_utils import GraspRectangles, detect_grasps
                import matplotlib.pyplot as plt

                # Optionally apply mask to qmap (sometimes improves localization)
                if args.apply_mask_to_q == 1:
                    # masks is a tensor; take batch 0, channel 0
                    m_np = masks[0, 0].detach().cpu().numpy().astype(np.float32)
                    q_out = q_out * m_np
                    w_out = w_out * m_np

                gs = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=args.no_grasps)
                gt = gtbb  # already a GraspRectangles



                #--------------------------
                # Added for output inspection
                #--------------------------
                log_print("idx = ", idx)
                log_print("q_out:", type(q_out), q_out.shape, q_out.min(), q_out.max())
                log_print("ang_out:", ang_out.shape,
                          np.degrees(ang_out.min()), np.degrees(ang_out.max()), "deg")

                log_print("Detected grasps:", len(gs))

                #--------------------------
                # Added for result inspection and plots
                #--------------------------

                # added for result inspection
                log_print(f"Detected {len(gs)} grasps:")
                for i, g in enumerate(gs):
                    log_print(f"  Grasp {i}: center=({g.center[0]:.1f}, {g.center[1]:.1f}), "
                          f"angle={np.degrees(g.angle):.1f} deg, width={g.width:.1f}px")


                fig, ax = plt.subplots(1)
                ax.imshow(images[0].permute(1,2,0).cpu().numpy(), cmap='gray')

                # gt.plot(ax, color='green') # COMMENT OUT IF USING CUSTOMER IMAGES WITHOUT GROUND TRUTH BOXES
                for g in gs:
                    g.plot(ax, color='red')

                # plt.show()  # disabled for docker/headless runs
                plt.savefig(os.path.join(out_dir, f"sample_{idx}.png"))
                plt.close()

                # --------------------------
                # Visualize grasp quality map
                # --------------------------

                fig, ax = plt.subplots(1, figsize=(6, 6))

                # IMPORTANT: clamp visualization range so low values are visible
                # im = ax.imshow(
                #     q_out,
                #     # cmap='Greys', #'jet',
                #     vmin=0.0,
                #     vmax=max(0.05, np.percentile(q_out, 99.5))
                # )

                # ax.imshow(rgb_vis)
                im = ax.imshow(q_out, 
                          cmap='jet', 
                          alpha=0.9,
                          vmin=0.0,
                          vmax=np.percentile(q_out, 99.5),
                          # vmax=1,
                          )


                for g in gs:
                    g.plot(ax, color='white')

                ax.set_title("Grasp Quality Map (q_out)")
                plt.colorbar(im, ax=ax) #, fraction=0.046)

                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"sample_{idx}_qmap.png"))
                plt.close()

                #--------------------------------
                # Added for converting grasp rectangle to 6d pose
                #--------------------------------
                # Intrinsics must be scaled to current image size
                # The intrinsics provided are for intr_w x intr_h.
                out_size = 1024
                sx = out_size / float(args.intr_w)
                sy = out_size / float(args.intr_h)
                intrinsics = CameraIntrinsics(
                    fx=args.fx * sx,
                    fy=args.fy * sy,
                    cx=args.cx * sx,
                    cy=args.cy * sy,
                )


                rgb_u8, depth_m = load_sample(args.root, sample_id=args.sample_id)
                depth_for_pose = depth_m

                for g in gs:
                    g.pos, g.quat, g.width_m = rectangle_to_pose_topdown(
                        g,
                        depth_for_pose,
                        intrinsics,
                        grasp_height_offset=0.15,
                    )




                #-----------------------------
                # Add data saving for grasps
                #-----------------------------

                # Save dense grasp maps per sample

                np.savez(
                    os.path.join(out_dir, f"sample_{idx}_maps.npz"),
                    q=q_out,
                    angle=ang_out,
                    width=w_out
                )

                # Convert grasps to a clean array and save

                grasp_array = np.array([
                    [
                        g.center[0],     # x (px)
                        g.center[1],     # y (px)
                        g.angle,         # rad
                        g.width          # px
                    ]
                    for g in gs
                ], dtype=np.float32)


                #  Save as npy
                np.save(
                    os.path.join(out_dir, f"sample_{idx}_grasps.npy"),
                    grasp_array
                )

                #  Save as human-readable JSON
                # grasp_list = [
                #     {
                #         "x": float(g.center[0]),
                #         "y": float(g.center[1]),
                #         "angle_rad": float(g.angle),
                #         "angle_deg": float(np.degrees(g.angle)),
                #         "width_px": float(g.width)
                #     }
                #     for g in gs
                # ]

                json_path = os.path.join(out_dir, f"sample_{idx}_grasps.json")

                # with open(os.path.join(out_dir, f"sample_{idx}_grasps.json"), "w") as f:
                #     json.dump(grasp_list, f, indent=2)


                # out_json = {
                #     "sample_id": idx,
                #     "num_grasps": len(grasps),
                #     "grasps": []
                # }

                # for g in grasps:
                #     out_json["grasps"].append([
                #         float(g.center[0]),
                #         float(g.center[1]),
                #         float(g.angle),
                #         float(g.width),
                #         float(g.score)
                #     ])

                # with open(json_path, "w") as f:
                #     json.dump(out_json, f, indent=2)

                # grasp_dicts = []
                # for g in gs:
                #     grasp_dicts.append(grasp_to_dict(g))

                # with open(json_path, "w") as f:
                #     json.dump(grasp_dicts, f, indent=2)

                grasp_dicts = [grasp_to_dict_with_pose(g) for g in gs]
                # with open(os.path.join(out_dir, f'sample_{k}_grasps.json'), 'w') as f:
                with open(os.path.join(out_dir, f'sample_0_grasps.json'), 'w') as f:
                    # json.dump({'sample_id': sid, 'grasps': grasp_dicts}, f, indent=2)
                    json.dump(grasp_dicts, f, indent=2)




                #------------------------------
                # Added part ends
                #------------------------------

                # success = calculate_iou_match(q_out, ang_out, 
                #                               # test_dataset.get_gtbb(didx, rot, zoom_factor), 
                #                               gtbb,
                #                               # no_grasps=1, 
                #                               no_grasps=no_grasps, 
                #                               grasp_width=w_out,
                #                               gs=gs)
            
            # if success:
            #     results["correct"] += 1
            # else:
            #     results["failed"] += 1
            
            # success_rate = 100 * results["correct"] / (results["correct"] + results["failed"])
            
            
            # log_print("success rate : {:.2f}% | correct : {},  failed : {}".format(success_rate, results["correct"], results["failed"]))

    log_f.close()


    # return 100 * results["correct"] / (results["correct"] + results["failed"])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    parser.add_argument("--seen-set", action="store_true", help="seen set")
    parser.add_argument("--dataset_name", type=str, default="jacquard", help="dataset name")
    parser.add_argument("--batch-size", type=int, default=1)
    
    parser.add_argument("--split", type=float, default=0.01)
    parser.add_argument("--root", type=str, help="dataset root")
    parser.add_argument("--ckp_path", type=str, help="ckp_path")
    parser.add_argument("--no-grasps", type=int, default=10, help="Top-K grasps to evaluate")
    parser.add_argument('--remove_background', type=int, default=0, choices=[0, 1],
                        help='If 1, apply mask to RGB input (background filled with bg_value)')
    parser.add_argument('--bg_value', type=float, default=0.0,
                        help='Background fill value in the SAME normalized tensor space as images')
    # Optional: also allow masking qmap for testing (off by default)
    parser.add_argument('--apply_mask_to_q', type=int, default=0, choices=[0, 1],
                        help='If 1, multiply q_out by the mask before detect_grasps')

    # Custom RGB-D / UOC-to-Jacquard-like input options
    parser.add_argument('--custom_no_gt', type=int, default=0, choices=[0, 1],
                        help='If 1, load custom Jacquard-like RGB-D data without requiring meaningful *_grasps.txt')
    parser.add_argument('--mask-id', dest='mask_id', type=int, default=0,
                        help='0 uses <sample_id>_mask.png; N uses <sample_id>_mask_instance_N.png')
    parser.add_argument('--sample-id', dest='sample_id', type=str, default='0_from_rgbd',
                        help='Sample id prefix, e.g., 0_from_rgbd')


    # Added: Intrinsics for converting rectangle->6D
    parser.add_argument('--fx', type=float, default=554.3827128226441)
    parser.add_argument('--fy', type=float, default=554.3827128226441)
    parser.add_argument('--cx', type=float, default=320.0)
    parser.add_argument('--cy', type=float, default=240.0)
    # parser.add_argument('--intr_w', type=int, default=640) # for gazebo panda sim camera
    # parser.add_argument('--intr_h', type=int, default=480) # for gazebo panda sim camera
    parser.add_argument('--intr_w', type=int, default=480) # for physical kinova gen3 camera color (rgb) camera
    parser.add_argument('--intr_h', type=int, default=270) # for physical kinova gen3 camera color (rgb) camera



    # Added to avoid hard-coding encode type
    parser.add_argument(
        "--sam-encoder-type",
        type=str,
        default="vit_h",
        help="SAM backbone type (vit_h, vit_l, vit_b, vit_t, eff_vit_t, eff_vit_t_w_ad)"
    )


    args = parser.parse_args()
    exp_name = time.strftime('%c', time.localtime(time.time()))

    main(args)
