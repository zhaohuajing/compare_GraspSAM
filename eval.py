import os
import time
import argparse
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.jacquard_data import JacquardDataset
from data.grasp_anything_data import GraspAnythingDataset

from model.planar_grasp_sam import PlanarGraspSAM

import matplotlib.pyplot as plt
from skimage.filters import gaussian
from data.utils.grasp_utils import *



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

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
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

def main(args, i=0):
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    
    if args.dataset_name == "jacquard":
        
        # train_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
        #                                 random_rotate=False, random_zoom=False,
        #                                 start=0.0, end=0.9, seen=True)
        # test_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
        #                                random_rotate=False, random_zoom=False,   
        #                                start=0.9, end=1.0, seen=False)


        train_dataset = JacquardDataset(root=args.root, crop_size=512, include_mask=True, 
                                        random_rotate=False, random_zoom=False,
                                        start=0.0, end=0.9, seen=True)
        test_dataset = JacquardDataset(root=args.root, crop_size=512, include_mask=True, 
                                       random_rotate=False, random_zoom=False,   
                                       start=0.9, end=1.0, seen=False)

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
        
        print("test_dataset size : {}".format(len(test_indices)))
    else:
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=False, 
        #                                           num_workers=4, shuffle=False)
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
    # Added
    #--------------------

    print("loading checkpoint from : ", os.path.basename(ckp_path))

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
    print("loaded. missing:", len(missing), "unexpected:", len(unexpected))

    #--------------------
    # Added end
    #--------------------

    
    print("-"*80) 
     
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
        
            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps    


            print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
            print("Reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
            print("Total    :", torch.cuda.get_device_properties(0).total_memory / 1024**2, "MB")

            # input()
                
            grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)    
            
            lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)


            loss = lossd["g_loss"]
            results['g_loss'] += loss.item() / ld
            for ln, l in lossd['g_losses'].items():
                if ln not in results['g_losses']:
                    results['g_losses'][ln] = 0
                results['g_losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])
                    

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

                # then use these
                gtbb = test_dataset.get_gtbb(didx_s, rot_s, zoom_s)

                #------------------------------
                # Added part ends
                #------------------------------


                #------------------------------
                # Add a one-off visualization for one sample
                #------------------------------

                from data.utils.grasp_utils import GraspRectangles, detect_grasps
                import matplotlib.pyplot as plt

                gs = detect_grasps(q_out, ang_out, width_img=w_out, no_grasps=5)
                gt = gtbb  # already a GraspRectangles

                fig, ax = plt.subplots(1)
                ax.imshow(images[0].permute(1,2,0).cpu().numpy(), cmap='gray')

                gt.plot(ax, color='green')
                for g in gs:
                    g.plot(ax, color='red')

                plt.show()
                plt.savefig(f"sample_{idx}.png")
                plt.close()

   
                #------------------------------
                # Added part ends
                #------------------------------

                success = calculate_iou_match(q_out, ang_out, 
                                              # test_dataset.get_gtbb(didx, rot, zoom_factor), 
                                              gtbb,
                                              # no_grasps=1, 
                                              no_grasps=40, 
                                              grasp_width=w_out)
            
            if success:
                results["correct"] += 1
            else:
                results["failed"] += 1
            
            success_rate = 100 * results["correct"] / (results["correct"] + results["failed"])
            
            
            print("success rate : {:.2f}% | correct : {},  failed : {}".format(success_rate, results["correct"], results["failed"]))

    return 100 * results["correct"] / (results["correct"] + results["failed"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    parser.add_argument("--seen-set", action="store_true", help="seen set")
    parser.add_argument("--dataset_name", type=str, default="jacquard", help="dataset name")
    parser.add_argument("--batch-size", type=int, default=1)
    
    parser.add_argument("--split", type=float, default=0.01)
    parser.add_argument("--root", type=str, help="dataset root")
    parser.add_argument("--ckp_path", type=str, help="ckp_path")

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
