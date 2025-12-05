import os
import sys

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as transforms


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="your/path/ckpt/model_tracker_fixed_e20.pt",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="VGGT")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument(
        "--merging", type=int, default=0, help="VGGT aggregator merging steps"
    )
    parser.add_argument("--kf", type=int, default=2, help="key frame")
    return parser


def main(args):

    print("start eval 7scenes & NRGBD!")
    
    from data import SevenScenes, NRGBD
    from utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError
    
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="your/path/7scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="your/path/neural_rgbd_data",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),
    }

    device = args.device
    model_name = args.model_name

    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from criterion import Regr3D_t_ScaleShiftInv, L21

    # Force use of bf16 data type
    dtype = torch.bfloat16
    # Load VGGT model
    model = VGGT()

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print("success load model!")

    model.update_patch_dimensions(518//14,392//14)
    
    os.makedirs(osp.join(args.output_dir, f"{args.kf}"), exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(osp.join(args.output_dir, f"{args.kf}"), name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, "logs.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0
            scene_infer_times = defaultdict(list)

            for data_idx in tqdm(range(len(dataset))):
                
                batch = default_collate([dataset[data_idx]])
                ignore_keys = set(
                    [
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                for view in batch:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []
                in_camera1 = None

                dtype = (
                    torch.bfloat16
                    if torch.cuda.get_device_capability()[0] >= 8
                    else torch.float16
                )
                with torch.cuda.amp.autocast(dtype=dtype):
                    if isinstance(batch, dict) and "img" in batch:
                        batch["img"] = (batch["img"] + 1.0) / 2.0
                    elif isinstance(batch, list) and all(
                        isinstance(v, dict) and "img" in v for v in batch
                    ):
                        for view in batch:
                            view["img"] = (view["img"] + 1.0) / 2.0
                        # Gather all `img` tensors into a single tensor of shape [N, C, H, W]
                        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)
                        
                        

                
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start = time.time()
                    preds = model(imgs_tensor)
                    torch.cuda.synchronize()
                    end = time.time()
                    inference_time_ms = (end - start) * 1000
                    print(f"Inference time: {inference_time_ms:.2f}ms")
                    
                with torch.cuda.amp.autocast(dtype=dtype):

                    # Wrap model outputs per-view to align with batch later
                    predictions = preds
                    views = batch  # list[dict]
                    if "pose_enc" in predictions:
                        B, S = predictions["pose_enc"].shape[:2]
                    elif "world_points" in predictions:
                        B, S = predictions["world_points"].shape[:2]
                    else:
                        raise KeyError(
                            "predictions is missing a key to infer sequence length"
                        )

                    ress = []

                    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs_tensor.shape[-2:])
                    point_map_pre = unproject_depth_map_to_point_map(predictions["depth"].squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))
                    point_map_pre = torch.from_numpy(point_map_pre).float()   # 转 tensor
                    point_map_pre = point_map_pre.unsqueeze(0).to(device)

                    for s in range(S):

                        res = {
                            "pts3d_in_other_view": point_map_pre[:, s],
                            "conf": predictions["depth_conf"][:, s],
                            "camera_pose": predictions["pose_enc"][:, s, :],
                        }

                        if (
                            isinstance(views, list)
                            and s < len(views)
                            and "valid_mask" in views[s]
                        ):
                            res["valid_mask"] = views[s]["valid_mask"]
                        if "track" in predictions:
                            res.update(
                                {
                                    "track": predictions["track"][:, s],
                                    "vis": (
                                        predictions.get("vis", None)[:, s]
                                        if "vis" in predictions
                                        else None
                                    ),
                                    "track_conf": (
                                        predictions.get("conf", None)[:, s]
                                        if "conf" in predictions
                                        else None
                                    ),
                                }
                            )
                        ress.append(res)

                    preds = ress

                    valid_length = len(preds) // args.revisit
                    if args.revisit > 1:
                        preds = preds[-valid_length:]
                        batch = batch[-valid_length:]

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        pts = pred_pts[j].cpu().numpy()[0]
                        conf = preds[j]["conf"].cpu().data.numpy()[0]

                        # mask = mask & (conf > 1.8)

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        images_all.append(image[None, ...])
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)

                scene_id = view["label"][0].rsplit("/", 1)[0]
                # Record average inference time per scene
                try:
                    scene_infer_times[scene_id].append(float(inference_time_ms))
                except Exception:
                    pass

                save_params = {}

                save_params["images_all"] = images_all
                save_params["pts_all"] = pts_all
                save_params["pts_gt_all"] = pts_gt_all
                save_params["masks_all"] = masks_all

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                mask = np.isfinite(pts_all_masked)
                pts_all_masked = pts_all_masked[mask]

                mask_gt = np.isfinite(pts_gt_all_masked)
                pts_gt_all_masked = pts_gt_all_masked[mask_gt]
                images_all_masked = images_all_masked[mask]

                # Reshape to point cloud (N, 3) before sampling
                pts_all_masked = pts_all_masked.reshape(-1, 3)
                pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                images_all_masked = images_all_masked.reshape(-1, 3)

                # If number of points exceeds threshold, sample by points
                if pts_all_masked.shape[0] > 999999:
                    sample_indices = np.random.choice(
                        pts_all_masked.shape[0], 999999, replace=False
                    )
                    pts_all_masked = pts_all_masked[sample_indices]
                    images_all_masked = images_all_masked[sample_indices]

                # Apply the same sampling to GT point cloud
                if pts_gt_all_masked.shape[0] > 999999:
                    sample_indices_gt = np.random.choice(
                        pts_gt_all_masked.shape[0], 999999, replace=False
                    )
                    pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                if args.use_proj:

                    def umeyama_alignment(
                        src: np.ndarray, dst: np.ndarray, with_scale: bool = True
                    ):
                        assert src.shape == dst.shape
                        N, dim = src.shape

                        mu_src = src.mean(axis=0)
                        mu_dst = dst.mean(axis=0)
                        src_c = src - mu_src
                        dst_c = dst - mu_dst

                        Sigma = dst_c.T @ src_c / N  # (3,3)

                        U, D, Vt = np.linalg.svd(Sigma)

                        S = np.eye(dim)
                        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                            S[-1, -1] = -1

                        R = U @ S @ Vt

                        if with_scale:
                            var_src = (src_c**2).sum() / N
                            s = (D * S.diagonal()).sum() / var_src
                        else:
                            s = 1.0

                        t = mu_dst - s * R @ mu_src

                        return s, R, t

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    s, R, t = umeyama_alignment(
                        pts_all_masked, pts_gt_all_masked, with_scale=True
                    )
                    pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                    pts_all_masked = pts_all_aligned

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

                trans_init = np.eye(4)

                threshold = 0.1
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation

                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                    file=open(log_file, "a"),
                )

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med

                # release cuda memory
                torch.cuda.empty_cache()

            # Get depth from pcd and run TSDFusion
            to_write = ""
            # Read the log file
            if os.path.exists(osp.join(save_path, "logs.txt")):
                with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
                    to_write += f_sub.read()

            with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                log_data = to_write
                metrics = defaultdict(list)
                for line in log_data.strip().split("\n"):
                    match = regex.match(line)
                    if match:
                        data = match.groupdict()
                        # Exclude 'scene_id' from metrics as it's an identifier
                        for key, value in data.items():
                            if key != "scene_id":
                                metrics[key].append(float(value))
                        metrics["nc"].append(
                            (float(data["nc1"]) + float(data["nc2"])) / 2
                        )
                        metrics["nc_med"].append(
                            (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                        )
                mean_metrics = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }

                c_name = "mean"
                print_str = f"{c_name.ljust(20)}: "
                for m_name in mean_metrics:
                    print_num = np.mean(mean_metrics[m_name])
                    print_str = print_str + f"{m_name}: {print_num:.3f} | "
                print_str = print_str + "\n"

                # Summarize per-scene average inference time
                time_lines = []
                all_times = []  

                for sid, times in scene_infer_times.items():
                    if len(times) > 0:
                        avg_time = np.mean(times)
                        time_lines.append(f"Idx: {sid}, Time_avg_ms: {avg_time:.2f}")
                        all_times.extend(times)


                if len(all_times) > 0:
                    overall_avg_time = np.mean(all_times)
                    time_lines.append(f"\nOverall_Avg_Time_ms: {overall_avg_time:.2f}")

                time_block = "\n".join(time_lines) + (
                    "\n" if len(time_lines) > 0 else ""
                )

                f.write(to_write + time_block + print_str)

    print("done!")

from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
