import torch
import os
import numpy as np
import argparse
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.eval_utils import load_image_file_crop
from vggt.utils.geometry import unproject_depth_map_to_point_map
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


def save_ply(points, colors, SAVE_ROOT, name, max_points=10000):
    """
    Save colored point cloud to a PLY file (ASCII format).  
    If the number of points exceeds `max_points`, random sampling is applied.

    Args:
        points (np.ndarray): (N, 3) point cloud coordinates in world space.
        colors (np.ndarray): (N, 3) RGB colors in [0, 255].
        SAVE_ROOT (str): Output directory.
        name (str): Output file name.
        max_points (int): Maximum number of points to store (default 10000).
    """
    assert points.shape[0] == colors.shape[0], "Points and colors must have the same length."

    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Remove invalid points (NaN or Inf)
    valid = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points, colors = points[valid], colors[valid].astype(np.uint8)

    total_points = len(points)
    if total_points == 0:
        print("⚠️ No valid points to save.")
        return

    # Random sampling if too many points
    if total_points > max_points:
        print(f"🔽 Too many points ({total_points}). Randomly sampling {max_points} points...")
        idx = np.random.choice(total_points, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    # Write PLY file (ASCII)
    save_path = os.path.join(SAVE_ROOT, name)
    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    print(f"✅ Point cloud saved: {save_path} ({len(points)} points)")


def get_args_parser():
    parser = argparse.ArgumentParser("run LiteVGGT demo", add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/LiteVGGT/te_dict.pt",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="VGGT")
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/home/your/image_dir/path",
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=0.42, help="ratio of points to keep based on confidence"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/output_dir",
        help="value for outdir",
    )
    return parser

def main(args):

    device = args.device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT().to(device)

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(torch.bfloat16)
    model.eval()
    print("Model loaded")

    img_dir = args.img_dir

    all_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    # all_images = [
    #     os.path.join(img_dir, f)
    #     for f in os.listdir(img_dir)
    #     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    #     and "color" in f.lower()
    # ]

    images=[]
    for img_path in all_images:

        # (H, W, 3) 0-1
        img = load_image_file_crop(img_path)
        # print(img.shape)
        # 3 h w 0-1
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        images.append(img)

    N = len(images)
    N_aligned = (N // 8) * 8 

    # # (N, 3, H, W) 0-1
    images = torch.stack(images[:N_aligned], dim=0).to(device)


    print(f"✅ images: {images.shape}")

    patch_width = images.shape[-1] // 14
    patch_height = images.shape[-2] // 14
    model.update_patch_dimensions(patch_width, patch_height)

    images = images[None] 

    with torch.no_grad():
        fp8_format=Format.E4M3
        fp8_recipe = DelayedScaling(
                fp8_format=fp8_format,
                amax_history_len=80,
                amax_compute_algo="max",
            )
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)

        with torch.amp.autocast("cuda",enabled=True, dtype=dtype):
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, patch_start_idx)

        # numpy (S, H, W, 3) word
        points_3d = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    w2c_pre.squeeze(0), 
                                                                    intrinsic.squeeze(0))
        


        # (N,3) array
        points = points_3d.reshape(-1, 3) 
        # images [1, S, 3, H, W]
        color_image = images[0].permute(0, 2, 3, 1).reshape(-1,3).cpu().numpy()
        colors = np.clip(color_image * 255.0, 0, 255).astype(np.uint8)
        
        conf_flat = depth_conf.reshape(-1).cpu().numpy()

        num_keep = int(len(conf_flat) * args.keep_ratio)
        sorted_indices = np.argsort(conf_flat)[::-1]  
        keep_indices = sorted_indices[:num_keep]

        points = points[keep_indices]
        colors = colors[keep_indices]
        conf_flat = conf_flat[keep_indices]

        save_ply(points,colors,args.output_dir,name="recon.ply",max_points=15000000)

        print("done!")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)