import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default="your/path/scannetv2/process_scannetv2"
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=Path,
        default="your/path/scannetv2/OpenDataLab___ScanNet_v2/raw/scans",
    )
    
    parser.add_argument("--output_path", type=Path, default="./eval_results")
    parser.add_argument(
        "--merging", type=int, default=0, help="VGGT aggregator merging steps"
    )
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold for filtering low confidence depth values",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold in Chamfer Distance computation, distances exceeding this value will be clipped",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames selected for processing per scene",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50,
        help="Maximum number of scenes to evaluate",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./ckpt/model_tracker_fixed_e20.pt",
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Whether to visualize attention maps during inference",
    )
    args = parser.parse_args()
    torch.manual_seed(33)

    # Scene sampling
    scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    print(f"Evaluate {len(scannet_scenes)} scenes")

    all_scenes_metrics = {"scenes": {}, "average": {}}
    # Force use of bf16 data type
    dtype = torch.bfloat16
    # Load VGGT model
    model = VGGT()
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)

    # Process each scene
    for scene in scannet_scenes:
        scene_dir = args.data_dir / f"{scene}"
        output_scene_dir = args.output_path / f"input_frame_{args.input_frame}" / scene
        if (output_scene_dir / "metrics.json").exists():
            continue

        # Load scene data
        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        image_paths = get_sorted_image_paths(images_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        if (
            poses_gt is None
            or first_gt_pose is None
            or available_pose_frame_ids is None
        ):
            print(f"Skipping scene {scene}: no pose data")
            continue
        
        all_image_frame_ids = [int(path.stem) for path in image_paths]
        valid_frame_ids = sorted(
            list(set(all_image_frame_ids) & set(available_pose_frame_ids))
        )

        if len(valid_frame_ids) < args.input_frame:
            print(f"Skipping scene {scene}: only {len(valid_frame_ids)} frames (need {args.input_frame})")
            continue

        # Frame filtering
        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )

        # Get corresponding poses
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths

        if len(image_paths) == 0:
            print(f"No images found in {images_dir}")
            continue

        print("🚩Processing", scene, f"Found {len(image_paths)} images")
        all_cam_to_world_mat = []
        all_world_points = []

        try:
            # Load images
            images = load_images_rgb(image_paths)

            if not images or len(images) < 3:
                print(f"Skipping {scene}: insufficient valid images")
                continue

            frame_ids = selected_frame_ids
            images_array = np.stack(images)
            vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
            print(f"Patch dimensions: {patch_width}x{patch_height}")

            # Update model attention layers with dynamic patch dimensions
            model.update_patch_dimensions(patch_width, patch_height)

            # Inference + Reconstruction
            (
                extrinsic_np,
                intrinsic_np,
                all_world_points,
                all_point_colors,
                all_cam_to_world_mat,
                inference_time_ms,
            ) = infer_vggt_and_reconstruct(
                model, vgg_input, dtype, args.depth_conf_thresh, image_paths
            )
            print(f"Inference time: {inference_time_ms:.2f}ms")

            # Process results
            if not all_cam_to_world_mat or not all_world_points:
                print(
                    f"Skipping {scene}: failed to obtain valid camera poses or point clouds"
                )
                continue

            # Evaluate and save
            metrics = evaluate_scene_and_save(
                scene,
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                args.gt_ply_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
            )
            if metrics is not None:
                all_scenes_metrics["scenes"][scene] = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key
                    in [
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                    ]
                }
                print("Complete metrics", all_scenes_metrics["scenes"][scene])

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback

            traceback.print_exc()

    # Summarize average metrics and save
    compute_average_metrics_and_save(
        all_scenes_metrics,
        args.output_path,
        args.input_frame,
    )
