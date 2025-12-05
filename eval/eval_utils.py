import numpy as np
import os

def umeyama_alignment(src, dst, with_scale=True):
    """
    src, dst: (3, N)
    Returns: scale s, rotation R, translation t
    """
    assert src.shape[0] == 3 and dst.shape[0] == 3
    mean_src = np.mean(src, axis=1, keepdims=True)
    mean_dst = np.mean(dst, axis=1, keepdims=True)

    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov = dst_centered @ src_centered.T / src.shape[1]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(src_centered ** 2) / src.shape[1]
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0

    t = mean_dst.flatten() - s * R @ mean_src.flatten()
    return s, R, t

def align_pre_to_gt(pred_poses,gt_poses, with_scale=True):

    assert len(gt_poses) == len(pred_poses), "Pose counts must match."

    # ---- Step 1: Compute camera centers
    def get_cam_center(Tcw):
        R = Tcw[:3, :3]
        t = Tcw[:3, 3]
        return -R.T @ t  # camera center in world coords

    gt_positions = np.array([get_cam_center(T) for T in gt_poses])  # (N,3)
    pred_positions = np.array([get_cam_center(T) for T in pred_poses])  # (N,3)

    # print("first gt position:",gt_positions[0])
    # print("first pre position:",pred_positions[0])

    # ---- Step 2: Compute Umeyama alignment (GT -> Pred)
    s, R, t = umeyama_alignment(pred_positions.T, gt_positions.T, with_scale)

    return s, R, t 


def align_gt_to_pred(gt_poses, pred_poses, with_scale=True):

    assert len(gt_poses) == len(pred_poses), "Pose counts must match."

    # ---- Step 1: Compute camera centers
    def get_cam_center(Tcw):
        R = Tcw[:3, :3]
        t = Tcw[:3, 3]
        return -R.T @ t  # camera center in world coords

    gt_poses = normalize_to_first_pose(gt_poses)
    gt_positions = np.array([get_cam_center(T) for T in gt_poses])  # (N,3)
    pred_positions = np.array([get_cam_center(T) for T in pred_poses])  # (N,3)

    print("first gt position:",gt_positions[0])
    print("first pre position:",pred_positions[0])

    # ---- Step 2: Compute Umeyama alignment (GT -> Pred)
    s, R, t = umeyama_alignment(gt_positions.T, pred_positions.T, with_scale)

    aligned_gt_poses = []
    for T in gt_poses:
        R_gt = T[:3, :3]
        t_gt = T[:3, 3]
        C_gt = -R_gt.T @ t_gt  # camera center in world

        # ---- Step 3: Align GT camera to Pred world
        C_aligned = s * (R @ C_gt) + t
        R_aligned = R @ R_gt

        # ---- Step 4: Rebuild Tcw (world→camera)
        T_new = np.eye(4)
        T_new[:3, :3] = R_aligned
        T_new[:3, 3] = -R_aligned @ C_aligned
        aligned_gt_poses.append(T_new)

    print(f"✅ Alignment done: scale={s:.4f}")
    return aligned_gt_poses, pred_poses

def to_homo(w2c):
    N = w2c.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]]), (N, 1, 1))
    return np.concatenate([w2c, bottom], axis=1)


def normalize_to_first_pose(Tcw_list):

    Tcw_0 = Tcw_list[0]
    Twc_0 = np.linalg.inv(Tcw_0)
    Tcw_new = [Tcw @ Twc_0 for Tcw in Tcw_list]
    return Tcw_new

import open3d as o3d

def create_camera_mesh(scale=0.1, color=[1, 0, 0]):

    vertices = np.array([
        [0, 0, 0],       
        [1, 1, 2],
        [1, -1, 2],
        [-1, -1, 2],
        [-1, 1, 2],
    ]) * scale

    triangles = np.array([
        [0,1,2],
        [0,2,3],
        [0,3,4],
        [0,4,1],
        [1,2,3],
        [1,3,4]
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    return mesh

def export_cameras_as_mesh(w2c_list, SAVE_ROOT, save_name="cams.ply", color=[1, 0, 0], scale=0.007):
    
    os.makedirs(SAVE_ROOT, exist_ok=True)
    all_meshes = []
    for T in w2c_list:
        
        R = T[:3, :3]
        t = T[:3, 3]

      
        cam_mesh = create_camera_mesh(scale, color)

        
        cam_mesh.rotate(R.T, center=(0, 0, 0))
        cam_mesh.translate(-R.T @ t)       

        all_meshes.append(cam_mesh)

   
    merged = all_meshes[0]
    for mesh in all_meshes[1:]:
        merged += mesh

    out_file = os.path.join(SAVE_ROOT, save_name)

    o3d.io.write_triangle_mesh(out_file, merged)
    print(f"✅ Saved camera mesh: {out_file}")