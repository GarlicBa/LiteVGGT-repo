import os
import cv2
import numpy as np
import os.path as osp
from collections import deque
from base import BaseStereoViewDataset
import dataset_utils.cropping as cropping
from vggt.utils.eval_utils import imread_cv2, shuffle_deque


class SevenScenes(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        if self.tuple_list is not None:
            # Use pre-defined simplerecon scene_ids
            self.scene_list = [
                "stairs/seq-06",
                "stairs/seq-02",
                "pumpkin/seq-06",
                "chess/seq-01",
                "heads/seq-02",
                "fire/seq-02",
                "office/seq-03",
                "pumpkin/seq-03",
                "redkitchen/seq-07",
                "chess/seq-02",
                "office/seq-01",
                "redkitchen/seq-01",
                "fire/seq-01",
            ]
            print(f"Found {len(self.scene_list)} sequences in split {self.split}")
            return

        scenes = os.listdir(base_dir)

        file_split = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}[self.split]

        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()

                for seq_id in seq_ids:
                    # seq is string, take the int part and make it 01, 02, 03
                    # seq_id = 'seq-{:2d}'.format(int(seq_id))
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_id = f"seq-{num_part.zfill(2)}"
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")

        print(f"SevenScenes Found {len(self.scene_list)} sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if "color" in name])
            img_idxs = [f"{i:06d}" for i in range(num_files)]
            img_idxs = img_idxs[:: self.kf_every]

            
            if len(img_idxs) % 8 != 0:
                n = len(img_idxs) - (len(img_idxs) % 8)
                img_idxs = img_idxs[:n]

        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.color.png")
            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")
            posepath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.pose.txt")

            rgb_image = imread_cv2(impath)

            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0

            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (518, 392), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )
         

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7scenes",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )
        return views


class NRGBD(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        scenes = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if self.test_id is not None:
            self.scene_list = [self.test_id]

        else:
            self.scene_list = scenes

        print(f"NRGBD Found {len(self.scene_list)} sequences in split {self.split}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(os.listdir(os.path.join(self.ROOT, scene_id, "images")))
            img_idxs = [f"{i}" for i in range(num_files)]
            img_idxs = img_idxs[:: min(self.kf_every, len(img_idxs) // 2)]

            ## 裁剪为8的倍数
            if len(img_idxs) % 8 != 0:
                n = len(img_idxs) - (len(img_idxs) % 8)
                img_idxs = img_idxs[:n]
                
        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        posepath = osp.join(self.ROOT, scene_id, f"poses.txt")
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)
        views = []



        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, "images", f"img{im_idx}.png")
            depthpath = osp.join(self.ROOT, scene_id, "depth", f"depth{im_idx}.png")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0
            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (518, 392), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )
            

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="nrgbd",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


import os
import struct
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images,load_image_file_crop

class TnTDataset(Dataset):

    def __init__(self, root_dir, colmap_dir, scene_name="advanced__Auditorium",kf_every=1):
        scene_name_ori = scene_name
        self.kf_every = kf_every
        # print(f"key frame select:{self.kf_every}")
        level, scene_name = scene_name.split("__")

        self.scene_dir = os.path.join(root_dir, f"{level}", f"{scene_name}")
        self.scene_name = scene_name
        self.test_samples = []

        bin_path = os.path.join(colmap_dir, scene_name_ori, "0", "images.bin")
        # bin_path = os.path.join(self.scene_dir,"colmap","sparse","0","images.bin")
        # c2w 4×4
        self.poses = read_images_bin(bin_path)

        print(f"pose len:{len(self.poses)}")

        for img in self.poses.keys():
            img = os.path.basename(img)
            # print(img)

            name, ext = os.path.splitext(img)
            digits = ''.join([c for c in name if c.isdigit()])
            if len(digits) == 5:
                new_name = name.replace(digits, digits.zfill(6)) + ext
            else:
                new_name = img

            img_path = os.path.join(self.scene_dir, new_name)
            if not os.path.exists(img_path):
                continue
            self.test_samples.append(img_path)

        self.all_samples = self.test_samples
        print(f"num:{len(self.all_samples)}")


    def __len__(self):
        return len(self.all_samples) // self.kf_every
    
    def __getitem__(self, idx):
        return self._load_sample(self.all_samples[idx])
    

    def _load_sample(self, rgb_path):
        img_name = os.path.basename(rgb_path)
        if self.scene_name == "Caterpillar":
            color = load_image_file_crop(rgb_path)
            color = torch.from_numpy(np.transpose(color, (2, 0, 1)))
            pose = torch.from_numpy(self.poses[os.path.join("rgb",img_name)]).float()
        else:
            if img_name not in self.poses:
                name, ext = os.path.splitext(img_name)
                digits = ''.join([c for c in name if c.isdigit()])

                if len(digits) == 6:
                    alt_name = digits[1:] + ext  
                elif len(digits) == 5:
                    alt_name = digits.zfill(6) + ext
                else:
                    alt_name = img_name

                if alt_name in self.poses:
                    img_name = alt_name
                else:
                    raise KeyError(f"Pose not found for {img_name} or {alt_name}")
         
            # H W 3 numpy -> 3 H W tensor            
            color = load_image_file_crop(rgb_path)
            color = torch.from_numpy(np.transpose(color, (2, 0, 1)))

            pose = torch.from_numpy(self.poses[img_name]).float()

        return dict(
            img=color,
            camera_pose=pose,  # cam2world
            dataset="tnt",
            true_shape=torch.tensor([392, 518]),
            label=img_name,
            instance=img_name,
        )



def read_images_bin(bin_path: str | Path):
    bin_path = Path(bin_path)
    poses = {}

    with bin_path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]  # uint64
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.frombuffer(f.read(8 * 4), dtype=np.float64)  # qw,qx,qy,qz
            tvec = np.frombuffer(f.read(8 * 3), dtype=np.float64)  # tx,ty,tz
            cam_id = struct.unpack("<I", f.read(4))[0]  # camera_id

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8").split("/")[-1]  

            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, 1)

            # world→cam to cam→world
            qw, qx, qy, qz = qvec
            R_wc = np.array(
                [
                    [
                        1 - 2 * qy * qy - 2 * qz * qz,
                        2 * qx * qy + 2 * qz * qw,
                        2 * qx * qz - 2 * qy * qw,
                    ],
                    [
                        2 * qx * qy - 2 * qz * qw,
                        1 - 2 * qx * qx - 2 * qz * qz,
                        2 * qy * qz + 2 * qx * qw,
                    ],
                    [
                        2 * qx * qz + 2 * qy * qw,
                        2 * qy * qz - 2 * qx * qw,
                        1 - 2 * qx * qx - 2 * qy * qy,
                    ],
                ]
            )
            t_wc = -R_wc @ tvec
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_wc.astype(np.float32)
            c2w[:3, 3] = t_wc.astype(np.float32)

            poses[name] = c2w
    return poses


import glob
import os
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images


class Mip360Dataset(Dataset):
    def __init__(self, root_dir, scene_name="bicycle",kf_every=3):
        
        self.kf_every = kf_every
        self.scene_dir = os.path.join(
            root_dir,
            f"{scene_name}",
        )

        self.test_samples = sorted(
            glob.glob(os.path.join(self.scene_dir, "images_8", "*.JPG"))
        )
        # self.train_samples = sorted(glob.glob(os.path.join(self.train_seqs, "rgb", "*.png")))
        self.all_samples = self.test_samples  # + self.train_samples
        bin_path = os.path.join(self.scene_dir, "sparse", "0", "images.bin")
        self.poses = read_images_bin(bin_path)

    def __len__(self):
        return len(self.all_samples) // self.kf_every
    
    def __getitem__(self, idx):
        sample_idx = idx * self.kf_every
        return self._load_sample(self.all_samples[sample_idx])

    def _load_sample(self, rgb_path):
        img_name = os.path.basename(rgb_path)
                # H W 3 numpy -> 3 H W tensor 
        color = load_image_file_crop(rgb_path)
        color = torch.from_numpy(np.transpose(color, (2, 0, 1)))
        pose = torch.from_numpy(self.poses[img_name]).float()

        return dict(
            img=color,
            camera_pose=pose,  # cam2world
            dataset="mip360",
            true_shape=torch.tensor([392, 518]),
            label=img_name,
            instance=img_name,
        )


def read_images_bin(bin_path: str | Path):
    bin_path = Path(bin_path)
    poses = {}

    with bin_path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]  # uint64
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.frombuffer(f.read(8 * 4), dtype=np.float64)  # qw,qx,qy,qz
            tvec = np.frombuffer(f.read(8 * 3), dtype=np.float64)  # tx,ty,tz
            cam_id = struct.unpack("<I", f.read(4))[0]  # camera_id

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, 1)

            # world→cam to cam→world
            qw, qx, qy, qz = qvec
            R_wc = np.array(
                [
                    [
                        1 - 2 * qy * qy - 2 * qz * qz,
                        2 * qx * qy + 2 * qz * qw,
                        2 * qx * qz - 2 * qy * qw,
                    ],
                    [
                        2 * qx * qy - 2 * qz * qw,
                        1 - 2 * qx * qx - 2 * qz * qz,
                        2 * qy * qz + 2 * qx * qw,
                    ],
                    [
                        2 * qx * qz + 2 * qy * qw,
                        2 * qy * qz - 2 * qx * qw,
                        1 - 2 * qx * qx - 2 * qy * qy,
                    ],
                ]
            )
            t_wc = -R_wc @ tvec
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_wc.astype(np.float32)
            c2w[:3, 3] = t_wc.astype(np.float32)

            poses[name] = c2w

    return poses

class DTUDataset(Dataset):
    def __init__(self, root_dir, scene_name=None):

        self.root_dir = root_dir

        if scene_name is None:

            self.scenes = [
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
            self.scenes.sort()
        else:

            if not os.path.isdir(os.path.join(root_dir, scene_name)):
                raise ValueError(f"Scene {scene_name} does not exist!")
            self.scenes = [scene_name]

        print(f"Loaded {len(self.scenes)} scenes:")
        print(self.scenes)


    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):

        scene = self.scenes[idx]

        scene_dir = os.path.join(self.root_dir, scene, "images")

        image_paths = sorted([
            os.path.join(scene_dir, f)
            for f in os.listdir(scene_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])

        cam_dir = os.path.join(self.root_dir, scene, "cams")

        colors = []
        poses = []

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            cam_name = os.path.splitext(img_name)[0] + "_cam.txt"
            cam_path = os.path.join(cam_dir, cam_name)

            color = load_image_file_crop(img_path)
            color = torch.from_numpy(np.transpose(color, (2, 0, 1)))

            pose = torch.from_numpy(read_extrinsic(cam_path))

            colors.append(color)
            poses.append(pose)

        poses = torch.stack(poses, dim=0)
        colors = torch.stack(colors, dim=0)

        return dict(
            imgs=colors,
            poses=poses,
            scene=scene
        )





import numpy as np

def read_extrinsic(cam_path):

    with open(cam_path, "r") as f:
        lines = f.readlines()

    start = [i for i, l in enumerate(lines) if "extrinsic" in l][0] + 1
    extrinsic = np.array(
        [list(map(float, lines[start + i].split())) for i in range(4)],
        dtype=np.float32
    )

    return extrinsic
