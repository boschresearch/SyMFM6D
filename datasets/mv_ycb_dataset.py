#!/usr/bin/env python3
from math import inf
import os
from re import sub
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
from cv2 import imshow, waitKey
import cv2
import OpenEXR
import array
import yaml
from models.RandLA.helper_tool import DataProcessing as DP
import matplotlib
import matplotlib.pyplot as plt
import normalSpeed
import pandas as pd
import utils.symmetry_transforms as symmetry

from datasets.dataset_base_class import DatasetBase


class MvYcbDataset(DatasetBase):
    def __init__(self, dataset_mode='train', variant='SymMovCam', sift_fps=False, multi_view=False, noisy_cam_pose=True,
                 set_views=None, symmetries=False, rotationals=True, n_rot_sym=16, sym_classes=None, new=False,
                 train_txt='image_sets/train.txt', train_val_split=0.9, real_ds=None):
        print('Using: ', variant)
        config = self.config = Config(ds_name=variant)
        
        super().__init__(config, dataset_mode, multi_view, set_views)

        bs_utils = self.bs_utils = Basic_Utils(config)
        if sym_classes:
            config.sym_cls_ids = sym_classes
        self.variant = variant
        self.sift_fps = sift_fps

        if real_ds is not None:
            self.real_lst = real_ds.real_lst
            self.real_ds_root = real_ds.root

        self.camera_intrinsics = np.asarray([[497.77777777777777, 0., 320.],
                                             [0., 497.77777777777777, 240.],
                                             [0., 0., 1.]])

        if symmetries and variant == 'SymMovCam':
            if new:
                self.symmetries = pd.read_csv(config.symmetries_new)
            else:
                self.symmetries = pd.read_csv(config.symmetries)
            self.provide_symmetry = True
            self.rotationals = rotationals
            self.n_rot_sym = n_rot_sym
            print('using symmetry')
        else:
            assert not symmetries, "This dataset has no symmetric objects!"
            self.provide_symmetry = False
            self.rotationals = False
            self.n_rot_sym = 0

        self.cam_scale = 1
        self.n_scenes = config.n_scenes
        if set_views:
            self.n_frames = set_views
        else:
            self.n_frames = 4 if variant == 'MovCam' or variant == 'SymMovCam' else 3
        self.train_val_split = train_val_split  # 0.9

        self.noisy_cam_pose = noisy_cam_pose
        if self.noisy_cam_pose:
            print('using nosy cam poses')

        self.n_sample_points = config.n_sample_points

        self.diameters = {}
        self.cls_lst = bs_utils.read_lines(config.cls_lst_p)
        self.obj_dict = {}
        self.obj_dict_without_numbers = {}
        self.root = config.root
        self.sym_cls_ids = []
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
            if variant == 'SymMovCam':
                self.obj_dict_without_numbers[cls[4:]] = cls_id
        self.rng = np.random
        if dataset_mode == 'train':
            self.add_noise = True
            self.add_depth_noise = False
            self.add_color_jitter = False
            self.path = os.path.join(self.root, train_txt)
            self.all_lst = bs_utils.read_lines(self.path)
            if self.multi_view:
                self.sequence_samples = self.generate_sequence_samples(0, int(self.n_scenes * self.train_val_split))
            self.n_batches_per_epoch = len(self) // config.mini_batch_size
        else:
            self.pp_data = None

            self.path = os.path.join(self.root, 'image_sets/val.txt')
            self.all_lst = bs_utils.read_lines(self.path)
            if self.multi_view:
                self.sequence_samples = self.generate_sequence_samples(int(self.n_scenes * self.train_val_split), self.n_scenes)
        print(f"{variant} {dataset_mode} dataset_size: ", len(self))

    def get_normal(self, cld):
        cloud = pcl.PointCloud()
        cld = cld.astype(np.float32)
        cloud.from_array(cld)
        ne = cloud.make_NormalEstimation()
        kdtree = cloud.make_kdtree()
        ne.set_SearchMethod(kdtree)
        ne.set_KSearch(10)
        n = ne.compute()

        n = n.to_array()
        return n

    def rotate(self, x, i):
        return x[i:] + x[:i]

    def all_rotations(self, x):
        rotations = []
        for i in range(len(x)):
            rotations.append(self.rotate(x, i))
        return rotations

    def generate_sequence_samples(self, start, end):
        sequence_samples = []
        for scene_idx in range(start, end):
            scene = []
            for frame_idx in range(self.n_frames):
                scene.append(f"sequence{(scene_idx + 1):09}/TYPE{(frame_idx + 50):04}")
            sequence_samples.append(scene)
            scene = scene[1:] + scene[:1]
            sequence_samples.append(scene)
            scene = scene[1:] + scene[:1]
            sequence_samples.append(scene)
        return sequence_samples

    def get_labels(self, label_path):
        """
        Load (semantic) segmentation map and parse it to a usable class label map
        """
        labels = np.asarray(Image.open(os.path.join(self.root, "data", label_path)))
        return labels

    def get_pose_gt_info(self, cld, labels, poses):
        item_dict = {}
        config = self.config
        bs_utils = self.bs_utils
        RTs = np.zeros((config.n_objects, 3, 4), dtype=np.float32)
        cls_ids = np.zeros((config.n_objects, 1), dtype=np.float32)

        if self.provide_symmetry:
            max_refl_sym = 2
            max_discrete_rots = self.n_rot_sym if self.rotationals else 0
            max_rot_sym = 1 * max(0, (max_discrete_rots - 1))
            max_sym = max_refl_sym + max_rot_sym
            kp_3ds = np.ones((config.n_objects, max_sym + 1, config.n_keypoints, 3), dtype=np.float32) * np.nan
            ctr_3ds = np.zeros((config.n_objects, 1, 3), dtype=np.float32)
            kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3), dtype=np.float32)
            kp_targ_ofst_sym = np.zeros((config.n_sample_points, config.n_keypoints, 3), dtype=np.float32)
            kp_targ_ofst_sym_kp = np.zeros((config.n_sample_points, max_sym + 1, config.n_keypoints, 3), dtype=np.float32)
            ctr_targ_ofst = np.zeros((config.n_sample_points, 3), dtype=np.float32)
            ctr_targ_ofst_sym = np.zeros((config.n_sample_points, 3), dtype=np.float32)
            ctr_targ_ofst_sym_kp = np.zeros((config.n_sample_points, 1, 3), dtype=np.float32)
        else:
            kp_3ds = np.zeros((config.n_objects, config.n_keypoints, 3), dtype=np.float32)
            ctr_3ds = np.zeros((config.n_objects, 3), dtype=np.float32)
            kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3), dtype=np.float32)
            ctr_targ_ofst = np.zeros((config.n_sample_points, 3), dtype=np.float32)

        visible_objects = dict()

        for obj_name, vals in poses.items():
            if obj_name == "cam":
                continue

            if vals["in_frame"]:
                visible_objects[obj_name] = vals

        for i, (obj_name, vals) in enumerate(poses.items()):
            if obj_name == "cam":
                continue

            obj_name = obj_name[:-4]

            if self.provide_symmetry:
                obj_symmetries = self.symmetries[self.symmetries['obj'] == obj_name]

            target_pose = np.array(vals['matrix'])[:-1]
            target_pose[1, :] *= -1

            r = target_pose[:3, :3]
            t = np.expand_dims(target_pose[:3, 3], axis=-1)

            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            if self.variant == 'SymMovCam':
                cls_id = self.obj_dict_without_numbers[obj_name]
                assert obj_name == self.cls_lst[cls_id - 1][4:]
            else:
                cls_id = self.obj_dict[obj_name]
                assert obj_name == self.cls_lst[cls_id - 1]

            ctr = bs_utils.get_ctr(obj_name, ds_type=config.dataset_name).copy()[:, None]
            mesh_ctr = ctr.copy()
            ctr = np.dot(ctr.T, r.T) + t[:, 0]

            if self.provide_symmetry:
                refl_sym = np.stack(
                    [np.stack([float(obj_symmetries[f'refl{s + 1}_{dim_}']) for dim_ in ['x', 'y', 'z']], axis=0) for s
                     in range(3)], axis=0).squeeze()
                rot_sym = np.stack(
                    [np.stack([float(obj_symmetries[f'rot{s + 1}_{dim_}']) for dim_ in ['x', 'y', 'z']], axis=0) for s
                     in range(3)], axis=0).squeeze()
                n_rot = np.stack([float(obj_symmetries[f'n_rot{s + 1}']) for s in range(3)], axis=0).squeeze()
                sym_ctr = np.stack([float(obj_symmetries[f'ctr_{dim_}']) for dim_ in ['x', 'y', 'z']], axis=0).squeeze()

            msk_idx = np.where(labels == cls_id)[0]
            cls_ids[i, :] = np.array([cls_id])

            if self.provide_symmetry:
                ctr_3ds[i, 0, :] = ctr[0]
                target_offset = np.array(np.add(cld, -1.0 * ctr_3ds[i, 0, :]))
                ctr_targ_ofst_sym[msk_idx, :] = target_offset[msk_idx, :]
                ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
                ctr_targ_ofst_sym_kp[msk_idx, 0, :] = target_offset[msk_idx, :]
            else:
                ctr_3ds[i, :] = ctr[0]
                target_offset = np.array(np.add(cld, -1.0 * ctr_3ds[i, :]))
                ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]

            if config.n_keypoints == 8:
                if self.variant == 'SymMovCam':
                    kp_type = '8_kps'
                elif self.sift_fps:
                    kp_type = 'sift_fps_kps'
                else:
                    kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(config.n_keypoints)

            kps = bs_utils.get_kps(obj_name, kp_type=kp_type, ds_type=self.variant).copy()
            mesh_kps = kps.copy()
            if self.provide_symmetry:
                kps = np.dot(kps, r.T) + t[:, 0]
                kp_3ds[i, 0] = kps
                for si, s in enumerate(refl_sym):
                    if np.isnan(s).any() or cls_id not in self.config.sym_cls_ids:
                        continue
                    symm_kps = symmetry.mirror_mesh_np(mesh_kps, sym_ctr, s)
                    symm_kps = np.dot(symm_kps, r.T) + t[:, 0]
                    kp_3ds[i, si + 1] = symm_kps

                if self.rotationals:
                    for si, (ns, s) in enumerate(zip(n_rot, rot_sym)):
                        if np.isnan(s).any() or ns <= 0 or cls_id not in self.config.sym_cls_ids:
                            continue
                        v1 = sym_ctr.squeeze() + s
                        v2 = sym_ctr.squeeze() - s

                        if ns == inf:
                            ns = max_discrete_rots

                        for step in range(1, int(ns)):
                            theta = step * (360 / ns)
                            symm_kps = symmetry.rotate_np(mesh_kps, v1, v2, theta).T
                            symm_kps = np.dot(symm_kps, r.T) + t[:, 0]

                            kp_3ds[i, max_refl_sym + si + step] = symm_kps
            else:
                kps = np.dot(kps, r.T) + t[:, 0]
                kp_3ds[i] = kps

            if self.provide_symmetry:
                target = []
                for kp in kp_3ds[i][0]:
                    target.append(np.add(cld, -1.0 * kp))
                target_offset = np.array(target).transpose(1, 0, 2)  # [n_pts, n_kps, c]
                kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

                target_offsets = []
                for ikpg, kps_sym_group in enumerate(kp_3ds[i]):
                    target = []
                    for kp in kps_sym_group:
                        target.append(np.add(cld, -1.0 * kp))
                    target_offsets.append(np.array(target).transpose(1, 0, 2))  # [n_pts, n_kps, c]
                    kp_targ_ofst_sym_kp[msk_idx, ikpg, :, :] = np.array(target).transpose(1, 0, 2)[msk_idx, :, :]

                target_offset = np.stack(target_offsets, axis=1)
                target_offset[np.isnan(target_offset)] = np.inf
                dists = np.sqrt(target_offset[..., 0] ** 2 + target_offset[..., 1] ** 2 + target_offset[..., 2] ** 2)
                min_dists = np.min(dists, axis=1)[:, None, :]
                mask = (dists == min_dists)
                target_offset[~mask, :] = np.inf
                target_offset = np.min(target_offset, axis=1)

                kp_targ_ofst_sym[msk_idx, :, :] = target_offset[msk_idx, :, :]

                item_dict['kp_targ_ofst_sym'] = kp_targ_ofst_sym
                item_dict['kp_targ_ofst_sym_kp'] = kp_targ_ofst_sym_kp
                item_dict['ctr_targ_ofst_sym'] = ctr_targ_ofst_sym
                item_dict['ctr_targ_ofst_sym_kp'] = ctr_targ_ofst_sym_kp

            else:
                target = []
                for kp in kps:
                    target.append(np.add(cld, -1.0 * kp))
                target_offset = np.array(target).transpose(1, 0, 2)  # [n_pts, n_kps, c]
                kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        item_dict['RTs'] = RTs
        item_dict['kp_3ds'] = kp_3ds
        item_dict['ctr_3ds'] = ctr_3ds
        item_dict['cls_ids'] = cls_ids
        item_dict['kp_targ_ofst'] = kp_targ_ofst
        item_dict['ctr_targ_ofst'] = ctr_targ_ofst

        return item_dict  # dict of NumPy arrays

    def get_item(self, item_name):
        """
        :param item_name: e.g. 'sequence000009942/TYPE0051'
        :return: item_dict with all necessary information including RGB image, PCL, camera pose, labels, etc.
        """
        pose_path = item_name.replace('TYPE', 'poses/pose') + ".yaml"
        if self.noisy_cam_pose:
            item_name = f"{item_name[:-4]}{(int(item_name[-4:]) + self.n_frames):04}"

        rgb_path = item_name.replace('TYPE', 'images/frame') + '.' + self.config.rgb_filetype
        depth_path = item_name.replace('TYPE', 'depth/depth') + '.' + self.config.dpt_filetype
        label_path = item_name.replace('TYPE', 'segment/segment') + '.' + self.config.segment_filetype
        labels = self.get_labels(label_path)
        rgb_path = os.path.join(self.root, "data", rgb_path)

        with Image.open(rgb_path) as ri:
            if self.add_color_jitter:
               ri = self.trancolor(ri)
            rgb = np.array(ri)  # [:, :, :3]
            if self.add_noise:
                rgb = self.rgb_add_noise(rgb)

        dpt = np.asarray(Image.open(os.path.join(self.root, "data", depth_path)), np.float32) / 65536.0
        
        # reverse correction
        f_pix = self.camera_intrinsics[0, 0]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]

        pixel_coord = np.mgrid[:dpt.shape[0], :dpt.shape[1]]

        # Distance of every pixel to the center in pixel units
        d_pix = np.sqrt((pixel_coord[1, :, :] - cx) ** 2 + (pixel_coord[0, :, :] - cy) ** 2)
        # arctan gives the ray angle here
        dpt_m = np.divide(dpt, np.cos(np.arctan(d_pix / f_pix)))

        cam_scale = self.cam_scale
        dpt_m = self.bs_utils.fill_missing(dpt_m, cam_scale, 1)
        if self.add_depth_noise:
            dpt_m = self.depth_img_add_noise(img=dpt_m)

        msk_dp = dpt_m > 1e-6

        # add real RGB-D background as it is done for YCB-Video
        if self.config.add_real_back and self.config.add_real_back >= self.rng.rand():
            rgb, dpt_um = self.add_real_back(rgb, labels, dpt_m * 10000, msk_dp, name)
            dpt_m = dpt_um / 10000  # undo the conversion from the YCB-Video depth scale into meters.

        K = self.camera_intrinsics
        nrm_map = normalSpeed.depth_normal(
            (dpt_m * 1000).astype(np.uint16), K[0][0], K[1][1], 5, 2000, 20, False
        )

        dpt = dpt_m.astype(np.float32) / cam_scale

        dpt_xyz = self.dpt_2_pcld(dpt, 1.0, self.camera_intrinsics)
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > self.config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.config.n_sample_points - len(choose_2)), 'wrap')

        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]

        rgb_labels = labels.copy()
        labels = labels.flatten()[choose]
        choose = np.array([choose])

        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1)
        source_cld = cld.copy()

        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        with open(os.path.join(self.root, "data", pose_path)) as poses_file:
            poses = yaml.load(poses_file, Loader=yaml.SafeLoader)
        cam_pose = np.array(poses['cam'], np.float32)
        cam_pose[:, 1:3] *= -1
        cam_pose = np.linalg.inv(cam_pose)

        item_dict = self.get_pose_gt_info(cld, labels, poses)

        if not self.multi_view:
            h, w = rgb_labels.shape
            xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
            msk_lst = [dpt_xyz[2, :, :] > 1e-8]

            for i in range(3):
                scale = pow(2, i + 1)
                nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
                ys, xs = np.mgrid[:nh, :nw]
                xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])
                msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
            sr2dptxyz = {
                pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
            }
            sr2msk = {
                pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
            }

            rgb_ds_sr = [4, 8, 8, 8]
            n_ds_layers = 4
            pcld_sub_s_r = [4, 4, 4, 4]

            # DownSample stage
            for i in range(n_ds_layers):
                nei_idx = DP.knn_search(
                    cld[None, ...], cld[None, ...], 16
                ).astype(np.int32).squeeze(0)
                sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
                pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
                up_i = DP.knn_search(
                    sub_pts[None, ...], cld[None, ...], 1
                ).astype(np.int32).squeeze(0)
                item_dict['cld_xyz%d' % i] = cld.astype(np.float32).copy()
                item_dict['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
                item_dict['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
                item_dict['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
                nei_r2p = DP.knn_search(
                    sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
                ).astype(np.int32).squeeze(0)
                item_dict['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
                nei_p2r = DP.knn_search(
                    sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
                ).astype(np.int32).squeeze(0)
                item_dict['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
                cld = sub_pts

            n_up_layers = 3
            rgb_up_sr = [4, 2, 2]
            for i in range(n_up_layers):
                r2p_nei = DP.knn_search(
                    sr2dptxyz[rgb_up_sr[i]][None, ...],
                    item_dict['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...], 16
                ).astype(np.int32).squeeze(0)
                item_dict['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
                p2r_nei = DP.knn_search(
                    item_dict['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...],
                    sr2dptxyz[rgb_up_sr[i]][None, ...], 1
                ).astype(np.int32).squeeze(0)
                item_dict['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        # convert numpy arrays to torch tensors
        for key in item_dict.keys():
            item_dict[key] = torch.from_numpy(item_dict[key])
        item_dict['rgb'] = torch.from_numpy(rgb.astype(np.float32))
        item_dict['cld_rgb_nrm'] = torch.from_numpy(cld_rgb_nrm.astype(np.float32))
        item_dict['cld'] = torch.from_numpy(source_cld.astype(np.float32))
        item_dict['dpt_xyz'] = torch.from_numpy(dpt_xyz.astype(np.float32))
        item_dict['choose'] = torch.LongTensor(choose.astype(np.int32))
        item_dict['labels'] = torch.LongTensor(labels.astype(np.int32))
        item_dict['rgb_labels'] = torch.LongTensor(rgb_labels.astype(np.int32))
        item_dict['camera_intrinsics'] = torch.from_numpy(self.camera_intrinsics.astype(np.float32))
        item_dict['cam_pose'] = torch.from_numpy(cam_pose.astype(np.float32))

        return item_dict

    def cut_or_pad_mask(self, choose_2):
        if len(choose_2) > self.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.n_sample_points - len(choose_2)), 'wrap')
        return choose_2

    def transform_sample(self, target_sample, sample, i):
        """
        Transforms a point cloud from camera frame cam0 to camera frame cam1
        cam0 and cam1 are actually expected to be the inverse of the camera position, following YCB's convention
        """
        # keys to transform: cld_rgb_nrm, kp_targ_ofst, ctr_targ_ofst
        # input keys to transform: cld_xyz0, cld_xyz1, cld_xyz2, cld_xyz3
        camchange = (target_sample['cam_pose'] @ torch.inverse(sample['cam_pose'])).float()

        new_cloud = self.transform_data(camchange, sample['cld_rgb_nrm'][..., :3])
        sample['cld'] = self.transform_data(camchange, sample['cld'])
        sample['dpt_xyz'] = self.transform_data(camchange, sample['dpt_xyz'])

        new_normals = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3], new_cloud, sample['cld_rgb_nrm'][..., -3:])
        new_cld_rgb_norm = torch.cat([new_cloud, sample['cld_rgb_nrm'][..., 3:6], new_normals], dim=1)

        sample['kp_targ_ofst'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3].view(-1, 1, 3).repeat(1, 8, 1),
                                                        new_cloud.view(-1, 1, 3).repeat(1, 8, 1), sample['kp_targ_ofst'])
        sample['ctr_targ_ofst'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3], new_cloud, sample['ctr_targ_ofst'])

        if self.provide_symmetry:
            sample['kp_targ_ofst_sym'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3].view(-1, 1, 3).repeat(1, 8, 1),
                                                                new_cloud.view(-1, 1, 3).repeat(1, 8, 1), sample['kp_targ_ofst_sym'])
            sample['ctr_targ_ofst_sym'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3], new_cloud, sample['ctr_targ_ofst_sym'])

            sample['kp_targ_ofst_sym_kp'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3].view(-1, 1, 3).repeat(1, 8, 1).unsqueeze(1),
                                                                new_cloud.view(-1, 1, 3).repeat(1, 8, 1).unsqueeze(1), sample['kp_targ_ofst_sym_kp'])
            sample['ctr_targ_ofst_sym_kp'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3].unsqueeze(1), new_cloud.unsqueeze(1), sample['ctr_targ_ofst_sym_kp'])

        sample['cld_rgb_nrm'] = new_cld_rgb_norm

        return sample

    def transform_data(self, camchange, data):
        ones = torch.ones(*data.shape[:-1], 1)
        if data.dim() > 2:
            ones_flatt = ones.view(-1, 1)
            data_flatt = data.view(-1, 3)
            new_data_flatt = camchange @ torch.cat([data_flatt, ones_flatt], dim=-1).T
            new_data_flatt = new_data_flatt[:3, :].T
            new_data = new_data_flatt.view(data.shape)
        else:
            new_data = camchange @ torch.cat([data, ones], dim=-1).T
            new_data = new_data[:3, :].T
        return new_data

    def fuse_sequence(self, sequence):
        views = len(sequence)
        item_dict = {}
        item_dict['rgb'] = torch.stack([view['rgb'] for view in sequence], dim=0)

        item_dict['cld_rgb_nrm'] = torch.cat([view['cld_rgb_nrm'] for view in sequence], dim=0)
        item_dict['cld'] = torch.cat([view['cld'] for view in sequence], dim=0)

        item_dict['kp_targ_ofst'] = torch.cat([view['kp_targ_ofst'] for view in sequence], dim=0)
        item_dict['ctr_targ_ofst'] = torch.cat([view['ctr_targ_ofst'] for view in sequence], dim=0)

        if self.provide_symmetry:
            item_dict['kp_targ_ofst_sym'] = torch.cat([view['kp_targ_ofst'] for view in sequence], dim=0)
            item_dict['ctr_targ_ofst_sym'] = torch.cat([view['ctr_targ_ofst'] for view in sequence], dim=0)

            item_dict['kp_targ_ofst_sym_kp'] = torch.cat([view['kp_targ_ofst_sym_kp'] for view in sequence], dim=0)
            item_dict['ctr_targ_ofst_sym_kp'] = torch.cat([view['ctr_targ_ofst_sym_kp'] for view in sequence], dim=0)

        item_dict['labels'] = torch.cat([view['labels'] for view in sequence], dim=0)

        item_dict['choose'] = torch.stack([view['choose'] for view in sequence], dim=0)  # views 1 pts
        item_dict['rgb_labels'] = torch.stack([view['rgb_labels'] for view in sequence], dim=0)

        item_dict['cls_ids'] = sequence[0]['cls_ids']
        item_dict['RTs'] = sequence[0]['RTs']
        item_dict['kp_3ds'] = sequence[0]['kp_3ds']
        item_dict['ctr_3ds'] = sequence[0]['ctr_3ds']
        item_dict['cam_pose'] = sequence[0]['cam_pose']
        item_dict['camera_intrinsics'] = sequence[0]['camera_intrinsics']

        h, w = item_dict['rgb_labels'][0].shape
        item_dict['dpt_xyz'] = torch.stack([view['dpt_xyz'] for view in sequence], dim=0)
        inputs = {}
        sr2dptxyz_lst = []
        for s in sequence:
            dpt_xyz = s['dpt_xyz'].numpy()
            xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w

            for i in range(3):
                scale = pow(2, i + 1)
                nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
                ys, xs = np.mgrid[:nh, :nw]
                xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])
            sr2dptxyz = {
                pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
            }
            sr2dptxyz_lst.append(sr2dptxyz)

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]

        cld = item_dict['cld'].numpy()
        cld_views = [view['cld'].numpy() for view in sequence]
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[::pcld_sub_s_r[i], :]
            pool_i = nei_idx[::pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()

            inputs[f'p2r_ds_nei_idx{i}'] = []
            inputs[f'r2p_ds_nei_idx{i}'] = []
            for v, sr2dptxyz in enumerate(sr2dptxyz_lst):
                cld_view = cld_views[v]
                sub_pts_view = cld_view[::pcld_sub_s_r[i], :]
                inputs['cld_xyz%d%d' % (i, v)] = cld_views[v].astype(np.float32).copy()
                nei_r2p = DP.knn_search(
                    sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts_view[None, ...], 16
                ).astype(np.int32).squeeze(0)
                inputs[f'r2p_ds_nei_idx{i}'].append(nei_r2p.copy())

                nei_p2r = DP.knn_search(
                    sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
                ).astype(np.int32).squeeze(0)
                inputs[f'p2r_ds_nei_idx{i}'].append(nei_p2r.copy())
                cld_views[v] = sub_pts_view
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            inputs[f'r2p_up_nei_idx{i}'] = []
            inputs[f'p2r_up_nei_idx{i}'] = []
            for v, sr2dptxyz in enumerate(sr2dptxyz_lst):
                r2p_nei = DP.knn_search(
                    sr2dptxyz[rgb_up_sr[i]][None, ...],
                    inputs[f'cld_xyz{(n_ds_layers - i - 1)}{v}'][None, ...], 16
                ).astype(np.int32).squeeze(0)
                inputs[f'r2p_up_nei_idx{i}'].append(r2p_nei.copy())
                p2r_nei = DP.knn_search(
                    inputs[f'cld_xyz{(n_ds_layers - i - 1)}'][None, ...],
                    sr2dptxyz[rgb_up_sr[i]][None, ...], 1
                ).astype(np.int32).squeeze(0)
                inputs[f'p2r_up_nei_idx{i}'].append(p2r_nei.copy())

        for k in inputs.keys():
            if k.startswith('r2p') or k.startswith('p2r'):
                inputs[k] = np.stack([v for v in inputs[k]], axis=0)
        inputs = {key: torch.from_numpy(value) for key, value in zip(inputs.keys(), inputs.values())}

        item_dict.update(inputs)

        return item_dict

    def __getitem__(self, idx):
        config = self.config
        bs_utils = self.bs_utils
        if self.dataset_mode == 'train':
            if self.multi_view:
                sequence = self.get_sequence(self.sequence_samples[idx])
                for i in range(1, len(sequence)):
                    sequence[i] = self.transform_sample(sequence[0], sequence[i], i)

                fused_sample = self.fuse_sequence(sequence)
                fused_sample['cld_rgb_nrm'] = np.swapaxes(fused_sample['cld_rgb_nrm'], 0, 1)
                return fused_sample

            item_name = self.all_lst[idx]

            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            data['cld_rgb_nrm'] = np.swapaxes(data['cld_rgb_nrm'], 0, 1)
            return data
        else:
            item_name = self.all_lst[idx]
            data = self.get_item(item_name)
            if self.multi_view:
                sequence = self.get_sequence(self.sequence_samples[idx])
                for i in range(1, len(sequence)):
                    sequence[i] = self.transform_sample(sequence[0], sequence[i], i)

                fused_sample = self.fuse_sequence(sequence)
                fused_sample['cld_rgb_nrm'] = np.swapaxes(fused_sample['cld_rgb_nrm'], 0, 1)
                return fused_sample
            data['cld_rgb_nrm'] = np.swapaxes(data['cld_rgb_nrm'], 0, 1)
            return data
