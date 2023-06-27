#!/usr/bin/env python3
from audioop import mul
import os
from random import uniform
import cv2
import torch
import numpy as np
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
import pandas as pd
import utils.symmetry_transforms as symmetry

from datasets.dataset_base_class import DatasetBase


class YcbDataset(DatasetBase):
    def __init__(self, dataset_mode='train', multi_view=False, set_views=1, uniform_pre=False, symmetries=False,
                 n_rot_sym=4, syn_train_data_ratio=0.0):
        config = Config(ds_name='ycb')
        super().__init__(config, dataset_mode, multi_view, set_views, syn_train_data_ratio)

        self.bs_utils = Basic_Utils(config)
        self.uniform_pre = uniform_pre
        self.diameters = {}
        self.cls_lst = self.bs_utils.read_lines(config.cls_lst_p)
        self.obj_dict = {}
        self.obj_dict_id = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
            self.obj_dict_id[cls_id] = cls
        self.rng = np.random
        self.root = config.root
        self.real_ds_root = self.root
        if dataset_mode == 'train':
            self.add_noise = True
            if self.multi_view:
                self.path = os.path.join(config.exp_dir, 'datasets/ycb/dataset_config/train_data_list_real_only.txt')  # contains only real YCB-Video data and no synthetic frames
                self.all_lst = self.bs_utils.read_lines(self.path)
                self.sequences = {i.split('/')[1]: [] for i in self.all_lst}
                for s in self.all_lst:
                    self.sequences[s.split('/')[1]].append(s.split('/')[2])

                self.n_batches_per_epoch = len(self.all_lst) // config.mini_batch_size
                self.real_lst = self.all_lst
                self.sequence_samples = self.generate_sequence_samples(self.sequences)  # train sequences with real data
            else:
                self.path = os.path.join(config.exp_dir, 'datasets/ycb/dataset_config/train_data_list.txt')
                self.all_lst = self.bs_utils.read_lines(self.path)
                self.n_batches_per_epoch = len(self.all_lst) // config.mini_batch_size
                self.real_lst = []
                self.syn_lst = []
                for item in self.all_lst:
                    if item[:5] == 'data/':
                        self.real_lst.append(item)
                    else:
                        self.syn_lst.append(item)
        else:
            if self.multi_view:
                self.pp_data = None
                self.path = os.path.join(config.exp_dir, 'datasets/ycb/dataset_config/test_data_list.txt')
                self.all_lst = self.bs_utils.read_lines(self.path)
                self.sequences = {i.split('/')[1]: [] for i in self.all_lst}
                for s in self.all_lst:
                    self.sequences[s.split('/')[1]].append(s.split('/')[2])

                self.sequence_samples = self.generate_sequence_samples(self.sequences)
            else:
                self.pp_data = None
                self.path = os.path.join(config.exp_dir, '/datasets/ycb/dataset_config/test_data_list.txt')
                self.all_lst = self.bs_utils.read_lines(self.path)
        print(f"YCB {dataset_mode} dataset_size: ", len(self.all_lst))

        self.sym_cls_ids = [13, 16, 19, 20, 21]
        if symmetries:
            self.symmetries = pd.read_csv(os.path.join(config.exp_dir, 'datasets/ycb', 'dataset_config/symmetries.txt'))
            self.provide_symmetry = True
            self.rotationals = True
            self.n_rot_sym = n_rot_sym
            print('using symmetry')
        else:
            self.provide_symmetry = False
            self.rotationals = False
            self.n_rot_sym = 0

    def get_item(self, item_name):
        """
        :param item_name: e.g. 'data/0043/000155'
        :return: item_dict with all necessary information including RGB image, PCL, camera pose, labels, etc.
        """
        with Image.open(os.path.join(self.root, item_name + '-depth.png')) as di:
            dpt_um = np.array(di)
        with Image.open(os.path.join(self.root, item_name + '-label.png')) as li:
            labels = np.array(li)
        rgb_labels = labels.copy()
        meta = scio.loadmat(os.path.join(self.root, item_name + '-meta.mat'))
        if item_name[:8] != 'data_syn' and int(item_name[5:9]) >= 60:
            K = self.config.intrinsic_matrix['ycb_K2']
        else:
            K = self.config.intrinsic_matrix['ycb_K1']

        rgb_path = os.path.join(self.root, item_name + '-color.png')

        with Image.open(rgb_path) as ri:
            if self.add_noise:
                ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]
        rnd_typ = 'syn' if 'syn' in item_name else 'real'
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        msk_dp = dpt_um > 1e-6

        if self.add_noise and rnd_typ == 'syn':
            rgb = self.rgb_add_noise(rgb)
            rgb, dpt_um = self.add_real_back(rgb, rgb_labels, dpt_um, msk_dp)

            if self.rng.rand() < self.config.rgb_add_noise_twice_ratio:
                # apply rgb image disturbance a second time
                rgb = self.rgb_add_noise(rgb)

        dpt_um = self.bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6

        dpt_mm = (dpt_um.copy() / 10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )

        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)

        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
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
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)
        source_cld = cld.copy()

        cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)

        item_dict = self.get_pose_gt_info(cld, labels_pt, cls_id_lst, meta)

        h, w = rgb_labels.shape

        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        if not self.multi_view:
            xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w

            for i in range(3):
                scale = pow(2, i + 1)
                nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
                ys, xs = np.mgrid[:nh, :nw]
                xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])  # subsampling: takes only every scale-th pixel

            sr2dptxyz = {
                pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
            }  # list that contains dpt_xyz and multiple subsampled versions of it with different scaling factors.

            rgb_ds_sr = [4, 8, 8, 8]
            n_ds_layers = 4
            pcld_sub_s_r = [4, 4, 4, 4]

            # DownSample stage
            for i in range(n_ds_layers):
                nei_idx = DP.knn_search(
                    support_pts=cld[None, ...], query_pts=cld[None, ...], k=16
                ).astype(np.int32).squeeze(0)

                # reduce the point cloud by taking just the first 25% of the points.
                # Since cld is shuffled, it is the same as random sampling
                sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]

                pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
                up_i = DP.knn_search(
                    support_pts=sub_pts[None, ...], query_pts=cld[None, ...], k=1
                ).astype(np.int32).squeeze(0)
                item_dict['cld_xyz%d' % i] = cld.astype(np.float32).copy()
                item_dict['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
                item_dict['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
                item_dict['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
                nei_r2p = DP.knn_search(  # looks for the k nearest neighbors of the subsampled pointcloud in the subsampled depth image.
                    support_pts=sr2dptxyz[rgb_ds_sr[i]][None, ...], query_pts=sub_pts[None, ...], k=16  # K_r2p = 16
                ).astype(np.int32).squeeze(0)
                item_dict['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
                nei_p2r = DP.knn_search(  # looks for the k nearest neighbors of the subsampled depth image in the subsampled pointcloud.
                    support_pts=sub_pts[None, ...], query_pts=sr2dptxyz[rgb_ds_sr[i]][None, ...], k=1  # K_p2r = 1
                ).astype(np.int32).squeeze(0)
                item_dict['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
                cld = sub_pts

            n_up_layers = 3
            rgb_up_sr = [4, 2, 2]
            for i in range(n_up_layers):
                r2p_nei = DP.knn_search(
                    support_pts=sr2dptxyz[rgb_up_sr[i]][None, ...],
                    query_pts=item_dict['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...], k=16
                ).astype(np.int32).squeeze(0)
                item_dict['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
                p2r_nei = DP.knn_search(
                    support_pts=item_dict['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...],
                    query_pts=sr2dptxyz[rgb_up_sr[i]][None, ...], k=1
                ).astype(np.int32).squeeze(0)
                item_dict['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        cam_pose = meta.get('rotation_translation_matrix', np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
        cam_pose = np.concatenate((cam_pose, np.array([[0, 0, 0, 1]])), axis=0)

        item_dict['rgb'] = rgb.astype(np.float32)  # uint8, [c, h, w]
        item_dict['cld'] = source_cld.astype(np.float32)  # float32, (npts, 3)
        item_dict['dpt_xyz'] = dpt_xyz.astype(np.float32)  # float32, [h, w, c]
        item_dict['cld_rgb_nrm'] = cld_rgb_nrm.astype(np.float32).T  # [9, npts]
        item_dict['choose'] = choose.astype(np.int32)  # [1, npts]
        item_dict['labels'] = labels_pt.astype(np.int32)  # [npts]
        item_dict['rgb_labels'] = rgb_labels.astype(np.int32)  # [h, w]
        item_dict['dpt_map_m'] = dpt_m.astype(np.float32)  # [h, w]
        item_dict['cam_pose'] = cam_pose.astype(np.float32)
        item_dict['camera_intrinsics'] = K.astype(np.float32)

        return item_dict

    def get_pose_gt_info(self, cld, labels, cls_id_lst, meta):
        item_dict = {}
        RTs = np.zeros((self.config.n_objects, 3, 4), dtype=np.float32)
        cls_ids = np.zeros((self.config.n_objects, 1), dtype=np.float32)

        if self.provide_symmetry:
            max_refl_sym = 0
            max_discrete_rots = self.n_rot_sym if self.rotationals else 0
            max_rot_sym = 1 * max(0, (max_discrete_rots - 1))
            max_sym = max_refl_sym + max_rot_sym
            kp_3ds = np.ones((self.config.n_objects, max_sym + 1, self.config.n_keypoints, 3), dtype=np.float32) * np.nan
            ctr_3ds = np.ones((self.config.n_objects, max_sym + 1, 3), dtype=np.float32) * np.nan
            kp_targ_ofst = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3), dtype=np.float32)
            kp_targ_ofst_sym = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3), dtype=np.float32)
            kp_targ_ofst_sym_kp = np.zeros((self.config.n_sample_points, max_sym + 1, self.config.n_keypoints, 3),
                                           dtype=np.float32)
            ctr_targ_ofst = np.zeros((self.config.n_sample_points, 3), dtype=np.float32)
            ctr_targ_ofst_sym = np.zeros((self.config.n_sample_points, 3), dtype=np.float32)
            ctr_targ_ofst_sym_kp = np.zeros((self.config.n_sample_points, max_sym + 1, 3), dtype=np.float32)
        else:
            kp_3ds = np.zeros((self.config.n_objects, self.config.n_keypoints, 3), dtype=np.float32)
            ctr_3ds = np.zeros((self.config.n_objects, 3), dtype=np.float32)
            kp_targ_ofst = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3), dtype=np.float32)
            ctr_targ_ofst = np.zeros((self.config.n_sample_points, 3), dtype=np.float32)

        for i, cls_id in enumerate(cls_id_lst):
            if self.provide_symmetry:
                obj_symmetries = self.symmetries[self.symmetries['obj'] == self.obj_dict_id[cls_id]]

            r = meta['poses'][:, :, i][:, 0:3]
            t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            ctr = self.bs_utils.get_ctr(self.cls_lst[cls_id - 1]).copy()[:, None]
            ctr_obj = ctr.copy()
            ctr = np.dot(ctr.T, r.T) + t[:, 0]
            msk_idx = np.where(labels == cls_id)[0]
            cls_ids[i, :] = np.array([cls_id])

            if self.provide_symmetry:
                refl_sym = np.stack(
                    [np.stack([float(obj_symmetries[f'refl{s + 1}_{dim_}']) for dim_ in ['x', 'y', 'z']], axis=0) for s
                     in range(3)], axis=0).squeeze()
                rot_sym = np.stack(
                    [np.stack([float(obj_symmetries[f'rot{s + 1}_{dim_}']) for dim_ in ['x', 'y', 'z']], axis=0) for s
                     in range(3)], axis=0).squeeze()
                n_rot = np.stack([float(obj_symmetries[f'n_rot{s + 1}']) for s in range(3)], axis=0).squeeze()
                sym_ctr = np.stack([float(obj_symmetries[f'ctr_{dim_}']) for dim_ in ['x', 'y', 'z']], axis=0).squeeze()

            if self.provide_symmetry:

                ctr_3ds[i, 0, :] = ctr[0]
                target_offset = np.array(np.add(cld, -ctr_3ds[i, 0, :]))
                ctr_targ_ofst_sym[msk_idx, :] = target_offset[msk_idx, :]
                ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]

                for si, s in enumerate(refl_sym):
                    if np.isnan(s).any() or cls_id not in self.sym_cls_ids:
                        continue
                    symm_kps = symmetry.mirror_mesh_np(ctr_obj.T, sym_ctr, s)
                    symm_kps = np.dot(symm_kps, r.T) + t[:, 0]
                    ctr_3ds[i, si + 1] = symm_kps

                if self.rotationals:
                    for si, (ns, s) in enumerate(zip(n_rot, rot_sym)):
                        if np.isnan(s).any() or ns <= 0 or cls_id not in self.sym_cls_ids:
                            continue
                        v1 = sym_ctr.squeeze() + s
                        v2 = sym_ctr.squeeze() - s

                        if ns == np.inf:
                            ns = max_discrete_rots

                        for step in range(1, int(ns)):
                            theta = step * (360 / ns)
                            symm_kps = symmetry.rotate_np(ctr_obj.T, v1, v2, theta).T
                            symm_kps = np.dot(symm_kps, r.T) + t[:, 0]

                            ctr_3ds[i, max_refl_sym + si + step] = symm_kps

                for ikpg, kp in enumerate(ctr_3ds[i]):
                    target_offsets = np.array(np.add(cld, -kp))  # [n_pts, n_kps, c]
                    ctr_targ_ofst_sym_kp[msk_idx, ikpg, :] = target_offsets[msk_idx, :]

            else:
                ctr_3ds[i, :] = ctr[0]
                target_offset_old = np.array(np.add(cld, -ctr_3ds[i, :]))
                target_offset = cld - ctr_3ds[i, :]
                assert np.all(target_offset == target_offset_old)

                ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]

            if self.config.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(self.config.n_keypoints)
            kps = self.bs_utils.get_kps(self.cls_lst[cls_id - 1], kp_type=kp_type, ds_type='ycb').copy()
            mesh_kps = kps.copy()

            if self.provide_symmetry:
                kps = np.dot(kps, r.T) + t[:, 0]
                kp_3ds[i, 0] = kps
                for si, s in enumerate(refl_sym):
                    if np.isnan(s).any() or cls_id not in self.sym_cls_ids:
                        continue
                    symm_kps = symmetry.mirror_mesh_np(mesh_kps, sym_ctr, s)
                    symm_kps = np.dot(symm_kps, r.T) + t[:, 0]
                    kp_3ds[i, si + 1] = symm_kps

                if self.rotationals:
                    for si, (ns, s) in enumerate(zip(n_rot, rot_sym)):
                        if np.isnan(s).any() or ns <= 0 or cls_id not in self.sym_cls_ids:
                            continue
                        v1 = sym_ctr.squeeze() + s
                        v2 = sym_ctr.squeeze() - s

                        if ns == np.inf:
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
                item_dict['ctr_targ_ofst_sym'] = ctr_targ_ofst_sym
                item_dict['kp_targ_ofst_sym_kp'] = kp_targ_ofst_sym_kp
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

    def generate_sequence_samples(self, sequences):
        random_state = np.random.RandomState(0)
        sequence_samples = []
        if self.dataset_mode == 'train' and self.uniform_pre:
            new_sequences = {}
            for k, v in sequences.items():
                div = 5
                for i in range(div):
                    new_sequences[k + '_' + str(i)] = v[i::div]

            sequences = new_sequences
        for scene_idx in sequences.keys():
            group_ids = random_state.permutation(sequences[scene_idx])
            len_group = len(group_ids)
            for k, m in enumerate(np.arange(len_group)[::self.set_views]):
                ids_k = np.arange(len(group_ids))[m:m + self.set_views].tolist()
                if len(ids_k) < self.set_views:
                    continue
                ds_ids = group_ids[ids_k]
                scene = [os.path.join('data', scene_idx, ds_ids[i]) for i in range(self.set_views)]

                sequence_samples.append(scene)
                if self.dataset_mode == 'train':
                    for _ in range(self.set_views - 1):
                        scene = scene[1:] + scene[:1]
                        sequence_samples.append(scene)

        return sequence_samples

    def transform_sample(self, target_sample, sample, i):
        """
        Transforms a point cloud from camera frame cam0 to camera frame cam1
        cam0 and cam1 are actually expected to be the inverse of the camera position, following YCB's convention
        """
        # keys to transform: cld_rgb_nrm, kp_targ_ofst, ctr_targ_ofst
        # input keys to transform: cld_xyz0, cld_xyz1, cld_xyz2, cld_xyz3
        camchange = (target_sample['cam_pose'] @ np.linalg.inv(sample['cam_pose']))

        new_cloud = self.transform_data(camchange, sample['cld_rgb_nrm'][..., :3])
        sample['cld'] = self.transform_data(camchange, sample['cld']).astype(np.float32)
        sample['dpt_xyz'] = self.transform_data(camchange, sample['dpt_xyz']).astype(np.float32)

        new_normals = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3], new_cloud, sample['cld_rgb_nrm'][..., -3:]).astype(np.float32)
        new_cld_rgb_norm = np.concatenate([new_cloud, sample['cld_rgb_nrm'][..., 3:6], new_normals], axis=1).astype(np.float32)

        sample['kp_targ_ofst'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3].reshape(-1, 1, 3).repeat(repeats=8, axis=1),
                                                        new_cloud.reshape(-1, 1, 3).repeat(repeats=8, axis=1), sample['kp_targ_ofst']).astype(np.float32)
        sample['ctr_targ_ofst'] = self.transform_offsets(camchange, sample['cld_rgb_nrm'][..., :3], new_cloud, sample['ctr_targ_ofst']).astype(np.float32)

        if self.provide_symmetry:
            sample['kp_targ_ofst_sym'] = self.transform_offsets(
                camchange, sample['cld_rgb_nrm'][..., :3].reshape(-1, 1, 3).repeat(repeats=8, axis=1),
                new_cloud.reshape(-1, 1, 3).repeat(repeats=8, axis=1), sample['kp_targ_ofst_sym'])
            sample['ctr_targ_ofst_sym'] = self.transform_offsets(
                camchange, sample['cld_rgb_nrm'][..., :3], new_cloud, sample['ctr_targ_ofst_sym'])

            sample['kp_targ_ofst_sym_kp'] = self.transform_offsets(
                camchange, sample['cld_rgb_nrm'][..., :3].reshape(-1, 1, 3).repeat(repeats=8, axis=1)[:, None],
                new_cloud.reshape(-1, 1, 3).repeat(repeats=8, axis=1)[:, None], sample['kp_targ_ofst_sym_kp'])
            sample['ctr_targ_ofst_sym_kp'] = self.transform_offsets(
                camchange, sample['cld_rgb_nrm'][..., :3][:, None], new_cloud[:, None],
                sample['ctr_targ_ofst_sym_kp'])

        sample['cld_rgb_nrm'] = new_cld_rgb_norm.astype(np.float32)

        return sample

    def transform_data(self, camchange, data):
        """

        :param camchange:
        :param data:
        :return:
        """
        ones = np.ones((*data.shape[:-1], 1))
        if len(data.shape) > 2:
            ones_flatt = ones.reshape(-1, 1)
            data_flatt = data.reshape(-1, 3)
            new_data_flatt = camchange @ np.concatenate([data_flatt, ones_flatt], axis=-1).T
            new_data_flatt = new_data_flatt[:3, :].T
            new_data = new_data_flatt.reshape(data.shape)
        else:
            new_data = camchange @ np.concatenate([data, ones], axis=-1).T
            new_data = new_data[:3, :].T
        return new_data

    def fuse_sequence(self, sequence):
        """
        Create a sequence consisting of multiple views including RGB images, PCLs, etc.
        :param sequence:
        :return:
        """
        item_dict = {}
        item_dict['rgb'] = np.stack([view['rgb'] for view in sequence], axis=0)
        item_dict['cld_rgb_nrm'] = np.concatenate([view['cld_rgb_nrm'] for view in sequence], axis=0)
        item_dict['cld'] = np.concatenate([view['cld'] for view in sequence], axis=0)

        item_dict['kp_targ_ofst'] = np.concatenate([view['kp_targ_ofst'] for view in sequence], axis=0)
        item_dict['ctr_targ_ofst'] = np.concatenate([view['ctr_targ_ofst'] for view in sequence], axis=0)

        if self.provide_symmetry:
            item_dict['kp_targ_ofst_sym'] = np.concatenate([view['kp_targ_ofst_sym'] for view in sequence], axis=0)
            item_dict['ctr_targ_ofst_sym'] = np.concatenate([view['ctr_targ_ofst_sym'] for view in sequence], axis=0)

            item_dict['kp_targ_ofst_sym_kp'] = np.concatenate([view['kp_targ_ofst_sym_kp'] for view in sequence], axis=0)
            item_dict['ctr_targ_ofst_sym_kp'] = np.concatenate([view['ctr_targ_ofst_sym_kp'] for view in sequence], axis=0)

        item_dict['labels'] = np.concatenate([view['labels'] for view in sequence], axis=0)

        item_dict['choose'] = np.stack([view['choose'] for view in sequence], axis=0)
        item_dict['rgb_labels'] = np.stack([view['rgb_labels'] for view in sequence], axis=0)

        item_dict['cls_ids'] = sequence[0]['cls_ids']
        item_dict['RTs'] = sequence[0]['RTs']
        item_dict['kp_3ds'] = sequence[0]['kp_3ds']
        item_dict['ctr_3ds'] = sequence[0]['ctr_3ds']
        item_dict['cam_pose'] = sequence[0]['cam_pose']
        item_dict['camera_intrinsics'] = sequence[0]['camera_intrinsics']

        h, w = item_dict['rgb_labels'][0].shape
        item_dict['dpt_xyz'] = np.stack([view['dpt_xyz'] for view in sequence], axis=0)

        # For each view, sr2dptxyz_lst stores the depth image in original and multiple down-sampled resolutions:
        sr2dptxyz_lst = []
        for s in sequence:
            dpt_xyz = s['dpt_xyz']
            xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w

            for i in range(3):
                scale = pow(2, i+1)
                nh, nw = h // pow(2, i+1), w // pow(2, i+1)
                ys, xs = np.mgrid[:nh, :nw]
                xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            sr2dptxyz = {
                pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
            }
            sr2dptxyz_lst.append(sr2dptxyz)

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]

        cld = item_dict['cld']
        cld_views = [view['cld'] for view in sequence]

        # DownSample stage
        inputs = {}
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                support_pts=cld[None, ...], query_pts=cld[None, ...], k=16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[::pcld_sub_s_r[i], :]  # subsampled multi-view point cloud
            pool_i = nei_idx[::pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                support_pts=sub_pts[None, ...], query_pts=cld[None, ...], k=1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()

            # nearest neighbor indices with i-th downsampling layer and v-th view
            inputs[f'p2r_ds_nei_idx{i}'] = []
            inputs[f'r2p_ds_nei_idx{i}'] = []
            for v, sr2dptxyz in enumerate(sr2dptxyz_lst):
                cld_view = cld_views[v]  # point cloud of v-th view
                sub_pts_view = cld_view[::pcld_sub_s_r[i], :]  # subsampled point cloud of v-th view
                inputs['cld_xyz%d%d' % (i, v)] = cld_views[v].astype(np.float32).copy()

                # compute the k=16 nearest neighbors of the subsampled single-view point cloud
                # in the (subsampled) depth image of the same view v
                nei_r2p = DP.knn_search(
                    support_pts=sr2dptxyz[rgb_ds_sr[i]][None, ...],
                    query_pts=sub_pts_view[None, ...],
                    k=16
                ).astype(np.int32).squeeze(0)
                inputs[f'r2p_ds_nei_idx{i}'].append(nei_r2p.copy())
                # r2p_ds_nei_idx0 refers to the 0-th downsampling layer.

                # compute the nearest neighbors of the (subsampled) depth image
                # in the subsampled multi-view point cloud
                nei_p2r = DP.knn_search(
                    support_pts=sub_pts[None, ...],
                    query_pts=sr2dptxyz[rgb_ds_sr[i]][None, ...],
                    k=1
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
                    support_pts=sr2dptxyz[rgb_up_sr[i]][None, ...],
                    query_pts=inputs[f'cld_xyz{(n_ds_layers-i-1)}{v}'][None, ...], k=16
                ).astype(np.int32).squeeze(0)
                inputs[f'r2p_up_nei_idx{i}'].append(r2p_nei.copy())
                p2r_nei = DP.knn_search(
                    support_pts=inputs[f'cld_xyz{(n_ds_layers-i-1)}'][None, ...],
                    query_pts=sr2dptxyz[rgb_up_sr[i]][None, ...], k=1
                ).astype(np.int32).squeeze(0)
                inputs[f'p2r_up_nei_idx{i}'].append(p2r_nei.copy())

        # convert the nearest neighbor indices from list to tensor
        # idx0 refers to the 0-th downsampling layer
        for k in inputs.keys():
            if k.startswith('r2p') or k.startswith('p2r'):
                inputs[k] = np.stack([v for v in inputs[k]], axis=0)

        item_dict.update(inputs)

        return item_dict

    def __getitem__(self, idx):
        if self.dataset_mode == 'train':
            if self.multi_view:
                if self.syn_train_data_ratio and self.rng.rand() < self.syn_train_data_ratio:
                    n = len(self.syn_ds.sequence_samples)
                    idx = self.rng.randint(0, n)
                    sequence = self.syn_ds.get_sequence(self.syn_ds.sequence_samples[idx])
                    for i in range(1, len(sequence)):
                        sequence[i] = self.syn_ds.transform_sample(sequence[0], sequence[i], i)
                    fused_sample = self.syn_ds.fuse_sequence(sequence)
                else:
                    n = len(self.sequence_samples)
                    idx = self.rng.randint(0, n)
                    sequence = self.get_sequence(self.sequence_samples[idx])
                    for i in range(1, len(sequence)):
                        sequence[i] = self.transform_sample(sequence[0], sequence[i], i)

                    fused_sample = self.fuse_sequence(sequence)

                    # convert numpy to torch
                    for key in fused_sample.keys():
                        fused_sample[key] = torch.from_numpy(fused_sample[key])

                fused_sample['cld_rgb_nrm'] = np.swapaxes(fused_sample['cld_rgb_nrm'], 0, 1)
                return fused_sample

            else:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
                while data is None:
                    item_name = self.real_syn_gen()
                    data = self.get_item(item_name)
                data['cld_rgb_nrm'] = np.swapaxes(data['cld_rgb_nrm'], 0, 1)
                return data
        else:
            if self.multi_view:
                sequence = self.get_sequence(self.sequence_samples[idx])
                for i in range(1, len(sequence)):
                    sequence[i] = self.transform_sample(sequence[0], sequence[i], i)

                fused_sample = self.fuse_sequence(sequence)
                fused_sample['cld_rgb_nrm'] = np.swapaxes(fused_sample['cld_rgb_nrm'], 0, 1)
                return fused_sample
            else:
                item_name = self.all_lst[idx]
                data = self.get_item(item_name)
                data['cld_rgb_nrm'] = np.swapaxes(data['cld_rgb_nrm'], 0, 1)
                return data
