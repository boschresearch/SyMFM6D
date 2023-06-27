#!/usr/bin/env python3
import os
import csv
import time
import torch
import numpy as np
import pickle as pkl
import concurrent.futures

from common import Config
from utils.basic_utils import Basic_Utils
from utils.meanshift_pytorch import MeanShiftTorch
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except Exception:
    from cv2 import imshow, waitKey

import utils.symmetry_transforms as symmetry
from sklearn.neighbors import NearestNeighbors


dataset_global = None

config = Config()
bs_utils = Basic_Utils(config)


# ###############################YCB Evaluation###############################
def cal_frame_poses(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
    gt_kps, gt_ctrs, kp_type='farthest',proposals=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    cls_lst = config.cls_lst
    
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    # Use center clustering filter to improve the predicted mask.
    pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
    if use_ctr_clus_flter:
        ctrs = []
        for icls, cls_id in enumerate(pred_cls_ids):
            cls_msk = (mask == cls_id)
            ms = MeanShiftTorch(bandwidth=radius)
            ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
            ctrs.append(ctr.detach().contiguous().cpu().numpy())
        try:
            ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
            n_ctrs, _ = ctrs.size()
            pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
            ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
            ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
            min_dis, min_idx = torch.min(ctr_dis, dim=1)
            msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
            new_msk = mask.clone()
            for cls_id in pred_cls_ids:
                if cls_id == 0:
                    break
                min_msk = min_dis < config.r_lst[cls_id-1] * 0.8
                update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
                new_msk[update_msk] = msk_closest_ctr[update_msk]
            mask = new_msk
        except Exception:
            pass

    # 3D keypoints voting and least squares fitting for pose parameters estimation.
    pred_pose_lst = []
    pred_kps_lst = []
    in_pred_kp_lst=[]
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            break
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            pred_pose_lst.append(np.identity(4)[:3, :])
            pred_kps_lst.append(np.zeros((n_kps+1, 3)))
            continue

        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps
        if proposals:
            in_pred_kp_lst.append(in_pred_kp.detach().contiguous().cpu().numpy())

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)


        # Get mesh keypoint & center point in the object coordinate system.
        # If you use your own objects, check that you load them correctly.
        mesh_kps = bs_utils.get_kps(cls_lst[cls_id-1], kp_type=kp_type)
        if use_ctr:
            mesh_ctr = bs_utils.get_ctr(cls_lst[cls_id-1]).reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_kpc = cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        pred_RT = bs_utils.best_fit_transform(mesh_kps, pred_kpc)
        pred_kps_lst.append(pred_kpc)
        pred_pose_lst.append(pred_RT)

    if proposals:
        return pred_cls_ids, pred_pose_lst, pred_kps_lst, in_pred_kp_lst
    else:
        return pred_cls_ids, pred_pose_lst, pred_kps_lst


def eval_metric(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label,
    gt_kps, gt_ctrs, pred_kpc_lst
):
    cls_lst = config.cls_lst

    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_kp_err = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls].contiguous().detach().cpu().numpy()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_kp = np.zeros(gt_kp.shape)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        kp_err = np.linalg.norm(gt_kp-pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return cls_add_dis, cls_adds_dis, cls_kp_err


def eval_one_frame_pose(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, gt_kps, gt_ctrs, kp_type = item

    pred_cls_ids, pred_pose_lst, pred_kpc_lst = cal_frame_poses(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        gt_kps, gt_ctrs, kp_type=kp_type
    )

    cls_add_dis, cls_adds_dis, cls_kp_err = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs,
        pred_kpc_lst
    )
    return cls_add_dis, cls_adds_dis, None, pred_cls_ids, pred_pose_lst, cls_kp_err

# ###############################End YCB Evaluation###############################


# ###############################Scape Evaluation###############################

def cal_frame_poses_scape(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
    gt_kps, gt_ctrs, kp_type='farthest', proposals=False, ds=None
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    if ds is not None:
        config = Config(ds_name=ds)
    else:
        config = Config(ds_name=dataset_global)
        ds = dataset_global
    bs_utils = Basic_Utils(config)
    cls_lst = config.cls_lst
    
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    # Use center clustering filter to improve the predicted mask.
    pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
    if use_ctr_clus_flter:
        ctrs = []
        for icls, cls_id in enumerate(pred_cls_ids):
            cls_msk = (mask == cls_id)
            ms = MeanShiftTorch(bandwidth=radius)
            ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
            ctrs.append(ctr.detach().contiguous().cpu().numpy())
        try:
            ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
            n_ctrs, _ = ctrs.size()
            pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
            ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
            ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
            min_dis, min_idx = torch.min(ctr_dis, dim=1)
            msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
            new_msk = mask.clone()
            for cls_id in pred_cls_ids:
                if cls_id == 0:
                    continue
                min_msk = min_dis < config.r_lst[cls_id-1] * 0.8
                update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
                new_msk[update_msk] = msk_closest_ctr[update_msk]
            mask = new_msk
        except Exception:
            pass

    # 3D keypoints voting and least squares fitting for pose parameters estimation.
    pred_pose_lst = []
    pred_kps_lst = []
    in_pred_kp_lst = []
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            continue
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            pred_pose_lst.append(np.identity(4)[:3, :])
            pred_kps_lst.append(np.zeros((n_kps+1, 3)))
            continue

        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps
        if proposals:
            in_pred_kp_lst.append(in_pred_kp.detach().contiguous().cpu().numpy())

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)


        # Get mesh keypoint & center point in the object coordinate system.
        # If you use your own objects, check that you load them correctly.
        mesh_kps = bs_utils.get_kps(cls_lst[cls_id-1], kp_type=kp_type, ds_type=ds)
        if use_ctr:
            mesh_ctr = bs_utils.get_ctr(cls_lst[cls_id-1], ds_type=ds).reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_kpc = cls_kps[cls_id].squeeze().detach().contiguous().cpu().numpy()
        pred_RT = bs_utils.best_fit_transform(mesh_kps, pred_kpc)
        pred_kps_lst.append(pred_kpc)
        pred_pose_lst.append(pred_RT)

    if proposals:
        return pred_cls_ids, pred_pose_lst, pred_kps_lst, in_pred_kp_lst
    else:
        return pred_cls_ids, pred_pose_lst, pred_kps_lst


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = bs_utils.best_fit_transform_(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = bs_utils.best_fit_transform_(A, src[:m, :].T)

    return T, distances, i


def symmetric_kps(mesh_kps, mesh_ctr, refl_sym, rot_sym, n_rot):
    kp3ds = np.ones((8, 6, 3))*np.nan
    kp3ds[:, 0, :] = mesh_kps
    act_sym=0
    for si, s in enumerate(refl_sym):
        if np.isnan(s).any():
            continue
        symm_kps = symmetry.mirror_mesh_np(mesh_kps, mesh_ctr.squeeze(), s)
        kp3ds[:, si+1, :] = symm_kps
        act_sym += 1

    for si, (ns, s) in enumerate(zip(n_rot,rot_sym)):
        if np.isnan(s).any() or ns <= 0:
            continue
        v1 = mesh_ctr.squeeze() + s
        v2 = mesh_ctr.squeeze() - s

        if ns == np.inf:
            ns = 4

        for step in range(1,int(ns)):
            theta = step*(360/ns)
            symm_kps = symmetry.rotate_np(mesh_kps, v1, v2, theta).T


            kp3ds[:,act_sym+si+step,:] = symm_kps
    return kp3ds


def eval_metric_scape(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs, pred_kpc_lst
):
    config = Config(ds_name=dataset_global)
    bs_utils = Basic_Utils(config)
    cls_lst = config.cls_lst

    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_add_sym_dis = [list() for i in range(n_cls)]
    cls_kp_err = [list() for i in range(n_cls)]

    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            continue

        gt_kp = gt_kps[icls].contiguous().cpu().numpy()
        gt_ctr = gt_ctrs[icls].contiguous()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_kp = np.zeros(gt_kp.shape)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        kp_err = np.linalg.norm(gt_kp-pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1], ds_type=dataset_global).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        if dataset_global == "scape_2":
            addsym = bs_utils.cal_add_sym_cuda(pred_RT, gt_RT, mesh_pts, cls_id, gt_ctr)
            cls_add_sym_dis[cls_id].append(addsym.item())
            cls_add_sym_dis[0].append(addsym.item())

        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return cls_add_dis, cls_adds_dis, cls_add_sym_dis, cls_kp_err


def eval_one_frame_pose_scape(item):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, gt_kps, gt_ctrs, kp_type = item

    pred_cls_ids, pred_pose_lst, pred_kpc_lst = cal_frame_poses_scape(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        gt_kps, gt_ctrs, kp_type=kp_type, ds=dataset_global
    )

    cls_add_dis, cls_adds_dis, cls_add_sym_dis, cls_kp_err = eval_metric_scape(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs,
        pred_kpc_lst
    )
    return cls_add_dis, cls_adds_dis, cls_add_sym_dis, pred_cls_ids, pred_pose_lst, cls_kp_err


# ###############################End Scape Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval:

    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_sym_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self, save_path=None):
        cls_lst = config.cls_lst

        add_auc_lst = []  # ADD AUC
        adds_auc_lst = []  # ADD-S AUC
        add_s_auc_lst = []  # ADD(-S) AUC
        add_precision_lst = []  # ADD < 2cm
        adds_precision_lst = []  # ADD-S < 2cm
        add_s_precision_lst = []  # ADD(-S) < 2cm
        for cls_id in range(1, self.n_cls):
            if cls_id in config.sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)

            n = len(self.cls_add_dis[i])
            if n > 0:
                # compute ADD* < 2cm
                add_precision = sum(map(lambda x: x <= 0.02, self.cls_add_dis[i])) / n * 100
                adds_precision = sum(map(lambda x: x <= 0.02, self.cls_adds_dis[i])) / n * 100
                add_s_precision = sum(map(lambda x: x <= 0.02, self.cls_add_s_dis[i])) / n * 100
                add_precision_lst.append(add_precision)
                adds_precision_lst.append(adds_precision)
                add_s_precision_lst.append(add_s_precision)

            if i == 0:  # class number 0 is not an object -> We can ignore it.
                continue
            print(cls_lst[i-1])
            print(" * ADD:          ", add_auc)
            print(" * ADD-S:        ", adds_auc)
            print(" * ADD(-S):      ", add_s_auc)

            print(" * ADD < 2cm:    ", add_precision)
            print(" * ADD-S < 2cm:  ", adds_precision)
            print(" * ADD(-S) < 2cm:", add_s_precision)

        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])

        print("Average of all object:")
        print(" * ADD:          ", np.mean(add_auc_lst[1:]))
        print(" * ADD-S:        ", np.mean(adds_auc_lst[1:]))
        print(" * ADD(-S):      ", np.mean(add_s_auc_lst[1:]))

        print(" * ADD < 2cm:    ", np.mean(add_precision_lst[1:]))
        print(" * ADD-S < 2cm:  ", np.mean(adds_precision_lst[1:]))
        print(" * ADD(-S) < 2cm:", np.mean(add_s_precision_lst[1:]))

        print("All object (following PoseCNN):")
        print(" * ADD:          ", add_auc_lst[0])
        print(" * ADD-S:        ", adds_auc_lst[0])
        print(" * ADD(-S):      ", add_s_auc_lst[0])

        print(" * ADD < 2cm:    ", add_precision_lst[0])
        print(" * ADD-S < 2cm:  ", adds_precision_lst[0])
        print(" * ADD(-S) < 2cm:", add_s_precision_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )
        if not os.path.exists(os.path.join(config.log_eval_dir, save_path)):
            os.makedirs(os.path.join(config.log_eval_dir, save_path))
        sv_pth = os.path.join(
            config.log_eval_dir,
            save_path,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            config.log_eval_dir,
            save_path,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))
        return sv_info

    def cal_auc_scape(self, save_path, varient):
        config = Config(ds_name=varient)
        bs_utils = Basic_Utils(config)
        cls_lst = config.cls_lst

        add_auc_lst = []  # ADD AUC
        adds_auc_lst = []  # ADD-S AUC
        add_s_auc_lst = []  # ADD(-S) AUC
        add_precision_lst = []  # ADD < 2cm
        adds_precision_lst = []  # ADD-S < 2cm
        add_s_precision_lst = []  # ADD(-S) < 2cm
        add_sym_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if cls_id in config.sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_sym_auc = bs_utils.cal_auc(self.cls_add_sym_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            add_sym_auc_lst.append(add_sym_auc)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print(" * ADD:     ", add_auc)
            print(" * ADD-sym: ", add_sym_auc)
            print(" * ADD-S:   ", adds_auc)
            print(" * ADD(-S): ", add_s_auc)
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print(" * ADD:     ", np.mean(add_auc_lst[1:]))
        print(" * ADD-sym: ", np.mean(add_sym_auc_lst[1:]))
        print(" * ADD-S:   ", np.mean(adds_auc_lst[1:]))
        print(" * ADD(-S): ", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print(" * ADD:     ", add_auc_lst[0])
        print(" * ADD-sym: ", add_sym_auc_lst[0])
        print(" * ADD-S:   ", adds_auc_lst[0])
        print(" * ADD(-S): ", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add_sym_auc_lst=add_sym_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )

        if not os.path.exists(os.path.join(config.log_eval_dir, save_path)):
            os.makedirs(os.path.join(config.log_eval_dir, save_path))

        sv_pth = os.path.join(
            config.log_eval_dir,
            save_path,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0], add_sym_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            config.log_eval_dir,
            save_path,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0], add_sym_auc_lst[0]
            )
        )
        pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))
        return sv_info

    def eval_pose_parallel(
        self, pclds, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]

        data_gen = zip(
            pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
            cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
            labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:
            if ds == 'ycb':
                eval_func = eval_one_frame_pose
            elif ds == 'SymMovCam':
                eval_func = eval_one_frame_pose_scape
                global dataset_global
                dataset_global = ds

            for res in executor.map(eval_func, data_gen):
                if ds == 'ycb' or 'scape' in ds or ds == 'SymMovCam':
                    cls_add_dis_lst, cls_adds_dis_lst, cls_add_sym_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs = res
                    self.pred_id2pose_lst.append(
                        {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                    )

                    self.pred_kp_errs = self.merge_lst(
                        self.pred_kp_errs, pred_kp_errs
                    )
                    if cls_add_sym_dis_lst:
                        self.cls_add_sym_dis = self.merge_lst(
                            self.cls_add_sym_dis, cls_add_sym_dis_lst
                        )

                else:
                    cls_add_dis_lst, cls_adds_dis_lst = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )

    @staticmethod
    def merge_lst(targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ
