import io
import os
import cv2
import random
import shutil
import argparse
import traceback
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tensorboardX import SummaryWriter

if os.name == 'posix':
    import resource  # unix specific

from common import Config, ConfigRandLA
from models.loss import OFLoss, SymOFLoss, SymMultiOFLoss, FocalLoss

from datasets.ycb.ycb_dataset import YcbDataset
from datasets.mv_ycb_dataset import MvYcbDataset
from utils.basic_utils import Basic_Utils

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor
from utils.run_utils import get_current_git_hash, get_pip_packages
from utils.lightning_hacks import increase_filedesc_limit, MyDDP
from utils.lightning_callbacks import LRLoggingCallback
from utils.validation_interval_callback import ValidationIntervalScheduler
from utils.batchnorm_scheduler_callback import BNMomentumScheduler

from contextlib import redirect_stdout

np.set_printoptions(linewidth=150, suppress=True)
torch.set_printoptions(linewidth=150, sci_mode=False)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_n_batches_per_epoch(dataset_name):
    if dataset_name == "ycb":
        Dataset_desc = YcbDataset
        kwargs = {}
    elif dataset_name == 'SymMovCam':
        Dataset_desc = MvYcbDataset
        kwargs = {'variant': dataset_name, 'multi_view': False, 'noisy_cam_pose': False}

    return Dataset_desc('train', **kwargs).n_batches_per_epoch


def cal_view_pred_pose(model, config, bs_utils, cu_dt, end_points, dataset,obj_id=-1):
    _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

    pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
    if dataset == "ycb":
        pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
            pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
            end_points['pred_kp_ofs'][0], True, config.n_objects, True,
            None, None
        )
    elif dataset == 'SymMovCam':
        pred_cls_ids, pred_pose_lst, _ = cal_frame_poses_scape(
            pcld=pcld[0], mask=classes_rgbd[0], ctr_of=end_points['pred_ctr_ofs'][0],
            pred_kp_of=end_points['pred_kp_ofs'][0], use_ctr=True, n_cls=config.n_objects, use_ctr_clus_flter=True,
            gt_kps=None, gt_ctrs=None, ds=dataset
        )

    np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
    if dataset == 'ycb' or dataset == 'SymMovCam':
        np_rgb = np_rgb[:, :, ::-1].copy()
    for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
        idx = np.where(pred_cls_ids == cls_id)[0]
        if len(idx) == 0:
            continue
        pose = pred_pose_lst[idx[0]]
        if dataset == 'ycb' or dataset == 'SymMovCam':
            obj_id = int(cls_id[0])
        mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=dataset).copy()
        mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
        if dataset == "ycb":
            K = config.intrinsic_matrix["ycb_K1"]
        elif dataset == 'SymMovCam':
            K = config.intrinsic_matrix['scape']
        mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
        color = bs_utils.get_label_color(obj_id, n_classes=22, mode=2)
        np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
    if dataset == 'ycb' or dataset == 'SymMovCam':
        bgr = np_rgb

    return bgr[:, :, ::-1]


def model_fn_decorator(criterion, criterion_of, criterion_of_sym, criterion_of_sym_multi, config, hparams):
    def model_fn(model, data):
        cu_dt = {}

        for key in data.keys():
            if type(data[key]) is dict:
                cu_dt[key] = data[key]
            elif type(data[key]) is list:
                if type(data[key][0]) is not torch.tensor:
                    cu_dt[key] = data[key]
                else:
                    cu_dt[key] = [[obj.cuda() for obj in sub_list] for sub_list in data[key]]
            elif data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
            else:
                cu_dt[key] = data[key].to('cuda')
        end_points = model(cu_dt)
        labels = cu_dt['labels']
        loss_rgbd_seg = criterion(
            end_points['pred_rgbd_segs'], labels.view(-1)
        ).sum()
        if hparams['symmetry']:
            if hparams['multi_instance']:
                loss_kp_of = criterion_of_sym_multi(
                    end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst_sym_kp'], cu_dt['labels_instance'], cld_rgb_nrm=cu_dt['cld_rgb_nrm']
                ).sum()
            else:
                loss_kp_of = criterion_of_sym(
                    end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst_sym_kp'], labels
                ).sum()
            if hparams['dataset'] == 'ycb':
                loss_ctr_of = criterion_of_sym(
                end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst_sym_kp'].unsqueeze(-2), labels
            ).sum()
            else:
                loss_ctr_of = criterion_of(
                    end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
                ).sum()
        else:
            loss_kp_of = criterion_of(
                end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst'], labels
            ).sum()
            loss_ctr_of = criterion_of(
                end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
            ).sum()

        loss_lst = [
            (loss_rgbd_seg, hparams["loss_weights"][0]),
            (loss_kp_of, hparams["loss_weights"][1]), 
            (loss_ctr_of, hparams["loss_weights"][2]),
        ]
        loss = sum([ls * w for ls, w in loss_lst])
        loss_unweighted = sum([ls for ls, w in loss_lst])

        _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()
        acc_dict = {
            'acc_rgbd': acc_rgbd
        }
        loss_dict = {
                'loss_rgbd_seg': loss_rgbd_seg,
                'loss_kp_of': loss_kp_of,
                'loss_ctr_of': loss_ctr_of,
                'loss_all': loss,
                'loss_target': loss,
                'loss_unweighted': loss_unweighted
            }
        info_dict = loss_dict.copy()
        info_dict.update(acc_dict)

        return (
            end_points, loss, info_dict
        )

    return model_fn


class SyMFM6DModule(lightning.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.batch_size = hparams['mini_batch_size']

        self.config = self._get_hparam('config')
        self.bs_utils = self._get_hparam('bs_utils')
        self.symmetry = self._get_hparam('symmetry')
        self.criterion = FocalLoss(gamma=2)

        if self.symmetry:
            self.criterion_of_sym = SymOFLoss()
            self.criterion_of_sym_multi = SymMultiOFLoss()
            self.criterion_of = OFLoss()
        else:
            self.criterion_of_sym = None
            self.criterion_of_sym_multi = None
            self.criterion_of = OFLoss()

        from utils.pvn3d_eval_utils_kpls import TorchEval
        self.teval = TorchEval(self._get_hparam('n_classes'))

        self.rndla_cfg = self._get_hparam('rndla_cfg')

        from models.SyMFM6D import SyMFM6D
        self.model = SyMFM6D(n_classes=self.config.n_objects, n_pts=self.config.n_sample_points,
                           rndla_cfg=self.rndla_cfg, n_kps=self.config.n_keypoints, 
                           multi_view=self._get_hparam('multi_view'))
                           
        self.model_fn = model_fn_decorator(self.criterion, self.criterion_of, self.criterion_of_sym, self.criterion_of_sym_multi, self.config, self.hparams)

        self.it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`

        self.viz = self._get_hparam('viz')

    def _get_hparam(self, key):
        """
            default value: None
        """
        if key in self.hparams.keys():
            return self.hparams[key]
        else:
            print("? %s is not defined -> use default value" % key)
            return None

    def setup(self, stage: str):
        if self.hparams['dataset'] == 'ycb':
            Dataset_desc = YcbDataset
            views = 1 if self.hparams['set_views'] is None else int(self.hparams['set_views'])
            kwargs = {'multi_view': self.hparams['multi_view'], 
                      'set_views': views, 
                      'symmetries': self.hparams['symmetry'],
                      'n_rot_sym': self.hparams['n_rot_sym'],
                      'syn_train_data_ratio': self.hparams['syn_train_data_ratio']}
        elif self.hparams['dataset'] == 'SymMovCam':
            Dataset_desc = MvYcbDataset
            views = None if self.hparams['set_views'] is None else int(self.hparams['set_views'])
            kwargs = {'variant': self.hparams['dataset'], 
                      'sift_fps': self.hparams['sift_fps_kps'],
                      'multi_view': self.hparams['multi_view'],  
                      'noisy_cam_pose':  self.hparams['noisy_cam_pose'], 
                      'set_views': views,
                      'symmetries': self.hparams['symmetry'],
                      'n_rot_sym': self.hparams['n_rot_sym'],
                      'rotationals': (not self.hparams['no_rotationals'] and self.hparams['symmetry']),
                      'sym_classes': self.hparams['sym_classes'],
                      'new': self.hparams['new_sym']}
        else:
            raise NotImplementedError("dataset %s is not supported" % self.hparams['dataset'])

        if stage == 'fit':
            self.train_ds = Dataset_desc('train', **kwargs)
            self.val_ds = Dataset_desc('test', **kwargs)
        elif stage == 'test':
            self.test_ds = Dataset_desc('test', **kwargs)

    def collate_batch(self, batch_in):
        # batch_in is list of samples
        batch_out = {k: [] for k in batch_in[0].keys()}
        for sample in batch_in:
            # sample is dict of tensors and lists
            for k, v in sample.items():
                if type(v) is list:
                    batch_out[k].append(v)
                elif type(v) is dict:
                    batch_out[k].append(v)
                else:
                    if type(batch_out[k]) is list:
                        batch_out[k] = v.unsqueeze(0)
                    else:
                        if k == 'labels_instance':
                            dim_diff = v.shape[0] - batch_out[k].shape[1]
                            if dim_diff > 0:
                                pad = torch.zeros(batch_out[k].shape[0], abs(dim_diff), batch_out[k].shape[-1])
                                batch_out[k] = torch.cat((batch_out[k], pad), dim=1)

                            elif dim_diff < 0:
                                pad = torch.zeros(abs(dim_diff), v.shape[-1])
                                v = torch.cat((v, pad), dim=0)
                        batch_out[k] = torch.cat((batch_out[k], v.unsqueeze(0)), dim=0)

        return batch_out

    def train_dataloader(self) -> DataLoader:
        training_params = {"batch_size": self.config.mini_batch_size,
                           "shuffle": True,
                           "drop_last": True,
                           "pin_memory": True,
                           "num_workers": self.hparams['number_workers']}

        if self.hparams['multi_instance']:
            train_loader = DataLoader(self.train_ds, collate_fn=self.collate_batch, **training_params)
        else:
            train_loader = DataLoader(self.train_ds, **training_params)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_params = {"batch_size": self.config.val_mini_batch_size,
                       "shuffle": False,
                       "drop_last": False,
                       "num_workers": self.hparams['number_workers']}   

        if self.hparams['multi_instance']:
            val_loader = DataLoader(self.val_ds, collate_fn=self.collate_batch, **val_params)
        else:
            val_loader = DataLoader(self.val_ds, **val_params)

        return val_loader
    
    def test_dataloader(self) -> DataLoader:
        test_params = {"batch_size": self.config.test_mini_batch_size,
                       "shuffle": False,
                       "num_workers": self.hparams['number_workers']}  

        if self.hparams['multi_instance']:
            test_loader = DataLoader(self.test_ds, collate_fn=self.collate_batch, **test_params)
        else:
            test_loader = DataLoader(self.test_ds, **test_params)

        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams['lr'], weight_decay=self.hparams['weight_decay'])

        if self.hparams['lr_scheduler'] == 'cyclic':

            lr_mode = self.hparams['lr_mode'] if self.hparams['lr_mode'] else "triangular"

            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.hparams['lr'], max_lr=self.hparams['lr']*100,
                cycle_momentum=False,
                step_size_up=7,
                step_size_down=7,
                mode=lr_mode
            )
        else:
            lr_mode = self.hparams['lr_mode'] if self.hparams['lr_mode'] else "min"
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=lr_mode, factor=0.5, patience=3, threshold=0.0001, threshold_mode='rel',
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
            "interval": "step"
            }

    def training_step(self, batch, batch_idx):
        self.it += 1

        _, loss, res = self.model_fn(self.model, batch)
        
        logs = {
            "train/loss": loss,
            "train/loss_rgbd_seg": res['loss_rgbd_seg'],
            "train/loss_target": res['loss_target'],
            "train/loss_unweighted": res['loss_unweighted'],
            "train/loss_kp_of": res['loss_kp_of'],
            "train/loss_ctr_of": res['loss_ctr_of'],
            "train/acc_rgbd": res['acc_rgbd'],
        }

        for k, v in logs.items():
            self.log(k, v, sync_dist=True)

        if self.viz is not None:
            self.viz.update("train", self.it, res)

        return {
            'loss': loss,
        }

    def training_epoch_end(self, outputs):
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.it += 1
        eval_dict = {}
        total_loss = 0.0
        count = 1
        count += 1
        torch.cuda.empty_cache()
        with torch.no_grad():
            _, loss, eval_res = self.model_fn(self.model, batch)
        if 'loss_target' in eval_res.keys():
            total_loss += eval_res['loss_target']
        else:
            total_loss += loss.item()
        for k, v in eval_res.items():
            if v is not None:
                eval_dict[k] = eval_dict.get(k, []) + [v.item()]

        return {'val_loss': loss,
                "val_loss_rgbd_seg": eval_res['loss_rgbd_seg'],
                'val_loss_target': eval_res['loss_target'],
                'val_loss_unweighted': eval_res['loss_unweighted'],
                'val_loss_kp_of': eval_res['loss_kp_of'],
                'val_loss_ctr_of': eval_res['loss_ctr_of'],
                'val_acc_rgbd': eval_res['acc_rgbd'], 
                }

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean().detach()
        loss_rgbd_seg = torch.stack([o['val_loss_rgbd_seg'] for o in outputs]).mean().detach()
        loss_target = torch.stack([o['val_loss_target'] for o in outputs]).mean().detach()
        loss_unweighted = torch.stack([o['val_loss_unweighted'] for o in outputs]).mean().detach()
        loss_kp_of = torch.stack([o['val_loss_kp_of'] for o in outputs]).mean().detach()
        loss_ctr_of = torch.stack([o['val_loss_ctr_of'] for o in outputs]).mean().detach()
        val_acc_rgbd = torch.stack([o['val_acc_rgbd'] for o in outputs]).mean().detach()
        torch.cuda.empty_cache()

        delta = datetime.now() - self.hparams['start_time']
        logs = {
            "hours": 24*delta.days + delta.seconds / 3600,
            "val/loss": loss,
            "val/loss_rgbd_seg": loss_rgbd_seg,
            "val/loss_target": loss_target,
            "val/loss_unweighted": loss_unweighted,
            "val/loss_kp_of": loss_kp_of,
            "val/loss_ctr_of": loss_ctr_of,
            "val/acc_rgbd": val_acc_rgbd,
        }

        for k, v in logs.items():
            self.log(k, v, on_epoch=True, sync_dist=True)

        return {
            "val_loss": loss,
            'val_acc': val_acc_rgbd, 
        }

    @torch.no_grad()
    def on_test_start(self) -> None:
        self.cls_add_dis = [list() for _ in range(self.hparams['n_classes'])]
        self.cls_adds_dis = [list() for _ in range(self.hparams['n_classes'])]
        self.cls_add_s_dis = [list() for _ in range(self.hparams['n_classes'])]
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        cu_dt = {}
        for key in batch.keys():
            if type(batch[key]) is dict:
                cu_dt[key]=batch[key]
            elif type(batch[key]) is list:
                if type(batch[key][0]) is not torch.tensor:
                    cu_dt[key]=batch[key]
                else:
                    cu_dt[key] = [[obj.cuda() for obj in sub_list] for sub_list in batch[key]]
            elif batch[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(batch[key].astype(np.float32)).cuda()
            elif batch[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(batch[key].astype(np.int32)).cuda()
            elif batch[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = batch[key].float().cuda()
            elif batch[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = batch[key].long().cuda()
            else:
                cu_dt[key] = batch[key].to('cuda')
        with torch.no_grad():
            end_points = self.model(cu_dt)
        labels = cu_dt['labels']
        _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        cld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        kp_type = 'sift_fps_kps' if hparams['sift_fps_kps'] else 'farthest'

        with torch.no_grad():
            if self.hparams['multi_instance']:

                self.teval.eval_pose_parallel_mimo(
                    cld, cu_dt['rgb'], cls_rgbd, end_points['pred_ctr_ofs'],
                    cu_dt['ctr_targ_ofst'], labels, self.current_epoch, cu_dt['cls_ids'],
                    cu_dt['RTs'], end_points['pred_kp_ofs'],
                    cu_dt['kp_3ds'], cu_dt['ctr_3ds'],
                    ds=self.hparams['dataset'], obj_id=config.cls_id,
                    min_cnt=1, use_ctr_clus_flter=False, use_ctr=True, kp_type=kp_type, n_objs=cu_dt['n_objs']
                )
            elif self.hparams['gt_masks']:
                print('using gt masks')
                self.teval.eval_pose_parallel(
                    cld, cu_dt['rgb'], labels, end_points['pred_ctr_ofs'],
                    cu_dt['ctr_targ_ofst'], labels, self.current_epoch, cu_dt['cls_ids'],
                    cu_dt['RTs'], end_points['pred_kp_ofs'],
                    cu_dt['kp_3ds'], cu_dt['ctr_3ds'],
                    ds=self.hparams['dataset'], obj_id=config.cls_id,
                    min_cnt=1, use_ctr_clus_flter=True, use_ctr=True, kp_type=kp_type
                )
            else:
                pass

            with torch.no_grad():
                self.teval.eval_pose_parallel(
                    cld, cu_dt['rgb'], cls_rgbd, end_points['pred_ctr_ofs'],
                    cu_dt['ctr_targ_ofst'], labels, self.current_epoch, cu_dt['cls_ids'],
                    cu_dt['RTs'], end_points['pred_kp_ofs'],
                    cu_dt['kp_3ds'], cu_dt['ctr_3ds'],
                    ds=hparams['dataset'], obj_id=config.cls_id,
                    min_cnt=1, use_ctr_clus_flter=True, use_ctr=True, kp_type=kp_type
                )

        return torch.tensor([1])

    @torch.no_grad()
    def on_test_end(self) -> None:
        if hasattr(self.config, 'out_dir'):
            save_path = ""
        else:
            save_path = self.hparams['logger_name'] + '/' + self.hparams['logger_version']
        f = io.StringIO()
        with redirect_stdout(f):
            if self.hparams['dataset'] == 'ycb' or self.hparams['dataset'] == 'SymMovCam':
                eval_dict = self.teval.cal_auc(save_path=save_path)
                if not self.hparams['multi_view']:
                    self.teval.save_csv(save_path=save_path)
        out = f.getvalue()
        print(out)
        md_out = out.replace('\n','<br>')
        md_out = md_out.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;')
        md_out = md_out.replace('*', '\*')
        self.logger.experiment.add_text("eval", md_out, self.it)

        self.trainer.save_checkpoint(os.path.join(self.config.log_eval_dir, self.hparams['logger_name'] + '/' + self.hparams['logger_version'],
                                    f"checkpoint_{eval_dict['adds_auc_lst'][0]}_{eval_dict['add_auc_lst'][0]}_{eval_dict['add_s_auc_lst'][0]}.ckpt"))

    def forward(self, x):
        return self.model_fn(self.model, x)


def get_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('--dataset', type=str, default='dataset to be used [ycb, SymMovCam]', required=True)
    parser.add_argument('--run_name', type=str, default="SyMFM6D_test_run", help='The name of the run and log dir')
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization coefficient [default: 0.0]")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate [default: 1e-4]")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay gamma [default: 0.5]")
    parser.add_argument("--decay_step", type=float, default=2e5, help="Learning rate decay step [default: 2e5]")
    parser.add_argument("--bn_momentum", type=float, default=0.9, help="Initial batch norm momentum [default: 0.9]")
    parser.add_argument("--bn_decay", type=float, default=0.5, help="Batch norm momentum decay gamma [default: 0.5]")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument("--run_eval", type=int, default=0, help="Run evaluation.")
    parser.add_argument('--non_deterministic', type=int, default=0)
    parser.add_argument('--opt_level', default="O0", type=str, help='opt level of apex mix precision training.')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size_per_gpu", type=int, default=3, help="The number of images per batch")
    parser.add_argument("--epochs", type=int, default=25, help="The number of total epochs to be trained")
    parser.add_argument("--workers_per_gpu", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--logger_version", type=int, default=None, help="Use the this to reuse a specific tensorboard log version")
    parser.add_argument("--checkpoint_logger", type=str, default=None, help="Path to the logger of the checkpoint")
    parser.add_argument('--sift_fps_kps', type=int, default=0, help="Use SIFT-FPS instead of FPS keypoints for YCB objects")
    parser.add_argument("--multi_view", type=int, default=0, help="Uses multi-view data")
    parser.add_argument("--set_views", default=None, help="determines how many views are used for muli view training, if not set the standard number of views for each dataset are used")
    parser.add_argument("--loss_weights", type=float, nargs='+', default=[2.0, 1.0, 1.0], help="Weighting of segmentation, keypoint offset, and keypoint center loss")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="accumulate gradient to simulate larger batch size")
    parser.add_argument("--noisy_cam_pose", type=int, default=0, help="Whether or not the cam poses have a small random offset")
    parser.add_argument("--new_optimizer", type=int, default=0, help="Whether use a fresh optimizer and lr scheduler when loading a checkpoint instead of loading the old one")

    parser.add_argument("--symmetry", type=int, default=0, help="Whether to include symmetric keypoints in the learning regime")
    parser.add_argument("--n_rot_sym", type=int, default=16, help="Number of discrete rotational symmetries for infinite rotational symmetries")
    parser.add_argument("--no_rotationals", type=int, default=0, help="Whether to include rotational symmetric keypoints in the learning regime")
    parser.add_argument("--custom", type=int, default=0, help="CAUTION: this might execute unwanted code. Configure this in the code before and only use this for experiments")
    parser.add_argument("--sym_classes", type=int, nargs='+', default=None)
    parser.add_argument('--new_sym', type=int, default=0, help='use updated symmetries')
    parser.add_argument('--lr_scheduler', type=str, default="reduce", help="options: cyclic, reduce, ..")
    parser.add_argument("--lr_mode", type=str, default="", help="Learning rate mode, e.g. triangular2")
    parser.add_argument('--multi_instance', action='store_true')
    parser.add_argument('--gt_masks', action='store_true')
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument('--short_test', type=int, default=0, help="run a short test training with a small fraction of the dataset")
    parser.add_argument('--syn_train_data_ratio', type=float, default=0.0, help="ratio of synthetic data in addition to real data for multi-view training")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    increase_filedesc_limit()
    args = get_args()

    config = Config(ds_name=args.dataset)
    config.update(args)
    bs_utils = Basic_Utils(config)

    # config has to be created before these imports
    from utils.pvn3d_eval_utils_kpls import TorchEval
    from models.SyMFM6D import SyMFM6D
    from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_scape, eval_metric

    if os.name == 'posix':
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

    num_gpus = torch.cuda.device_count()
    on_cluster = num_gpus > 1

    print(torch.cuda.get_device_name(0))

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    backend = 'ddp'

    if backend == 'dp':
        args.batch_size_per_gpu *= num_gpus
        print("# Adapt batchsize with the gpu count: %i" % args.batch_size_per_gpu)

    config.mini_batch_size = args.batch_size_per_gpu
    config.val_mini_batch_size = args.batch_size_per_gpu
    config.n_total_epoch = args.epochs
    hparams = vars(args)

    config_dict = dict((name, getattr(config, name)) for name in dir(config) if not name.startswith('__'))
    hparams.update(config_dict)
    rndla_cfg = ConfigRandLA
    rndla_cfg_dict = dict((name, getattr(rndla_cfg, name)) for name in dir(rndla_cfg) if not name.startswith('__'))
    hparams.update(rndla_cfg_dict)

    # store some run information
    hparams['config'] = config
    hparams['rndla_cfg'] = rndla_cfg
    hparams['bs_utils'] = bs_utils
    hparams['viz'] = None
    hparams['git_id'] = get_current_git_hash()
    hparams['network'] = args.run_name
    hparams['dataset'] = args.dataset

    hparams['number_workers'] = args.workers_per_gpu
    hparams['gpus'] = num_gpus

    hparams['run_eval'] = args.run_eval
    hparams['floating16'] = False
    hparams['start_time'] = datetime.now()

    logger = TensorBoardLogger(save_dir=config.tb_dir, name=hparams['run_name'], version=args.logger_version)
    print(f"Logger Dir: {logger.log_dir}")
    print(f"Logger Name: {logger.name}")
    print(f"Logger Root Dir: {logger.root_dir}")
    print(f"Logger Save Dir: {logger.save_dir}")
    print(f"Logger Version: {logger.version}")

    hparams['logger_name'] = logger.name
    hparams['logger_version'] = 'version_' + str(logger.version)

    if args.checkpoint:
        print("checkpoint: " + args.checkpoint)
        model = SyMFM6DModule.load_from_checkpoint(args.checkpoint, strict=False)
        if num_gpus == 1:
            ckpt = torch.load(args.checkpoint)
            glob_step = ckpt['global_step']
            model.it = glob_step
        
        for k, v in hparams.items():
            model.hparams[k] = v
        if hparams['multi_view']:
            model.model.multi_view = True
            model.model.views = 3
        if hparams['symmetry']:  # to train non sym models as sym model
            model.criterion_of_sym = SymOFLoss()
            model.criterion_of_sym_multi = SymMultiOFLoss()
            model.model_fn = model_fn_decorator(model.criterion, model.criterion_of, model.criterion_of_sym, model.criterion_of_sym_multi, model.config, model.hparams)
        model.config = config
        model.bs_utils = bs_utils

        # backward compatibility
        if 'multi_view' not in model.hparams.keys():
            model.hparams['multi_view'] = False

        if 'symmetry' not in model.hparams.keys():
            model.hparams['symmetry'] = False

    else:
        print("Don't load checkpoint.")
        args.checkpoint = None
        model = SyMFM6DModule(hparams)

    print("Args: %s" % args)
    print("HParams: %s" % hparams)

    checkpoint_callback = lightning.callbacks.ModelCheckpoint(
        save_top_k=6,
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_last=True,
    )

    bnm_clip = 1e-2
    bnm_lmbd = lambda it: max(
        args.bn_momentum
        * args.bn_decay ** (int(it * config.mini_batch_size / args.decay_step)),
        bnm_clip,
    )

    if args.checkpoint:
        restore_optimizer = not args.new_optimizer
    else:
        restore_optimizer = None

    # float for fraction of dataset (1.0 = 100% of dataset)
    # int for specific number of batches (500 = 500 batches)
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    max_epochs = 1000  # config.n_total_epoch
    if args.short_test:
        limit_train_batches = 5  # 5
        limit_val_batches = 5  # 5
        limit_test_batches = 5
        max_epochs = 42  # 2

    trainer = lightning.Trainer(default_root_dir=config.tb_dir,
                                min_epochs=config.n_total_epoch,
                                max_epochs=max_epochs,
                                gpus=-1, logger=logger,
                                distributed_backend=backend, 
                                checkpoint_callback=True,
                                precision=16 if hparams['floating16'] else 32,
                                sync_batchnorm=True,
                                callbacks=[LRLoggingCallback(), 
                                           EarlyStopping(monitor="val/loss", mode="min", patience=10),
                                           ValidationIntervalScheduler(tot_iter=config.n_total_epoch*get_n_batches_per_epoch(args.dataset)//num_gpus, clr_div=2),
                                           BNMomentumScheduler(model.model, bnm_lmbd),
                                           checkpoint_callback],
                                plugins=[MyDDP(find_unused_parameters=True)],
                                deterministic=not args.non_deterministic,
                                resume_from_checkpoint=args.checkpoint,
                                amp_backend="apex",
                                amp_level=args.opt_level,
                                check_val_every_n_epoch=1,
                                num_sanity_val_steps=1,
                                log_every_n_steps=50,
                                accumulate_grad_batches=args.accumulate_grad_batches,
                                restore_optimizer=restore_optimizer,
                                limit_train_batches=limit_train_batches,
                                limit_val_batches=limit_val_batches,
                                limit_test_batches=limit_test_batches,
                                )
    if not args.run_eval:
        try:
            trainer.fit(model)
        except Exception as e:
            print("Error occurred in trainer.fit(model)!")
            traceback.print_exc()
            output_file = os.path.join(config.log_dir, 'error.log')
            with open(output_file, 'a+') as f:
                f.write(traceback.format_exc())
            raise e
        print("trainer.fit(model) done!")

        try:
            _ = config.cls_id
        except:
            config.cls_id = None

        try:
            trainer.test(model)
        except Exception as e:
            print("Error occurred in trainer.test(model)!")
            traceback.print_exc()
            output_file = os.path.join(config.log_dir, 'error.log')
            with open(output_file, 'a+') as f:
                f.write(traceback.format_exc())
            raise e
        print("trainer.test(model) done!")
    else:
        if args.checkpoint is None:
            raise ValueError("Please provide a checkpoint to test")
        try:
            _ = config.cls_id
        except:
            config.cls_id = None

        try:
            trainer.test(model)
        except Exception as e:
            print("Error occurred in trainer.test(model)!")
            traceback.print_exc()
            output_file = os.path.join(config.log_dir, 'error.log')
            with open(output_file, 'a+') as f:
                f.write(traceback.format_exc())
            raise e
        print("trainer.test(model) done!")
