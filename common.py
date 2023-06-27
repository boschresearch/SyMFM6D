#!/usr/bin/env python3
import os
import yaml
import numpy as np


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


class ConfigRandLA:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 480 * 640 // 24  # Number of input points
    num_classes = 22  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size_per_gpu = 3  # batch_size_per_gpu during training
    val_batch_size = 3  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch
    in_c = 9

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [32, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]


class Singleton(type):
    _instances = {}

    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super().__call__(*args, **kwargs)
        return self._instances[self]


class Config(metaclass=Singleton):
    def __init__(self, ds_name='ycb'):
        self.dataset_name = ds_name

        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)
        self.resnet_ptr_mdl_p = "/path/to/resnet34-333f7ec4.pth"

        # log folder
        self.log_dir = os.path.abspath(os.path.join(self.exp_dir, 'train_log', self.dataset_name))
        ensure_fd(self.log_dir)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        ensure_fd(self.checkpoint_dir)
        self.log_eval_dir = os.path.join(self.log_dir, 'eval_results')
        ensure_fd(self.log_eval_dir)
        self.tb_dir = os.path.join(self.log_dir, 'train_info')
        ensure_fd(self.tb_dir)

        self.n_total_epoch = 25
        self.mini_batch_size = 3
        self.val_mini_batch_size = 3
        self.test_mini_batch_size = 1

        self.n_sample_points = 480 * 640 // 24
        self.n_keypoints = 8
        self.n_min_points = 400

        # RGB Augmentation parameters
        self.rgb_add_noise_twice_ratio = 0.2  # probability that the training color image will be disturbed a second time
        self.rgb_hsv_augment = 1.0
        self.rgb_sharpen_ratio = 0.2
        self.rgb_motion_blur_ratio = 0.2
        self.rgb_gaussian_blur_ratio = 0.2
        self.rgb_gaussian_blur_small_ratio = 0.8
        self.rgb_gaussian_noise_small_ratio = 0.8
        self.rgb_normal_noise_ratio = 0.2

        self.add_real_back = 0.0  # add a real background to the synthetic data
        self.depth_normal_noise_ratio = 1.0
        self.depth_normal_noise_scale = 0.03

        if self.dataset_name == 'ycb':
            print('Config variant ', self.dataset_name)
            self.n_objects = 21 + 1  # 21 objects + background
            self.n_classes = self.n_objects
            self.use_orbfps = True
            self.kp_orbfps_dir = os.path.abspath(os.path.join(self.exp_dir, 'datasets/ycb/ycb_kps/'))
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            self.cls_lst_p = os.path.abspath(os.path.join(self.exp_dir, 'datasets/ycb/dataset_config/classes.txt'))
            self.root = '/path/to/YCB_Video_Dataset'
            self.models_dir = "/path/to/YCB_Video_Dataset/models"
            self.kps_dir = os.path.abspath(os.path.join(self.exp_dir, 'datasets/ycb/ycb_kps/'))
            r_lst_p = os.path.abspath(os.path.join(self.exp_dir, 'datasets/ycb/dataset_config/radius.txt'))
            self.r_lst = list(np.loadtxt(r_lst_p))
            self.cls_lst = self.read_lines(self.cls_lst_p)
            self.sym_cls_ids = [13, 16, 19, 20, 21]
            self.rgb_filetype = "png"

        elif self.dataset_name == 'SymMovCam':
            print('Config variant ', self.dataset_name)
            self.n_objects = 21 + 1  # 21 objects + background
            self.n_classes = self.n_objects

            self.n_scenes = 8333
            self.cls_lst_p = "/path/to/SymMovCam/dataset_config/classes_with_numbers.txt"
            self.root = "/path/to/SymMovCam"
            self.models_dir = "/path/to/YCB_Video_Dataset/models"
            self.kps_dir = "/path/to/SymMovCam/obj_kps/"
            r_lst_p = "/path/to/SymMovCam/dataset_config/radius.txt"
            self.symmetries = "/path/to/SymMovCam/dataset_config/symmetries.txt"

            self.radius_lst = list(np.loadtxt(r_lst_p))
            self.cls_lst = self.read_lines(self.cls_lst_p)
            self.sym_cls_ids = [13, 16, 19, 20, 21]  # index starting at 1
            self.segment_filetype = "png"
            self.dpt_filetype = "png"
            self.rgb_filetype = "jpg"

        self.intrinsic_matrix = {
            'blender': np.array([[700.,     0.,     320.],
                                 [0.,       700.,   240.],
                                 [0.,       0.,     1.]]),
            'ycb_K1': np.array([[1066.778, 0.        , 312.9869],
                                [0.      , 1067.487  , 241.3109],
                                [0.      , 0.        , 1.0]], np.float32),
            'ycb_K2': np.array([[1077.836, 0.        , 323.7872],
                                [0.      , 1078.189  , 279.6921],
                                [0.      , 0.        , 1.0]], np.float32),
            'scape':  np.array([[497.77777777777777, 0., 320.],
                                [0., 497.77777777777777, 240.],
                                [0., 0.                , 1.]], np.float32)
        }

    def update(self, args):
        # log folder
        if args.out_dir:
            self.out_dir = args.out_dir
            self.log_dir = args.out_dir

            ensure_fd(self.log_dir)
            self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
            ensure_fd(self.checkpoint_dir)
            self.log_eval_dir = os.path.join(self.log_dir, 'eval_results')
            ensure_fd(self.log_eval_dir)
            self.tb_dir = os.path.join(self.log_dir, 'tb')
            ensure_fd(self.tb_dir)

    @staticmethod
    def read_lines(p):
        with open(p, 'r') as f:
            return [line.strip() for line in f.readlines()]
