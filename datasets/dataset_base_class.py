import os
import cv2
import torch
import matplotlib
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from utils.basic_utils import rand_range


class DatasetBase:
    def __init__(self, config, dataset_mode, multi_view, set_views, syn_train_data_ratio=0.0):
        self.config = config
        self.dataset_mode = dataset_mode
        self.multi_view = multi_view
        self.set_views = set_views

        self.dataset_mode = dataset_mode
        self.xmap = np.array([[j for _ in range(640)] for j in range(480)], dtype=np.int32)
        self.ymap = np.array([[i for i in range(640)] for _ in range(480)], dtype=np.int32)

        self.trancolor = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.syn_train_data_ratio = syn_train_data_ratio  # ratio of synthetic data in addition to real data for multi-view training

        if self.dataset_mode != 'train':
            self.add_noise = False
            self.add_depth_noise = False
            self.add_color_jitter = False

    def real_syn_gen(self):
        if not self.syn_train_data_ratio or self.rng.rand() > self.syn_train_data_ratio:
            item = self.real_gen()
        else:
            n = len(self.syn_lst)
            idx = self.rng.randint(0, n)
            item = self.syn_lst[idx]
        return item

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def gaussian_noise(self, rng, img, sigma):
        """
        add Gaussian noise of given sigma to image independently for each pixel
        :param rng: random generator
        :param img: image as numpy array e.g. (480, 640, 3)
        :param sigma: ing e.g. 4
        :return: image as numpy array e.g. (480, 640, 3)
        """
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """
        :param img: image as numpy array e.g. (480, 640, 3)
        :param angle: in degree, integer e.g. 84
        :param length: integer
        :return: image as numpy array e.g. (480, 640, 3)
        """
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img, name=None):
        """
        Add different types of noise to an RGB image, e.g. additive Gaussian noise, motion blur, sharpening, etc.
        :param img:
        :param name: file_name for debugging only
        :return:
        """
        rng = self.rng

        # apply HSV augmentor
        if rng.rand() <= self.config.rgb_hsv_augment:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() <= self.config.rgb_sharpen_ratio:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() <= self.config.rgb_motion_blur_ratio:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() <= self.config.rgb_gaussian_blur_ratio:
            if rng.rand() <= self.config.rgb_gaussian_blur_small_ratio:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        # add Gaussian noise of fixed sigma [0, 15/25) to image independently for each pixel
        if rng.rand() <= self.config.rgb_gaussian_noise_small_ratio:
            img = self.gaussian_noise(rng, img, sigma=rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, sigma=rng.randint(25))

        if rng.rand() <= self.config.rgb_normal_noise_ratio:
            # add Gaussian noise of sigma=7 to image independently for each pixel
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def depth_img_add_noise(self, img, name=None):
        """
        Add different types of noise to a depth image, e.g. additive Gaussian noise
        :param img: NumPy, depth image e.g. (480, 640)
        :param name: file_name for debugging only
        :return:
        """
        rng = self.rng

        if rng.rand() <= self.config.depth_normal_noise_ratio:
            # add Gaussian noise of sigma=7 to image independently for each pixel
            img += np.random.normal(loc=0.0, scale=self.config.depth_normal_noise_scale, size=img.shape).astype(np.float32)

        return np.clip(img, 0, 1)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        """
            #     Loads a real image, removes all objects and places the objects of the synthetic input image in front of the real background
            #     :param rgb: synthetic RGB image, uint8 (480, 640, 3)
            #     :param labels: corresponding semantic segmentation label, uint8 (480, 640)
            #     :param dpt: corresponding depth image, int32 (480, 640)
            #     :param dpt_msk: corresponding depth mask, bool (480, 640) -> indicates where dpt has values > 1e-6  # TODO: Why are they useful?
            #     :return: rgb (480, 640, 3), dpt float64 (480, 640)
            #     """
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item + '-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K):
        """
        converts a depth image into a depth_xyz map where at each pixel the corresponding 3d location is stored!?
        :param dpt: depth image numpy array (h, w) e.g. (480, 640)
        :param cam_scale: scalar e.g. 1 or 10000
        :param K: camera intrinsic matrix, numpy array (3, 3)
        :return: depth_xyz map
        """
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def __len__(self):
        if self.multi_view:
            try:
                return len(self.sequence_samples)
            except AttributeError:
                return len(self.pp_data)
        return len(self.all_lst)

    def get_sequence(self, sequence_ids):
        sequence = []
        for sequence_id in sequence_ids:
            sequence.append(self.get_item(sequence_id))
        return sequence

    def transform_offsets(self, camchange, origin, transformed_origin, offset):
        offset_points = origin + offset
        new_offset_points = self.transform_data(camchange, offset_points)
        new_offsets = new_offset_points - transformed_origin
        return new_offsets
