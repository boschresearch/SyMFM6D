import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet import PSPNet
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet
import einops


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


class SyMFM6D(nn.Module):
    def __init__(self, n_classes, n_pts, rndla_cfg, n_kps=8, multi_view=False, symmetry=False):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        self.multi_view = multi_view
        self.views = 1 if not multi_view else 3
        self.symmetry = symmetry
        cnn = psp_models['resnet34'.lower()]()

        rndla = RandLANet(rndla_cfg)

        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool
        )
        self.rndla_pre_stages = rndla.fc0

        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,
            cnn.feats.layer2,
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)
        ])
        self.ds_sr = [4, 8, 8, 8]

        self.rndla_ds_stages = rndla.dilated_res_blocks

        self.ds_rgb_oc = [64, 128, 512, 1024]
        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out]
        self.ds_fuse_r2p_pre_layers = nn.ModuleList()
        self.ds_fuse_r2p_fuse_layers = nn.ModuleList()
        self.ds_fuse_p2r_pre_layers = nn.ModuleList()
        self.ds_fuse_p2r_fuse_layers = nn.ModuleList()
        self.ds_compress_r2p_layers = nn.ModuleList()

        for i in range(4):
            self.ds_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i]*2, self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.ds_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i]*2, self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_compress_r2p_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),
            nn.Sequential(cnn.up_2, cnn.drop_2),
            nn.Sequential(cnn.final),
            nn.Sequential(cnn.up_3, cnn.final)
        ])
        self.up_rgb_oc = [256, 64, 64]
        self.up_rndla_oc = []
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_pre_layers = nn.ModuleList()
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_pre_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        self.up_compress_r2p_layers = nn.ModuleList()

        for i in range(n_fuse_layer):
            self.up_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.up_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_compress_r2p_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

        self.rgbd_seg_layer = (
            pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_kps*3, activation=None)
        )

        if self.symmetry:
            self.sym_kp_ofst_layer = (
                pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(n_kps*3, activation=None)
            )

    @staticmethod
    def gather_and_maxpool(feature, pool_idx):
        """
        Gathers the features at pool_idx (nearest neighbor indices) and performs max pooling over the k nearest neighbors.

        :param feature: input features matrix [B, N, d]
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
            e.g. [n_views=3, 3200, 16]
        :return: pool_features = [B, N', d] pooled features matrix

        """
        if len(feature.size()) > 3:  # bs*c*p*1
            feature = feature.squeeze(dim=3)  # (batch, channel, n_points)
        n_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(n_points,n_samples)
        pool_features = torch.gather(
            input=feature, dim=2, index=pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, n_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*n_points*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*n_points*1
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            input=feature, dim=2, index=interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)
        return interpolated_features  # batch*channel*n_points*1

    @staticmethod
    def nearest_interpolation_multi(feature, interp_idx):
        """
        :param feature: [B, V, N, d] input features matrix
        :param interp_idx: [B, V, up_num_points, 1] nearest neighbour index
        :return: [B, V*up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=-1)  # batch*channel*(v*n_points)*1 -> batch*channel*(v*n_points)
        
        batch_size = interp_idx.shape[0]
        view_size = interp_idx.shape[1]
        up_num_points = interp_idx.shape[2]
        interp_idx = interp_idx.reshape(batch_size, view_size, up_num_points)
        feature = feature.unsqueeze(1).repeat(1, interp_idx.shape[1], 1, 1)
        interp_idx = interp_idx.unsqueeze(2).repeat(1, 1, feature.shape[2], 1)  # bs*v*c*n_points
        interpolated_features = torch.gather(
            input=feature, dim=3, index=interp_idx
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(-1)  # batch*channel*n_points*1
        return interpolated_features

    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features

    def forward(self, inputs, end_points=None, scale=1,):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, n_pts]
            choose      : LongTensor [bs, 1, n_pts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        rgb_emb, p_emb = self.pre_encode(inputs)

        # ###################### encoding stages #############################
        n_down_layers = 4
        ds_emb, rgb_emb, p_emb = self.encode(inputs, rgb_emb, p_emb, n_down_layers)

        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages)
        bs, rgbd_emb = self.decode(inputs, ds_emb, rgb_emb, p_emb, n_up_layers)

        # ###################### prediction stages #############################
        if self.symmetry:
            rgbd_segs, pred_kp_ofs, pred_ctr_ofs, pred_sym_kp_ofs = self.prediction(bs, rgbd_emb, symmetry=True)
        else:
            rgbd_segs, pred_kp_ofs, pred_ctr_ofs = self.prediction(bs, rgbd_emb)

        end_points['pred_rgbd_segs'] = rgbd_segs
        end_points['pred_kp_ofs'] = pred_kp_ofs
        end_points['pred_ctr_ofs'] = pred_ctr_ofs
        if self.symmetry:
            end_points['pred_sym_kp_ofs'] = pred_sym_kp_ofs

        return end_points

    def pre_encode(self, inputs):
        # ResNet pre + layer1 + layer2
        if self.multi_view:
            self.views = inputs['rgb'].shape[1]
            inputs['rgb'] = einops.rearrange(inputs['rgb'], 'b v c h w -> (b v) c h w')
        
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])

        # rndla pre
        p_emb = inputs['cld_rgb_nrm']
        p_emb = self.rndla_pre_stages(p_emb)
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*n_points*1

        return rgb_emb, p_emb

    def encode(self, inputs, rgb_emb, p_emb, n_down_layers):
        ds_emb = []
        for i_ds in range(n_down_layers):

            # encode rgb downsampled feature
            rgb_emb0 = self.cnn_ds_stages[i_ds](rgb_emb)  # (b, c, w, h)

            bs, c, hr, wr = rgb_emb0.size()
            bs = bs // self.views

            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            )

            f_sampled_i = self.gather_and_maxpool(
                feature=f_encoder_i,
                pool_idx=inputs['cld_sub_idx%d' % i_ds]
            )
            p_emb0 = f_sampled_i
            if i_ds == 0:
                ds_emb.append(f_encoder_i)

            # point-to-pixel fusion
            p2r_emb = self.ds_fuse_p2r_pre_layers[i_ds](p_emb0)
            if self.multi_view:
                p2r_emb = self.nearest_interpolation_multi(p2r_emb,  inputs['p2r_ds_nei_idx%d' % i_ds])
            else:
                p2r_emb = self.nearest_interpolation(p2r_emb, inputs['p2r_ds_nei_idx%d' % i_ds])
            
            p2r_emb = p2r_emb.view(self.views * bs, -1, hr, wr)

            rgb_emb = self.ds_fuse_p2r_fuse_layers[i_ds](  # MLP_f + ReLU
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # pixel-to-point fusion
            if self.multi_view:
                n_sample_pts = inputs['r2p_ds_nei_idx%d' % i_ds].shape[-2]
                r2p_emb = self.gather_and_maxpool(

                    feature=rgb_emb0.reshape(bs * self.views, c, hr * wr, 1),

                    pool_idx=inputs['r2p_ds_nei_idx%d' % i_ds].view(bs * self.views, n_sample_pts, -1)
                ).view(bs, self.views, c, n_sample_pts, 1) \
                    .permute(0, 2, 1, 3, 4) \
                    .reshape(bs, c, n_sample_pts * self.views, 1) \
                    .contiguous()
            else:
                r2p_emb = self.gather_and_maxpool(
                    feature=rgb_emb0.reshape(bs*self.views, c, hr * wr, 1),
                    pool_idx=inputs['r2p_ds_nei_idx%d' % i_ds]
                ).view(bs, c, -1, 1)

            r2p_emb = self.ds_fuse_r2p_pre_layers[i_ds](r2p_emb)

            p_emb = self.ds_fuse_r2p_fuse_layers[i_ds](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )
            ds_emb.append(p_emb)
        
        return ds_emb, rgb_emb, p_emb

    def decode(self, inputs, ds_emb, rgb_emb, p_emb, n_up_layers):
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb0 = self.cnn_up_stages[i_up](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()
            bs = bs//self.views

            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i

            # point-to-pixel fusion
            p2r_emb = self.up_fuse_p2r_pre_layers[i_up](p_emb0)
            if self.multi_view:
                p2r_emb = self.nearest_interpolation_multi(p2r_emb,  inputs['p2r_up_nei_idx%d' % i_up])
            else:
                p2r_emb = self.nearest_interpolation(
                    p2r_emb, inputs['p2r_up_nei_idx%d' % i_up]
                )
            p2r_emb = p2r_emb.view(self.views*bs, -1, hr, wr)

            rgb_emb = self.up_fuse_p2r_fuse_layers[i_up](
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # pixel-to-point fusion
            if self.multi_view:
                n_sample_pts = inputs['r2p_up_nei_idx%d' % i_up].shape[-2]
                r2p_emb = self.gather_and_maxpool(
                    feature=rgb_emb0.reshape(bs * self.views, c, hr * wr, 1),
                    pool_idx=inputs['r2p_up_nei_idx%d' % i_up].view(bs * self.views, n_sample_pts, -1)
                ).view(bs, self.views, c, n_sample_pts, 1) \
                    .permute(0, 2, 1, 3, 4) \
                    .reshape(bs, c, n_sample_pts * self.views, 1) \
                    .contiguous()
            else:
                r2p_emb = self.gather_and_maxpool(
                    feature=rgb_emb0.reshape(bs * self.views, c, hr * wr),
                    pool_idx=inputs['r2p_up_nei_idx%d' % i_up]
                ).view(bs, c, -1, 1)
            r2p_emb = self.up_fuse_r2p_pre_layers[i_up](r2p_emb)
            p_emb = self.up_fuse_r2p_fuse_layers[i_up](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )

        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1)

        bs, di, _, _ = rgb_emb.size()
        bs = bs//self.views

        if self.multi_view:
            rgb_emb_c = rgb_emb.view(bs, self.views, di, -1)
            choose_emb = inputs['choose'].repeat(1, 1, di, 1)
            rgb_emb_c = torch.gather(rgb_emb_c, 3, choose_emb)
            rgb_emb_c = rgb_emb_c.transpose(2, 1).reshape(bs, di, -1).contiguous()
        else:
            rgb_emb_c = rgb_emb.view(bs, di, -1)
            choose_emb = inputs['choose'].repeat(1, di, 1)
            rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()

        # Use DenseFusion in final layer, which will hurt performance due to overfitting
        # rgbd_emb = self.fusion_layer(rgb_emb, pcld_emb)

        # Use simple concatenation. Good enough for fully fused RGBD feature.
        rgbd_emb = torch.cat([rgb_emb_c, p_emb], dim=1)

        return bs, rgbd_emb

    def prediction(self, bs, rgbd_emb, symmetry = False):
        rgbd_segs = self.rgbd_seg_layer(rgbd_emb)
        pred_kp_ofs = self.kp_ofst_layer(rgbd_emb)
        pred_ctr_ofs = self.ctr_ofst_layer(rgbd_emb)

        pred_kp_ofs = pred_kp_ofs.view(
            bs, self.n_kps, 3, -1
        ).permute(0, 1, 3, 2).contiguous()

        pred_ctr_ofs = pred_ctr_ofs.view(
            bs, 1, 3, -1
        ).permute(0, 1, 3, 2).contiguous()
        if symmetry:
            pred_sym_kp_ofs = self.sym_kp_ofst_layer(rgbd_emb).view(
            bs, self.n_kps, 3, -1
            ).permute(0, 1, 3, 2).contiguous()
            return rgbd_segs, pred_kp_ofs, pred_ctr_ofs, pred_sym_kp_ofs

        else:
            return rgbd_segs, pred_kp_ofs, pred_ctr_ofs


# Copy from PVN3D: https://github.com/ethnhe/PVN3D
class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(32, 256, 1)

        self.conv3 = torch.nn.Conv1d(96, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1)
