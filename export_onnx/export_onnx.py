import argparse
import mmcv
import os
import numpy as np
import torch
import warnings
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset, build_dataloader_sequential
from torch import nn
from mmdet3d.models.detectors import VoteSegmentor
from torch.nn import functional as F
from copy import deepcopy
from mmcv.cnn import build_conv_layer
from funcs import layer_fusion
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.ops import scatter_sst, Connected_componet, Scatter_bool
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
import exptool


torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


def pre_voxelize(model, points, batch_idx):
    voxel_size = torch.tensor(model.cfg.pre_voxelization_size, device=batch_idx.device)
    pc_range = torch.tensor(model.cluster_assigner.point_cloud_range, device=points.device)
    coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
    coors = coors[:, [2, 1, 0]]  # to zyx order
    coors = torch.cat([batch_idx[:, None], coors], dim=1)
    return coors

def infer_voxelize(points, voxel_layer, encoder):
    coors = []
    coor_indexs = []
    grid_size = voxel_layer.grid_size
    # dynamic voxelization only provide a coors mapping
    for res in points:
        res_coors = voxel_layer(res)
        coors.append(res_coors)
    points = torch.cat(points, dim=0)
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coor_index = ((coor_pad[:, 0] * grid_size[2] + coor_pad[:, 1]) * grid_size[1] + coor_pad[:, 2]) * grid_size[0] + coor_pad[:, 3]
        coor_indexs.append(coor_index)
        coors_batch.append(coor_pad)
    coors = torch.cat(coors_batch, dim=0)
    coor_indexs = torch.cat(coor_indexs, dim=0)
    coor_index0, indice = torch.sort(coor_index)
    coors = coors[indice]
    points = points[indice]
    f_center = points.new_zeros(size=(points.size(0), 3))
    f_center[:, 0] = points[:, 0] - (
            coors[:, 3].type_as(points) * encoder.vx + encoder.x_offset)
    f_center[:, 1] = points[:, 1] - (
            coors[:, 2].type_as(points) * encoder.vy + encoder.y_offset)
    f_center[:, 2] = points[:, 2] - (
            coors[:, 1].type_as(points) * encoder.vz + encoder.z_offset)

    return points, coors, f_center

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--lidarbin', default='data/waymo/kitti_format/training/velodyne1000000.bin',
        help='single lidar frame data saved in bin file')
    parser.add_argument('--infer', default=True, help='wheter to run with infer mode')
    parser.add_argument('--export', default=True, help='wheter to run with infer mode')
    parser.add_argument('--fuse-conv-bn', default=False,
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    args = parser.parse_args()
    return args

def load_data(path, point_range, load_dim, use_dim):
    with open(path, 'rb') as f:
        points = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, load_dim)
        points = points[:, :use_dim]
        points = torch.as_tensor(points).cuda()
        in_range_flags = ((points[:, 0] > point_range[0])
                          & (points[:, 1] > point_range[1])
                          & (points[:, 2] > point_range[2])
                          & (points[:, 0] < point_range[3])
                          & (points[:, 1] < point_range[4])
                          & (points[:, 2] < point_range[5]))
        points = points[in_range_flags]
        return points


class segment_voxel_encoder(nn.Module):
    def __init__(self, model):
        super(segment_voxel_encoder, self).__init__()
        self.voxel_encoder = model.segmentor.voxel_encoder
    def forward(self, points_xyzs, points_feat, in_coors, centers):
        batch_points = points_xyzs
        coors = in_coors
        # coors = coors.long()
        voxel_features, voxel_coors, voxel2point_inds, count = self.voxel_encoder(batch_points, points_feat, coors, centers,
                                                                           return_inv=True)
        return voxel_features, voxel_coors, voxel2point_inds

class segment_backbone(nn.Module):
    def __init__(self, model):
        super(segment_backbone, self).__init__()
        self.backbone = deepcopy(model.segmentor.backbone)
    def forward(self, feat, coords):
        voxel_info = {}
        voxel_info['voxel_coors'] = coords
        voxel_info['voxel_feats'] = feat
        return self.backbone(voxel_info)

class segment_backbone_modified(nn.Module):
    def __init__(self, model):
        super(segment_backbone_modified, self).__init__()
        backbone = deepcopy(model.segmentor.backbone)
        self.backbone = layer_fusion(backbone)
        num_layers = len(self.backbone.decoder_channels)
        conv_cfg = {'type': 'SparseConv3d', 'indice_key': 'subm1'}
        for i in range(num_layers):
            i += 1
            tmp_merge = getattr(self.backbone, f'merge_layer{i}')
            device = tmp_merge.weight.device
            dtype = tmp_merge.weight.dtype
            in_channels = tmp_merge.out_channels
            out_channels = tmp_merge.in_channels
            conv_cfg['indice_key'] = tmp_merge.indice_key
            conv = build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                1,
                stride=1,
                padding=0,
                bias=False)
            conv.weight.to(device=device, dtype=dtype)
            weight_shape = conv.weight.shape
            weight = torch.cat([torch.eye(in_channels), torch.zeros((in_channels, in_channels))], dim = 1)
            weight = weight.to(device=device, dtype=dtype)
            weight = weight.view(weight_shape)
            with torch.no_grad():
                conv.weight =torch.nn.Parameter(weight)
            setattr(self.backbone, f'merge_layer_cat{i}_0', conv)
            conv = deepcopy(conv)
            weight = torch.cat([torch.zeros((in_channels, in_channels)), torch.eye(in_channels)], dim = 1)
            weight = weight.to(device=device, dtype=dtype)
            weight = weight.view(weight_shape)
            with torch.no_grad():
                conv.weight = torch.nn.Parameter(weight)
            setattr(self.backbone, f'merge_layer_cat{i}_1', conv)
            rinc = out_channels
            routc = rinc // 2
            conv = build_conv_layer(
                conv_cfg,
                rinc,
                routc,
                1,
                stride=1,
                padding=0,
                bias=False)

            conv.weight.to(device=device, dtype=dtype)
            weight = torch.zeros((rinc, routc)).to(device=device, dtype=dtype)
            for j in range(routc):
                weight[j * 2:j * 2 + 2, j] = 1.0
            weight = weight.view(conv.weight.shape)
            with torch.no_grad():
                conv.weight = torch.nn.Parameter(weight)
            setattr(self.backbone, f'reduce_channel{i}', conv)

    def numpy(self, a):
        return a.detach().cpu().numpy()

    def decoder_layer_forward(self, x_lateral, x_bottom, lateral_layer,
                              merge_layer, upsample_layer, cat0_layer,
                              cat1_layer, reduce_channel_layer):
        """Forward of upsample and residual block.

        Args:
            x_lateral (:obj:`SparseConvTensor`): Lateral tensor.
            x_bottom (:obj:`SparseConvTensor`): Feature from bottom layer.
            lateral_layer (SparseBasicBlock): Convolution for lateral tensor.
            merge_layer (SparseSequential): Convolution for merging features.
            upsample_layer (SparseSequential): Convolution for upsampling.

        Returns:
            :obj:`SparseConvTensor`: Upsampled feature.
        """
        x = lateral_layer(x_lateral)
        cat0 = cat0_layer(x_bottom)
        cat1 = cat1_layer(x)
        x = cat1.replace_feature(cat0.features + cat1.features)
        x_merge = merge_layer(x)
        x = reduce_channel_layer(x)
        x = x.replace_feature(x_merge.features + x.features)
        x = upsample_layer(x)
        return x

    def forward(self, feat, coords):
        voxel_info = {}
        voxel_info['voxel_coors'] = coords
        voxel_info['voxel_feats'] = feat

        coors = voxel_info['voxel_coors']
        if self.backbone.ndim == 2:
            assert (coors[:, 1] == 0).all()
            coors = coors[:, [0, 2, 3]]  # remove the z-axis indices
        if self.backbone.keep_coors_dims is not None:
            coors = coors[:, self.backbone.keep_coors_dims]
        voxel_features = voxel_info['voxel_feats']
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.backbone.sparse_shape, 1)
        x = self.backbone.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.backbone.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        x = encode_features[-1]
        for i in range(self.backbone.stage_num, 0, -1):
            x = self.decoder_layer_forward(encode_features[i - 1], x,
                                           getattr(self.backbone, f'lateral_layer{i}'),
                                           getattr(self.backbone, f'merge_layer{i}'),
                                           getattr(self.backbone, f'upsample_layer{i}'),
                                           getattr(self.backbone, f'merge_layer_cat{i}_0'),
                                           getattr(self.backbone, f'merge_layer_cat{i}_1'),
                                           getattr(self.backbone, f'reduce_channel{i}')
                                           )
            # decode_features.append(x)
        ret = {'voxel_feats': x.features}
        ret = [ret, ]  # keep consistent with SSTv2

        return ret


class segment_decode(nn.Module):
    def __init__(self, model):
        super(segment_decode, self).__init__()
        self.decode_neck = deepcopy(model.segmentor.decode_neck)
        self.segmentation_head = deepcopy(model.segmentor.segmentation_head)
        self.num_classes = model.num_classes
        self.cfg = model.cfg
        self.cluster_voxel_size = model.cluster_assigner.cluster_voxel_size
        cluster_voxel_size_inverse = {}
        for k, v in self.cluster_voxel_size.items():
            cluster_voxel_size_inverse[k] = torch.Tensor([1.0 / a for a in v]).cuda().float().view(1, 3)
        self.cluster_voxel_size_inverse = cluster_voxel_size_inverse
        self.min_points = model.cluster_assigner.min_points
        self.connected_dist = model.cluster_assigner.connected_dist
        self.class_names = model.cluster_assigner.class_names
        self.scatter_infer = scatter_sst()
        self.collect_component = Connected_componet()
        self.point_cloud_range = model.cluster_assigner.point_cloud_range
        self.point_cloud_down_range = torch.Tensor(model.cluster_assigner.point_cloud_range[:3]).view(1, 3).cuda()
        self.point_cloud_up_range = torch.Tensor(model.cluster_assigner.point_cloud_range[3:]).view(1, 3).cuda()
        self.backbone = deepcopy(model.backbone)
        self.bbox_head = deepcopy(model.bbox_head)
        self.as_rpn = model.as_rpn
        self.scatter_bool = Scatter_bool()
    def numpy(self, a):
        return a.detach().cpu().numpy()

    def get_fg_mask(self, seg_scores, cls_id):

        seg_scores = seg_scores[:, cls_id]
        cls_score_thr = self.cfg['score_thresh'][cls_id]
        fg_mask = seg_scores > cls_score_thr
        return fg_mask

    def update_sample_results_by_mask(self, sampled_out, valid_mask_list):
        for k in sampled_out:
            old_data = sampled_out[k]
            if len(old_data[0]) == valid_mask_list[0][1] or 'fg_mask' in k:
                if 'fg_mask' in k:
                    new_data_list = []
                    for data, mask in zip(old_data, valid_mask_list):
                        new_data = data.clone()
                        index = self.scatter_bool(new_data)
                        new_data[index] = mask[2]
                        assert new_data.sum() == mask[2].sum()
                        new_data_list.append(new_data)
                    sampled_out[k] = new_data_list
                else:
                    new_data_list = [data[mask[0]] for data, mask in zip(old_data, valid_mask_list)]
                    sampled_out[k] = new_data_list
        return sampled_out

    def clusster_assign(self, points, batch_idx, class_name, i):
        # cluster_vsize = self.cluster_voxel_size[class_name]
        #
        # voxel_size = torch.tensor(cluster_vsize, device=points.device)
        # pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        # coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int()

        cluster_vsize = self.cluster_voxel_size_inverse[class_name]
        pc_range = self.point_cloud_down_range
        coors = ((points - pc_range) * cluster_vsize).floor().int()

        # coors = points - pc_range
        # coors = coors[:, [2, 1, 0]] # to zyx order
        # coors = F.pad(coors, (1, 0), mode='constant', value=0)
        coors = torch.cat([batch_idx, coors], dim=1)

        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask_bool = cnt_per_point >= self.min_points
        valid_mask_len = len(valid_mask_bool)
        valid_mask = self.scatter_bool(valid_mask_bool)
        points = points[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv_once, count = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        num_points = count.cpu().numpy().shape[0]
        num_points = torch.Tensor([num_points]).to(device=count.device, dtype=count.dtype)
        sampled_centers = self.scatter_infer(points, unq_inv_once, count, num_points, 'mean')
        voxel_coors = new_coors
        # sampled_centers, voxel_coors = scatter_v2(points, coors, mode='avg', return_inv=False,
        #                                                     new_coors=new_coors, unq_inv=unq_inv_once)
        inv_inds = unq_inv_once
        dist = self.connected_dist[class_name]
        cluster_inds = self.collect_component(sampled_centers, voxel_coors[:, 0], dist).unsqueeze(dim=1)
        cluster_inds_per_point = cluster_inds[inv_inds]
        batch_idx = batch_idx[valid_mask]
        # clsid = torch.ones_like(batch_idx, dtype=batch_idx.dtype, device=batch_idx.device) * i
        cluster_inds_per_point = torch.cat([batch_idx + i, batch_idx, cluster_inds_per_point], dim=1)
        return cluster_inds_per_point, [valid_mask, valid_mask_len, valid_mask_bool]

    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0)
        return out_dict

    def extract_feat(self, points_xyzs, points_feat, pts_feats, pts_cluster_inds, center_preds):
        """Extract features from points."""
        new_coors, unq_inv, count = torch.unique(pts_cluster_inds, return_inverse=True, return_counts=True, dim=0)
        num_points = count.cpu().numpy().shape[0]
        num_points = torch.Tensor([num_points]).to(device=count.device, dtype=count.dtype)
        cluster_xyz = self.scatter_infer(center_preds, unq_inv, count, num_points, 'mean')

        # cluster_xyz, _= scatter_v2(center_preds, pts_cluster_inds, mode='avg', return_inv=False,
        #                                       new_coors=new_coors, unq_inv=unq_inv)
        inv_inds = unq_inv
        f_cluster = points_xyzs - cluster_xyz[inv_inds]

        out_pts_feats, cluster_feats, out_coors = self.backbone(points_xyzs, points_feat, pts_feats, pts_cluster_inds, f_cluster)
        out_dict = dict(
            cluster_feats=cluster_feats,
            cluster_xyz=cluster_xyz,
            cluster_inds=out_coors
        )
        if self.as_rpn:
            out_dict['cluster_pts_feats'] = out_pts_feats
            out_dict['cluster_points_xyzs'] = points_xyzs
            out_dict['cluster_points_feat'] = points_feat

        return out_dict

    def forward(self, points_xyzs, points_feat, pts_coors, pts_centers, voxel_feats, voxel2point_inds, decoder_coors, batch_idx):
        pts_feats = voxel_feats[
            voxel2point_inds]  # voxel_feats must be the output of torch_scatter, voxel2point_inds is the input of torch_scatter
        neg_one = torch.tensor(-1.0, device=pts_feats.device, dtype=pts_feats.dtype)
        pts_mask = torch.all((pts_feats != neg_one), dim=1)
        pts_mask = self.scatter_bool(pts_mask)
        pts_feats = pts_feats[pts_mask]
        points_xyzs = points_xyzs[pts_mask]
        points_feat = points_feat[pts_mask]
        local_xyz = pts_centers[pts_mask]

        feats = torch.cat([pts_feats, local_xyz], 1)

        seg_logits, vote_preds = self.segmentation_head.forward(feats)
        offsets = vote_preds * vote_preds.abs()

        decoder_coors = decoder_coors[pts_mask]
        decoder_coors, decoder_unq_inv, count = torch.unique(decoder_coors, return_inverse=True, return_counts=True, dim=0)
        num_points = count.cpu().numpy().shape[0]
        num_points = torch.Tensor([num_points]).to(device=count.device, dtype=count.dtype)
        seg_points_xyzs = self.scatter_infer(points_xyzs, decoder_unq_inv, count, num_points, 'mean')
        seg_points_feat = self.scatter_infer(points_feat, decoder_unq_inv, count, num_points, 'mean')
        seg_logits = self.scatter_infer(seg_logits, decoder_unq_inv, count, num_points, 'mean')
        seg_vote_preds = self.scatter_infer(vote_preds, decoder_unq_inv, count, num_points, 'mean')
        seg_feats = self.scatter_infer(feats, decoder_unq_inv, count, num_points, 'mean')
        vote_offsets = self.scatter_infer(offsets, decoder_unq_inv, count, num_points, 'mean')
        seg_scores = seg_logits.sigmoid()
        offset = vote_offsets.reshape(-1, self.num_classes, 3)
        fg_mask_list = []  # fg_mask of each cls
        center_preds_list = []
        for cls in range(self.num_classes):
            fg_mask = self.get_fg_mask(seg_scores, cls)
            cls_offset = offset[:, cls, :]
            bool_indice = self.scatter_bool(fg_mask)
            fg_mask_list.append([fg_mask, bool_indice])
            this_offset = cls_offset[bool_indice]
            this_points = seg_points_xyzs[bool_indice]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)
        output_dict = {}
        dict_to_sample = {}
        dict_to_sample['seg_points_xyzs'] = seg_points_xyzs
        dict_to_sample['seg_points_feat'] = seg_points_feat
        dict_to_sample['seg_logits'] = seg_logits
        dict_to_sample['seg_vote_preds'] = seg_vote_preds
        dict_to_sample['seg_feats'] = seg_feats
        dict_to_sample['vote_offsets'] = vote_offsets
        dict_to_sample['batch_idx'] = batch_idx
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask[1]])
            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = [a[0] for a in fg_mask_list]
        output_dict['center_preds'] = center_preds_list
        cluster_inds_list, valid_mask_list = [], []
        for i, class_name in enumerate(self.class_names):
            points = output_dict['center_preds'][i]
            batch_idx = output_dict['batch_idx'][i]
            cluster_inds, valid_mask = self.clusster_assign(points, batch_idx, class_name, i)
            cluster_inds_list.append(cluster_inds)
            valid_mask_list.append(valid_mask)
        pts_cluster_inds = torch.cat(cluster_inds_list, dim=0)
        sampled_out = self.update_sample_results_by_mask(output_dict, valid_mask_list)
        combined_out = self.combine_classes(sampled_out,
                                            ['seg_points_xyzs', 'seg_points_feat', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])
        points_xyzs = combined_out['seg_points_xyzs']
        points_feat = combined_out['seg_points_feat']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']],
                              dim=1)
        # assert len(pts_cluster_inds) == len(points) == len(pts_feats)
        # losses['num_fg_points'] = torch.ones((1,), device=points.device).float() * len(points)

        extracted_outs = self.extract_feat(points_xyzs, points_feat, pts_feats, pts_cluster_inds, combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds']

        outs = self.bbox_head(cluster_feats, cluster_xyz, cluster_inds)

        # bbox_list = self.bbox_head.get_bboxes(
        #     outs['cls_logits'], outs['reg_preds'],
        #     cluster_xyz, cluster_inds, img_metas,
        #     rescale=rescale,
        #     iou_logits=outs.get('iou_logits', None))
        result = []
        for k, v in outs.items():
            result.extend(v)
        return result


class sst_stage_two_pre_process(nn.Module):
    def __init__(self, model):
        super(sst_stage_two_pre_process, self).__init__()
        self.model = deepcopy(model)
        # self.test_cfg = model.test_cfg
        # self.num_classes = model.num_classes
        # self.prepare_multi_class_roi_input = model.prepare_multi_class_roi_input
        # self.prepare_roi_input = model.prepare_roi_input
        # self.roi_head = model.roi_head

    def forward(self, points_xyzs, points_feat, coords, centers, imetas):
        rpn_outs = self.model.before_second_stage(points_xyzs, points_feat, coords, centers, imetas)
        return rpn_outs

class sst_stage_two_nn_infer(nn.Module):
    def __init__(self, model):
        super(sst_stage_two_nn_infer, self).__init__()
        self.model = deepcopy(model)
        self.test_cfg = model.test_cfg
        self.scatter_bool = Scatter_bool()
        # self.num_classes = model.num_classes
        # self.prepare_multi_class_roi_input = model.prepare_multi_class_roi_input
        # self.prepare_roi_input = model.prepare_roi_input
        # self.roi_head = model.roi_head

    def forward(self,
            new_points_xyzs,
            new_points_feat,
            new_pts_feats,
            ext_pts_info,
            ext_pts_roi_inds,
            rois, class_labels, class_pred):
        cls_score, bbox_pred, valid_roi_mask = self.model.roi_head.bbox_head(
            new_points_xyzs,
            new_points_feat,
            new_pts_feats,
            ext_pts_info,
            ext_pts_roi_inds,
            rois
        )

        cls_score = cls_score.sigmoid()
        assert (class_pred[0] >= 0).all()

        if self.test_cfg.get('rcnn_score_nms', False):
            # assert class_pred[0].shape == cls_score.shape
            class_pred[0] = cls_score.squeeze(1)
        valid_roi_mask = self.scatter_bool(valid_roi_mask)

        # regard empty bboxes as false positive
        rois = rois[valid_roi_mask]
        cls_score = cls_score[valid_roi_mask]
        bbox_pred = bbox_pred[valid_roi_mask]

        # for i in range(len(class_labels)):
        #     class_labels[i] = class_labels[i][valid_roi_mask]
        #     class_pred[i] = class_pred[i][valid_roi_mask]
        labels = []
        cls_preds = []
        for label, pred in zip(class_labels, class_pred):
            labels.append(label[valid_roi_mask])
            cls_preds.append(pred[valid_roi_mask])
        roi_boxes = rois[..., 1:]
        # roi_ry = roi_boxes[..., 6].view(-1)
        # roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        rcnn_boxes3d = self.model.roi_head.bbox_head.bbox_coder.decode(local_roi_boxes, bbox_pred)
        return roi_boxes, bbox_pred, cls_score, rcnn_boxes3d, labels, cls_preds

def cos_similarity(data0, data1):
    result = torch.sum(data0 * data1) / \
    torch.sqrt(torch.sum(data0 * data0) * torch.sum(data1 * data1))
    return result

def main():
    torch.backends.cuda.matmul.allow_tf32 = False # prevent matmul error in 3090, a very tricky bug.
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if args.infer:
        if cfg.model.type == "FSD":
            cfg.model.type = "FSD_infer"
        if cfg.model.segmentor.type == "VoteSegmentor":
            cfg.model.segmentor.type = "VoteSegmentor_infer"
        if cfg.model.segmentor.voxel_encoder.type == "DynamicScatterVFE":
            cfg.model.segmentor.voxel_encoder.type = "DynamicScatterVFE_infer"
        if cfg.model.segmentor.decode_neck.type == 'Voxel2PointScatterNeck':
            cfg.model.segmentor.decode_neck.type = 'Voxel2PointScatterNeck_infer'
        if cfg.model.backbone.type == 'SIR':
            cfg.model.backbone.type = 'SIR_infer'
        if cfg.model.roi_head.bbox_head.type == 'FullySparseBboxHead':
            cfg.model.roi_head.bbox_head.type = 'FullySparseBboxHead_infer'
        if cfg.model.roi_head.type == "GroupCorrectionHead":
            cfg.model.roi_head.type = "GroupCorrectionHead_infer"
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    dataset = build_dataset(cfg.data.test)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # point_range = cfg.data.test.pipeline[1].transforms[2].point_cloud_range
    # load_dim = cfg.data.test.pipeline[0].load_dim
    # use_dim = cfg.data.test.pipeline[0].use_dim
    # path = args.lidarbin
    # points_all = load_data(path, point_range, load_dim, use_dim)
    model.eval()
    encoder_export = segment_voxel_encoder(model)
    tmp_backbone = segment_backbone_modified(model)
    tmp_backbone.backbone.eval()
    segment_decoder = segment_decode(model)
    segment_decoder.segmentation_head.eval()
    stage_two_pre_process = sst_stage_two_pre_process(model)
    stage_two_process = sst_stage_two_nn_infer(model)
    data = dataset[0]
    points = [data['points'][0].data]
    points[0] = points[0].cuda()
    imetas = [data['img_metas'][0].data]

    if model.segmentor.voxel_downsampling_size is not None:
        points = model.segmentor.voxel_downsample(points)
    infer_points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
    voxel_layer = model.segmentor.voxel_layer
    encoder = model.segmentor.voxel_encoder
    infer_points, coords, centers = infer_voxelize(infer_points, voxel_layer, encoder)
    batch_idx = coords[:, 0]
    decord_coords = pre_voxelize(model, infer_points, batch_idx)
    args.export = True
    infer_points_xyzs = infer_points[:, :3]
    infer_points_feat = infer_points[:, 3:]

    # np_points = infer_points.detach().cpu().numpy()
    # np_coords = coords.detach().cpu().numpy()
    # np.save('torch_tensorrt/points.npy', np_points)
    # np.save('torch_tensorrt/coords.npy', np_coords)


    with torch.no_grad():
        if args.export == False:
            out = model.simple_test(infer_points_xyzs, infer_points_feat, coords, centers, imetas)[0]
            print(out)
            obstacles = {
                'boxes': out['boxes_3d'].tensor.numpy(),
                'scores': out['scores_3d'].numpy(),
                'classes': out['labels_3d'].numpy()
            }
        else:
            out_stage_two_pre_process = stage_two_pre_process(infer_points_xyzs, infer_points_feat, coords, centers, imetas)
            test = stage_two_process(*out_stage_two_pre_process)
            export_out = encoder_export(infer_points_xyzs, infer_points_feat, coords, centers)
            gather_data = export_out[0].detach().cpu().numpy()
            gather_indice = export_out[2].detach().cpu().numpy()
            gather_unique = export_out[1].detach().cpu().numpy()
            np.save('./torch_tensorrt/gather_data.npy', gather_data)
            np.save('./torch_tensorrt/gather_indice.npy', gather_indice)
            np.save('./torch_tensorrt/gather_unique.npy', gather_unique)
            tmp_data = tmp_backbone(export_out[0], export_out[1].int())
            test_model = segment_backbone(model)
            test_out = test_model(export_out[0], export_out[1])
            result0 = tmp_data[0]['voxel_feats']
            result1 = test_out[0]['voxel_feats']
            cos_simi = cos_similarity(result0, result1)
            cos_simi = cos_simi.cpu().numpy()
            last = deepcopy(decord_coords[:, 0:1].int())
            segment_out = segment_decoder(infer_points_xyzs, infer_points_feat, coords, centers, result1, export_out[2], decord_coords, last)

            print("cos similarity %f of between segement modified model and origin model" % cos_simi)
            torch.onnx.export(encoder_export,
                (infer_points_xyzs, infer_points_feat, coords, centers),
                "voxel_encoder.onnx",
                opset_version=11,
                input_names=["points_xyzs", "points_feat", "coords", "centers"],
                output_names=["voxel_feat", "voxel_coords", "voxel2point_inds"],
                # example_outputs=export_out
                              )

            torch.onnx.export(segment_decoder,
                              (infer_points_xyzs, infer_points_feat, coords, centers, result1, export_out[2], decord_coords, last),
                              "stage_one.onnx",
                              opset_version=17,
                              input_names=["points_feat", "points_feat", "coords", "centers", "voxel_feats", "voxel2point_inds", "voxel_coords"],
                              # output_names=["voxel_feat", "voxel_coords", "voxel2point_inds"],
                              # example_outputs=segment_out
                              )

            torch.onnx.export(stage_two_process,
                              out_stage_two_pre_process,
                              "stage_two.onnx",
                              opset_version=17,
                              # input_names=["points", "coords", "centers", "voxel_feats", "voxel2point_inds",
                              #              "voxel_coords"],
                              # output_names=["voxel_feat", "voxel_coords", "voxel2point_inds"],
                              # example_outputs=test
                              )
            exptool.export_onnx(tmp_backbone, export_out[0], export_out[1].int(), 'Unet_Segmentor.onnx', None)



if __name__ == '__main__':
    main()
