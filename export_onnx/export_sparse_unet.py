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


def infer_voxelize(points, voxel_layer, encoder):
    coors = []
    # dynamic voxelization only provide a coors mapping
    for res in points:
        res_coors = voxel_layer(res)
        coors.append(res_coors)
    points = torch.cat(points, dim=0)
    coors_batch = []
    for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
    coors = torch.cat(coors_batch, dim=0)

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
    def forward(self, points, in_coors, centers):
        batch_points = points
        coors = in_coors
        coors = coors.long()
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(batch_points, coors, centers,
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

# class spconv_export_model(nn.Module):
#     def __init__(self, model):
#         super(spconv_export_model, self).__init__()
#         self.model = model
#
#     def forward(self, feat, coords):



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
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility


    model.eval()
    unet_backbone = segment_backbone(model)
    from funcs import layer_fusion
    unet_backbone = layer_fusion(unet_backbone)
    import exptool
    exptool.export_onnx(unet_backbone, export_out[0], export_out[1], 'u_net_spconv_3d.onnx', None)
    # torch.onnx.export(encoder_export,
    #                   (infer_points, coords, centers),
    #                   "voxel_encoder.onnx",
    #                   opset_version=13,
    #                   input_names=["points", "coords", "centers"],
    #                   output_names=["voxel_feat", "voxel_coords", "voxel2point_inds"],
    #                   example_outputs=export_out
    #                   )



if __name__ == '__main__':
    main()
