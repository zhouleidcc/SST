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
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

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
        coors = coors.int()
        batch_size = coors[:, 0].max().item() + 1
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.backbone.sparse_shape,
                                           batch_size)
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
    unet_backbone = segment_backbone(model)
    tmp_backbone = segment_backbone_modified(model)
    tmp_backbone.backbone.eval()
    data = dataset[0]
    points = [data['points'][0].data]
    points[0] = points[0].cuda()
    imetas = [data['img_metas'][0].data]
    with torch.no_grad():
        if args.infer == False:
            out = model.simple_test(points, imetas)[0]
        else:
            if model.segmentor.voxel_downsampling_size is not None:
                points = model.segmentor.voxel_downsample(points)
            infer_points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
            voxel_layer = model.segmentor.voxel_layer
            encoder = model.segmentor.voxel_encoder
            infer_points, coords, centers = infer_voxelize(infer_points, voxel_layer, encoder)
            # out = model.simple_test(infer_points, coords, centers, imetas)[0]
            export_out = encoder_export(infer_points, coords, centers)
            tmp_data = tmp_backbone(export_out[0], export_out[1])
            test_model = segment_backbone(model)
            test_out = test_model(export_out[0], export_out[1])
            diff = tmp_data[0]['voxel_feats'] - test_out[0]['voxel_feats']
            diff = diff.cpu().numpy()
            diff_min = diff.min()
            diff_max = diff.max()

            unet_backbone = layer_fusion(unet_backbone)
            res_unet = unet_backbone(export_out[0], export_out[1])
            torch.onnx.export(encoder_export,
                (infer_points, coords, centers),
                "voxel_encoder.onnx",
                opset_version=13,
                input_names=["points", "coords", "centers"],
                output_names=["voxel_feat", "voxel_coords", "voxel2point_inds"],
                example_outputs=export_out
                              )
            # spatial_shape = unet_backbone.sparse_shape[::-1]


            import exptool
            exptool.export_onnx(tmp_backbone, export_out[0], export_out[1], 'u_net_spconv_3d.onnx', None)
            # torch.onnx.export(encoder_export,
            #                   (infer_points, coords, centers),
            #                   "voxel_encoder.onnx",
            #                   opset_version=13,
            #                   input_names=["points", "coords", "centers"],
            #                   output_names=["voxel_feat", "voxel_coords", "voxel2point_inds"],
            #                   example_outputs=export_out
            #                   )
        # obstacles = {
        #     'boxes': out['boxes_3d'].tensor.numpy(),
        #     'scores': out['scores_3d'].numpy(),
        #     'classes': out['labels_3d'].numpy()
        # }


if __name__ == '__main__':
    main()
