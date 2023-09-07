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
import open3d as o3d
from visual import plot_boxes
from torch import nn

vis = o3d.visualization.Visualizer()
pointcloud = o3d.geometry.PointCloud()
def visual_boxes(points, detections, threshold):
    global vis, pointcloud
    lines = plot_boxes(detections, threshold)
    for line in lines:
        vis.add_geometry(line)
        vis.update_geometry(line)

    pointcloud.points = o3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pointcloud)
    # vis.update_geometry()
    # 注意，如果使用的是open3d 0.8.0以后的版本，这句话应该改为下面格式
    vis.update_geometry(pointcloud)

    vis.poll_events()
    vis.update_renderer()

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--lidarbin', default='data/waymo/kitti_format/training/velodyne1000000.bin',
        help='single lidar frame data saved in bin file')
    parser.add_argument('--show', default=True, help='Whether to show detection results')
    parser.add_argument('--fuse-conv-bn', action='store_true',
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

class segment_voxel_layer(nn.Module):
    def __init__(self, model):
        super(segment_voxel_layer, self).__init__()
        self.voxel_layer = model.segmentor.voxel_layer
        self.model = model

    # @torch.no_grad()
    # @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def forward(self, points):
        points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
        return self.voxelize(points)

class segment_voxel_encoder(nn.Module):
    def __init__(self, model):
        super(segment_voxel_encoder, self).__init__()
        self.voxel_encoder = model.segmentor.voxel_encoder
        self.model = model
    def forward(self, batch_points, coors, return_inv=True):
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(batch_points, coors, return_inv=return_inv)
        return voxel_features, voxel_coors, voxel2point_inds

class segment_middle_encoder(nn.Module):
    def __init__(self, model):
        super(segment_middle_encoder, self).__init__()
        self.middle_encoder = model.segmentor.middle_encoder
        self.model = model
    def forward(self, voxel_features, voxel_coors):
        voxel_info = self.middle_encoder(voxel_features, voxel_coors)
        return voxel_info

class segment_backbone(nn.Module):
    def __init__(self, model):
        super(segment_backbone, self).__init__()
        self.backbone = model.segmentor.backbone
        self.model = model
    def forward(self, voxel_info):
        x = self.backbone(voxel_info)[0]
        voxel_feats_reorder = x['voxel_feats']
        return voxel_feats_reorder

class segment_decode_neck(nn.Module):
    def __init__(self, model):
        super(segment_decode_neck, self).__init__()
        self.decode_neck = model.segmentor.decode_neck
        self.model = model
    def forward(self, batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding):
        out = self.decode_neck(batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding)
        return out

class segment_seg_head(nn.Module):
    def __init__(self, model):
        super(segment_seg_head, self).__init__()
        self.seg_head = model.segmentor.segmentation_head
        self.model = model
    def forward(self, feats):
        seg_logits, vote_preds = self.seg_head.forward_test(feats, [None], None)
        offsets = self.seg_head.decode_vote_targets(vote_preds)
        return seg_logits, vote_preds, offsets


class Segment(nn.Module):
    def __init__(self, model):
        super(Segment, self).__init__()
        self.voxel_layer = model.segmentor.voxel_layer
        self.voxel_encoder = model.segmentor.voxel_encoder
        self.middle_encoder = model.segmentor.middle_encoder
        self.backbone = model.segmentor.backbone
        self.decode_neck = model.segmentor.decode_neck
        self.segmentation_head = model.segmentor.segmentation_head


class FSD(nn.Module):
    def __init__(self, model):
        super(FSD, self).__init__()

        self.reader = model.reader
        self.backbone = model.backbone
        # self.with_neck = model.with_neck
        # self.neck = model.neck
        # self.bbox_head = model.bbox_head

    def forward(self, voxels, coordinates, num_points_in_voxel, num_voxels, batchsize):
        input_features = self.reader(voxels, num_points_in_voxel)
        x, voxel_feature = self.backbone(input_features, coordinates, num_voxels, batchsize)
        # if self.with_neck:
        #     x = self.neck(x)
        # preds = self.bbox_head(x)
        # for task in range(len(preds)):
        #     hm_preds = torch.sigmoid(preds[task]['hm'])
        #     preds[task]['dim'] = torch.exp(preds[task]['dim'])
        #     scores, labels = torch.max(hm_preds, dim=1)
        #     preds[task]["hm"] = (scores, labels)
        return voxel_feature['features'], voxel_feature['indices'], voxel_feature['num_voxels']


def main():
    torch.backends.cuda.matmul.allow_tf32 = False # prevent matmul error in 3090, a very tricky bug.
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # build the model and load checkpoint
    cfg.model.train_cfg = None
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
    test = dataset[0]

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
    data_len = len(dataset)
    for i in range(data_len):
        data = dataset[i]
        points = [data['points'][0].data]
        points[0] = points[0].cuda()
        imetas = [data['img_metas'][0].data]
        with torch.no_grad():
            out = model.simple_test(points, imetas)[0]
            obstacles = {
                'boxes': out['boxes_3d'].tensor.numpy(),
                'scores': out['scores_3d'].numpy(),
                'classes': out['labels_3d'].numpy()
            }
        if args.show and i < 100:
            if i == 0:
                global vis
                vis.create_window()
            visual_boxes(points[0].cpu().detach().numpy(), obstacles, 0.01)
        else:
            break
    print('ok')

if __name__ == '__main__':
    main()
