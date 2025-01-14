import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32, auto_fp16
from torch import nn
import random

from mmdet3d.ops import DynamicScatter
from .. import builder
from ..builder import VOXEL_ENCODERS
from .utils import VFELayer, DynamicVFELayer, get_paddings_indicator,  DynamicVFELayerV2
from ipdb import set_trace
from mmdet3d.ops import make_sparse_convmodule, scatter_v2, scatter_sst
from mmdet3d.ops import spconv as spconv
import pickle as pkl
import os
import torch_scatter

@VOXEL_ENCODERS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int): Number of features to use. Default: 4.
    """

    def __init__(self, num_features=4):
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
        """
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()


@VOXEL_ENCODERS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        self.fp16_enabled = False

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors


@VOXEL_ENCODERS.register_module()
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 ):
        super(DynamicVFE, self).__init__()
        # assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False):
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                num_points,
                coors,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if (self.fusion_layer is not None and img_feats is not None):
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
                                                coors, img_feats, img_metas)

        return voxel_feats

    def fusion_with_mask(self, features, mask, voxel_feats, coors, img_feats,
                         img_metas):
        """Fuse image and point features with mask.

        Args:
            features (torch.Tensor): Features of voxel, usually it is the
                values of points in voxels.
            mask (torch.Tensor): Mask indicates valid features in each voxel.
            voxel_feats (torch.Tensor): Features of voxels.
            coors (torch.Tensor): Coordinates of each single voxel.
            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.
            img_metas (list(dict)): Meta information of image and points.

        Returns:
            torch.Tensor: Fused features of each voxel.
        """
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats,
                                        img_metas)

        voxel_canvas = voxel_feats.new_zeros(
            size=(voxel_feats.size(0), voxel_feats.size(1),
                  point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out

@VOXEL_ENCODERS.register_module()
class DynamicScatterVFE(DynamicVFE):
    """ Same with DynamicVFE but use torch_scatter to avoid construct canvas in map_voxel_center_to_point.
    The canvas is very memory-consuming when use tiny voxel size (5cm * 5cm * 5cm) in large 3D space.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 unique_once=False,
                 ):
        super(DynamicScatterVFE, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.unique_once = unique_once

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False):
            
        if self.unique_once:
            new_coors, unq_inv_once = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv_once = None

        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, _, unq_inv = scatter_v2(features[:, :3], coors, mode='avg', new_coors=new_coors, unq_inv=unq_inv_once)
            points_mean = self.map_voxel_center_to_point(voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster / self.rel_dist_scaler)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, new_coors=new_coors, unq_inv=unq_inv_once)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class DynamicScatterVFE_infer(DynamicVFE):
    """ Same with DynamicVFE but use torch_scatter to avoid construct canvas in map_voxel_center_to_point.
    The canvas is very memory-consuming when use tiny voxel size (5cm * 5cm * 5cm) in large 3D space.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 unique_once=False,
                 ):
        super(DynamicScatterVFE_infer, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.unique_once = unique_once

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self, points_xyzs, points_feat,
                coors,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False,
                coor_center=None):

        if self.unique_once:
            new_coors, unq_inv_once, count = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        else:
            new_coors = unq_inv_once = None

        features_ls = [points_xyzs, points_feat]
        # origin_point_coors = points_xyzs
        # Find distance of x, y, and z from cluster center
        scatter_infer = scatter_sst()
        num_voxel = count.cpu().numpy().shape[0]
        num_voxel = torch.Tensor([num_voxel]).to(device=count.device, dtype=count.dtype)
        if self._with_cluster_center:
            # voxel_mean, _= scatter_v2(points_xyzs, coors, mode='avg', return_inv=False, new_coors=new_coors,
            #                                     unq_inv=unq_inv_once)
            voxel_mean = scatter_infer(points_xyzs, unq_inv_once, count, num_voxel, 'mean')

            ###########
            """
            from torchex import scatter_sum, scatter_max, scatter_mean, TorchTimer, ScatterData
            from copy import deepcopy
            tmp = deepcopy(points_xyzs)
            tmp = tmp.contiguous()
            data = ScatterData(unq_inv_once, count)
            ans2 = scatter_mean(tmp, data)
            flag1 = bool(((voxel_mean - ans2) < 1e-5).all().cpu())
            print(flag1)
            """

            ###########
            unq_inv = unq_inv_once
            points_mean = self.map_voxel_center_to_point(voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = points_xyzs - points_mean
            if self.rel_dist_scaler != 1.0:
                f_cluster /= self.rel_dist_scaler
            features_ls.append(f_cluster)
        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center and coor_center is None:
            if points is not None:
                f_center = points
            else:
                f_center = features.new_zeros(size=(features.size(0), 3))
                f_center[:, 0] = features[:, 0] - (
                        coors[:, 3].type_as(features) * self.vx + self.x_offset)
                f_center[:, 1] = features[:, 1] - (
                        coors[:, 2].type_as(features) * self.vy + self.y_offset)
                f_center[:, 2] = features[:, 2] - (
                        coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)
        elif self._with_voxel_center and coor_center is not None:
            features_ls.append(coor_center)

        if self._with_distance:
            points_dist = torch.norm(points_xyzs, 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        if len(features_ls) > 1:
            features = torch.cat(features_ls, dim=-1)
        else:
            features = f_cluster

        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            # voxel_feats, voxel_coors = scatter_v2(point_feats, coors, mode=self.mode, return_inv=False, new_coors=new_coors,
            #                                                unq_inv=unq_inv_once)
            voxel_feats = scatter_infer(point_feats, unq_inv_once, count, num_voxel, self.mode)
            import numpy as np
            np_point_feats = point_feats.cpu().numpy()
            np_unq_inv_once = unq_inv_once.cpu().numpy()
            np_count = count.cpu().numpy()
            np_voxel_feats = voxel_feats.cpu().numpy()
            path = '/home/nvidia/work/xavier_python_projects/python_projects/tensorrt_deploy_demo/sst/scatter/%s.npy'
            np.save(path % 'point_feats', np_point_feats)
            np.save(path % 'unq_inv_once', np_unq_inv_once)
            np.save(path % 'count', np_count)
            np.save(path % 'voxel_feats', np_voxel_feats)
            voxel_coors = new_coors
            unq_inv = unq_inv_once
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv, count
        else:
            return voxel_feats, voxel_coors


from mmdet3d.ops import build_mlp

@VOXEL_ENCODERS.register_module()
class SIRLayer(DynamicVFE):

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_rel_mlp=True,
                 rel_mlp_hidden_dims=[16,],
                 rel_mlp_in_channel=3,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 with_shortcut=True,
                 xyz_normalizer=[1.0, 1.0, 1.0],
                 act='relu',
                 dropout=0.0,
                 ):
        super().__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.with_shortcut = with_shortcut
        self._with_rel_mlp = with_rel_mlp
        self.xyz_normalizer = xyz_normalizer
        if with_rel_mlp:
            rel_mlp_hidden_dims.append(in_channels) # not self.in_channels
            self.rel_mlp = build_mlp(rel_mlp_in_channel, rel_mlp_hidden_dims, norm_cfg, act=act)

        if act != 'relu' or dropout > 0: # do not double in_filter
            feat_channels = [self.in_channels] + list(feat_channels)
            vfe_layers = []
            for i in range(len(feat_channels) - 1):
                in_filters = feat_channels[i]
                out_filters = feat_channels[i + 1]
                if i > 0:
                    in_filters *= 2

                vfe_layers.append(
                    DynamicVFELayerV2(
                        in_filters,
                        out_filters,
                        norm_cfg,
                        act=act,
                        dropout=dropout,
                    )
                )
            self.vfe_layers = nn.ModuleList(vfe_layers)
            self.num_vfe = len(vfe_layers)
        

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                f_cluster=None,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False,
                return_both=False,
                unq_inv_once=None,
                new_coors_once=None,
        ):

        xyz_normalizer = torch.tensor(self.xyz_normalizer, device=features.device, dtype=features.dtype)
        features_ls = [torch.cat([features[:, :3] / xyz_normalizer[None, :], features[:, 3:]], dim=1)]
        # origin_point_coors = features[:, :3]
        if self.with_shortcut:
            shortcut = features[:, 3:]
        if f_cluster is None:
            # Find distance of x, y, and z from cluster center
            voxel_mean, mean_coors, unq_inv = scatter_v2(features[:, :3], coors, mode='avg', unq_inv=unq_inv_once, new_coors=new_coors_once)
            points_mean = self.map_voxel_center_to_point(
                voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = (features[:, :3] - points_mean[:, :3]) / self.rel_dist_scaler
        else:
            f_cluster = f_cluster / self.rel_dist_scaler

        if self._with_cluster_center:
            features_ls.append(f_cluster / 10.0)

        if self._with_rel_mlp:
            features_ls[0] = features_ls[0] * self.rel_mlp(f_cluster)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        voxel_feats_list = []
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, unq_inv=unq_inv_once, new_coors=new_coors_once)
            voxel_feats_list.append(voxel_feats)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = torch.cat(voxel_feats_list, dim=1)

        if return_both:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats, voxel_coors

        if self.return_point_feats:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class SIRLayer_infer(DynamicVFE):

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_rel_mlp=True,
                 rel_mlp_hidden_dims=[16, ],
                 rel_mlp_in_channel=3,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 with_shortcut=True,
                 xyz_normalizer=[1.0, 1.0, 1.0],
                 act='relu',
                 dropout=0.0,
                 ):
        super().__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.with_shortcut = with_shortcut
        self._with_rel_mlp = with_rel_mlp
        self.xyz_normalizer = xyz_normalizer
        self.scatter_export = scatter_sst()
        self.xyz_normalizer_inverse = [1.0 /a for a in self.xyz_normalizer]
        self.rel_dist_scaler_inverse = 1.0 /self.rel_dist_scaler
        if with_rel_mlp:
            rel_mlp_hidden_dims.append(in_channels)  # not self.in_channels
            self.rel_mlp = build_mlp(rel_mlp_in_channel, rel_mlp_hidden_dims, norm_cfg, act=act)

        if act != 'relu' or dropout > 0:  # do not double in_filter
            feat_channels = [self.in_channels] + list(feat_channels)
            vfe_layers = []
            for i in range(len(feat_channels) - 1):
                in_filters = feat_channels[i]
                out_filters = feat_channels[i + 1]
                if i > 0:
                    in_filters *= 2

                vfe_layers.append(
                    DynamicVFELayerV2(
                        in_filters,
                        out_filters,
                        norm_cfg,
                        act=act,
                        dropout=dropout,
                    )
                )
            self.vfe_layers = nn.ModuleList(vfe_layers)
            self.num_vfe = len(vfe_layers)

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                points_xyzs,
                points_feat,
                features,
                coors,
                f_cluster=None,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False,
                return_both=False,
                unq_inv_once=None,
                new_coors_once=None,
                count=None
                ):

        xyz_normalizer = torch.tensor(self.xyz_normalizer_inverse, device=features.device, dtype=features.dtype)
        features_ls = [torch.cat([points_xyzs * xyz_normalizer[None, :], points_feat, features], dim=1)]
        # origin_point_coors = points_xyzs
        if self.with_shortcut:
            shortcut = features
        if f_cluster is None:
            # Find distance of x, y, and z from cluster center
            # voxel_mean = self.scatter_export(points_xyzs, unq_inv_once, 'mean')
            # unq_inv = unq_inv_once
            # mean_coors = new_coors_once
            voxel_mean, mean_coors, unq_inv = scatter_v2(points_xyzs, coors, mode='avg', unq_inv=unq_inv_once,
                                                         new_coors=new_coors_once)
            points_mean = self.map_voxel_center_to_point(
                voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = (points_xyzs - points_mean[:, :3]) * self.rel_dist_scaler_inverse
        else:
            f_cluster = f_cluster * self.rel_dist_scaler_inverse

        if self._with_cluster_center:
            features_ls.append(f_cluster * 0.1)

        if self._with_rel_mlp:
            features_ls[0] = features_ls[0] * self.rel_mlp(f_cluster)

        if self._with_distance:
            points_dist = torch.norm(points_xyzs, 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        if len(features_ls) == 1:
            features = features_ls[0]
        else:
            features = torch.cat(features_ls, dim=-1)

        voxel_feats_list = []
        scatter_infer = scatter_sst()
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            mode = 'mean' if self.mode == 'avg' else self.mode
            num_point = count.cpu().numpy().shape[0]
            num_point = torch.Tensor([num_point]).to(device=count.device, dtype=count.dtype)
            voxel_feats = scatter_infer(point_feats, unq_inv_once, count, num_point, mode)
            voxel_coors = new_coors_once
            unq_inv = unq_inv_once
            voxel_feats_list.append(voxel_feats)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = torch.cat(voxel_feats_list, dim=1)

        if return_both:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats, voxel_coors

        if self.return_point_feats:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors
