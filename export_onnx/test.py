import torch_scatter
import torch
import numpy as np

# class SparseConvFunc(torch.autograd.Function):
#     @staticmethod
#     def symbolic(g, cls, feat, in_pos, out_pos, voxel_size):
#         kernel = cls.state_dict()["kernel"]
#         offset = cls.state_dict()["offset"]
#         kernel = g.op("Constant", value_t=kernel)
#         offset = g.op("Constant", value_t=offset)
#         return g.op("SparseConv", feat, in_pos, out_pos, kernel, offset)
#
#     @staticmethod
#     def forward(self, cls, feat, in_pos, out_pos, voxel_size):
#         return cls.origin_forward(feat, in_pos, out_pos, voxel_size)
#
# class SparseConvONNX(SparseConv):
#     """
#     This is a support class which helps export network with SparseConv in ONNX format.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.origin_forward = super().forward
#
#     def forward(self, feat, in_pos, out_pos, voxel_size):
#         return SparseConvFunc.apply(self, feat, in_pos, out_pos, voxel_size)



class scatter_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, unq_inv, coor, mode):
        if mode == 'max':
            new_feat = torch_scatter.scatter_max(feat, unq_inv, dim=0)[0]
        elif mode == 'mean':
            new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='mean')
            tmp_feat = torch.ops.torch_scatter.scatter_mean(feat, unq_inv, 0, None, None)

        else:
            new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='sum')
            tmp_feat = torch.ops.torch_scatter.scatter_sum(feat, unq_inv, 0, None, None)
        np_feat0 = new_feat.cpu().numpy()
        np_feat1 = tmp_feat.cpu().numpy()
        diff = np_feat0 - np_feat1
        np_feat0_min = np_feat0.min()
        np_feat0_max = np_feat0.max()
        np_feat1_min = np_feat1.min()
        np_feat1_max = np_feat1.max()
        diff_min = diff.min()
        diff_max = diff.max()
        return new_feat

    @staticmethod
    def symbolic(g, feat, unq_inv, coor, mode):
        assert mode in ('max', 'mean', 'sum')
        return g.op('zf::scatter_sst', feat, unq_inv, coor, mode_s=mode)
scatter_fun = scatter_kernel.apply

class scatter_sst(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat, unq_inv, coor, mode='max'):
        new_feat = scatter_fun(feat, unq_inv, coor, mode)
        return new_feat

class test_nozero(torch.nn.Module):
    def __init__(self):
        super(test_nozero, self).__init__()
    def forward(self, data):
        b = torch.nonzero(data == 10).squeeze()
        return b

# def export_scatter_max(g, feat, unq_inv, dim, out, dim_size):
#     return g.op("zf::scatter_max", feat, unq_inv, dim)
# register_custom_op_symbolic('torch_scatter::scatter_max', export_scatter_max, 11)
#
# def export_scatter(g, feat, unq_inv, dim, out, dim_size, reduce):
#     return g.op("zf::scatter", feat, unq_inv, dim, reduce)
# register_custom_op_symbolic('torch_scatter::scatter', export_scatter, 11)

if __name__ == '__main__':
    test_model = test_nozero()
    a = torch.arange(3 * 5).reshape(3, 5).view(-1)
    b = test_model(a)
    torch.onnx.export(test_model, a, 'test_nonzero.onnx', opset_version=17, verbose=True)
    model = scatter_sst()
    feat = torch.from_numpy(np.load('feat_max_125026.npy'))
    coor = torch.from_numpy(np.load('coor_max_125026.npy'))
    indx = torch.from_numpy(np.load('indx_max_125026.npy'))
    one = torch.ones(1)
    test_0 = model(feat, indx, coor, 'sum')
    # test_1, _ = torch.scatter_reduce(dim=0, indx, feat, reduce='amax')
    mode = torch.ones(1) * 2
    print()
    torch.onnx.export(model, (feat, indx, coor), 'test.onnx', opset_version=11, verbose=True)
