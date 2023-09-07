import torch_scatter
import torch
import numpy as np


class SparseConvFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, cls, feat, in_pos, out_pos, voxel_size):
        kernel = cls.state_dict()["kernel"]
        offset = cls.state_dict()["offset"]
        kernel = g.op("Constant", value_t=kernel)
        offset = g.op("Constant", value_t=offset)
        return g.op("SparseConv", feat, in_pos, out_pos, kernel, offset)

    @staticmethod
    def forward(self, cls, feat, in_pos, out_pos, voxel_size):
        return cls.origin_forward(feat, in_pos, out_pos, voxel_size)

class SparseConvONNX(SparseConv):
    """
    This is a support class which helps export network with SparseConv in ONNX format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_forward = super().forward

    def forward(self, feat, in_pos, out_pos, voxel_size):
        return SparseConvFunc.apply(self, feat, in_pos, out_pos, voxel_size)

class scatter_all(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, unq_inv, value_s):
        if value_s == 'max':
            new_feat = torch_scatter.scatter_max(feat, unq_inv, dim=0)[0]
        elif value_s == 'mean':
            new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='mean')
        else:
            new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce='sum')
        return new_feat

    @staticmethod
    def symbolic(g, feat, unq_inv, mode):
        test = str(mode)
        key = 'Float('
        index = test.rfind(key) + len(key)
        test = float(test[index: index+1])
        assert test in (1, 2, 3)
        if test == 1:
            return g.op('zf::scatter_all', feat, unq_inv, value_s='max')
        elif test == 2:
            return g.op('zf::scatter_all', feat, unq_inv, value_s='mean')
        else:
            return g.op('zf::scatter_all', feat, unq_inv, value_s='sum')

scatter_fun = scatter_all.apply

class scatter_infer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat, unq_inv, mode):
        new_feat = scatter_fun(feat, unq_inv, mode)
        return new_feat

# def export_scatter_max(g, feat, unq_inv, dim, out, dim_size):
#     return g.op("zf::scatter_max", feat, unq_inv, dim)
# register_custom_op_symbolic('torch_scatter::scatter_max', export_scatter_max, 11)
#
# def export_scatter(g, feat, unq_inv, dim, out, dim_size, reduce):
#     return g.op("zf::scatter", feat, unq_inv, dim, reduce)
# register_custom_op_symbolic('torch_scatter::scatter', export_scatter, 11)

if __name__ == '__main__':
    model = scatter_infer()
    feat = torch.from_numpy(np.load('feat_max_125026.npy'))
    coor = torch.from_numpy(np.load('coor_max_125026.npy'))
    indx = torch.from_numpy(np.load('indx_max_125026.npy'))
    one = torch.ones(1)
    # test_0 = model(feat, indx, one)
    # test_1, _ = torch.scatter_reduce(dim=0, indx, feat, reduce='amax')
    mode = torch.ones(1) * 2
    print()
    torch.onnx.export(model, (feat, indx, mode), 'test0.onnx', opset_version=11, verbose=True)
