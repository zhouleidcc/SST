# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import collections
from spconv.pytorch import SparseSequential
from spconv.pytorch import conv
from mmdet3d.ops.sparse_block import SparseBasicBlock
import cumm.tensorview as tv
from mmdet3d.models.middle_encoders.sparse_unet import SimpleSparseUNet
import numpy as np
from mmdet3d.ops.norm import NaiveSyncBatchNorm1d

def make_new_repr(old_repr):
    def new_repr(self):
        s = old_repr(self)
        if self.act_type is not None:
            p = s.rfind(")")
            s = s[:p] + f', act={self.act_type}' + s[p:]
        return s
    return new_repr

# setup repr function, add activation
conv.SparseConvolution.__repr__ = make_new_repr(conv.SparseConvolution.__repr__)

def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim+1] + [i+1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    # OIDHW
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0,] + [i+2 for i in range(NDim)] + [1,]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)

def fuse_bn(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    conv.weight, conv.bias = fuse_bn_weights(conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

def load_scn_backbone_checkpoint(model, file, prefix='', permute=False):

    device   = next(model.parameters()).device
    ckpt     = torch.load(file, map_location=device)["state_dict"]
    new_ckpt = collections.OrderedDict()
    for key, val in ckpt.items():
        if key.startswith(prefix+'.'):
            newkey = key[key.find(".")+1:]

            if val.ndim == 5:
                val = val.permute(4, 0, 1, 2, 3)

            new_ckpt[newkey] = val

    model.load_state_dict(new_ckpt)
    return model

def new_sparse_basic_block_forward(self):
    def sparse_basic_block_forward(x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        self.relu.inplace = False
        out = out.replace_feature(self.relu(out.features))
        return out
    return sparse_basic_block_forward

def fuse_sparse_basic_block(self):
    self.forward = new_sparse_basic_block_forward(self)
    self.conv1.act_type = tv.gemm.Activation.ReLU
    fuse_bn(self.conv1, self.bn1)
    fuse_bn(self.conv2, self.bn2)
    delattr(self, "bn1")
    delattr(self, "bn2")

def fusion_SparseSequential(model, cname, child):
    if isinstance(child, SparseSequential):
        if isinstance(child[0], conv.SparseConvolution) and len(child) >= 3 \
                and isinstance(child[1], NaiveSyncBatchNorm1d):
            c, b, r = [child[i] for i in range(3)]
            fuse_bn(c, b)
            c.act_type = tv.gemm.Activation.ReLU
            if len(child) == 3:
                new_conv = c
            else:
                new_conv = SparseSequential(
                    *([c] + [child[i] for i in range(3, len(child))])
                )
            names = cname.split('.')
            # global model_m
            child_m = model
            for n in names[:-1]:
                child_m = getattr(child_m, n)
            setattr(child_m, names[-1], new_conv)
        else:
            for ccname, cchild in child.named_modules():
                if ccname != "":
                    fusion_SparseSequential(model, '.'.join([cname, ccname]), cchild)

model_m = None
def layer_fusion(model : SimpleSparseUNet):

    # fuse all conv
    global model_m
    model_m = model
    for name, block in model.named_modules():
        if name != "":
            fusion_SparseSequential(model, name, block)

    # fuse all SparseBasicBlock
    for name, block in model_m.named_modules():
        if isinstance(block, SparseBasicBlock):
            fuse_sparse_basic_block(block)
    return model

# This function stores a file that can be very easily loaded and used by c++
def save_tensor(tensor, file):

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().data.numpy()
    elif not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)

    dtype_map = {"float32" : 0, "float16" : 1, "int32" : 2, "int64" : 3}
    if str(tensor.dtype) not in dtype_map:
        raise RuntimeError(f"Unsupport dtype {tensor.dtype}")

    magic_number = 0x33ff1101
    with open(file, "wb") as f:
        head = np.array([magic_number, tensor.ndim, dtype_map[str(tensor.dtype)]], dtype=np.int32).tobytes()
        f.write(head)

        dims = np.array(tensor.shape, dtype=np.int32).tobytes()
        f.write(dims)
        
        data = tensor.tobytes()
        f.write(data)

# This function stores a file that can be very easily loaded and used by c++
def load_tensor(file):

    dtype_for_integer_mapping = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.int64}
    dtype_size_mapping        = {np.float32 : 4, np.float16 : 2, np.int32 : 4, np.int64 : 8}

    with open(file, "rb") as f:
        magic_number, ndim, dtype_integer = np.frombuffer(f.read(12), dtype=np.int32)
        if dtype_integer not in dtype_for_integer_mapping:
            raise RuntimeError(f"Can not find match dtype for index {dtype_integer}")

        dtype            = dtype_for_integer_mapping[dtype_integer]
        magic_number_std = 0x33ff1101
        assert magic_number == magic_number_std, f"this file is not tensor file"
        dims   = np.frombuffer(f.read(ndim * 4), dtype=np.int32)
        volumn = np.cumprod(dims)[-1]
        data   = np.frombuffer(f.read(volumn * dtype_size_mapping[dtype]), dtype=dtype).reshape(*dims)
        return data
