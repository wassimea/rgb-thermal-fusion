# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import (NORM_LAYERS, DropPath, build_activation_layer,
                             build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.runner.base_module import ModuleList, Sequential

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@NORM_LAYERS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)
            x = self.norm(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

            x = self.pointwise_conv1(x)
            x = self.act(x)
            x = self.pointwise_conv2(x)

            if self.linear_pw_conv:
                x = x.permute(0, 3, 1, 2)  # permute back

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


@BACKBONES.register_module()
class ConvNeXt(BaseBackbone):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 Fusion=None,
                 modality_gates=False,
                 separate_gates=False,
                 modality_gates_act="relu",
                 gate_mode="separate",
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.Fusion = Fusion
        self.modality_gates = modality_gates
        self.separate_gates = separate_gates
        self.modality_gates_act = modality_gates_act
        self.gate_mode = gate_mode

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()
        num_filters=in_channels

        pds=1
        KernelS =3
        self.bev_fc1 =nn.ModuleList()
        self.cyl_fc1 =nn.ModuleList()
        self.bev_fc2=nn.ModuleList()
        self.cyl_fc2=nn.ModuleList()
        self.bev_att_path =nn.ModuleList()
        self.cyl_att_path=nn.ModuleList()
        self.fusion_transform =nn.ModuleList()
        self.fusion_transform_MFB=nn.ModuleList()
        self.combine_trans =nn.ModuleList()
        #for i, num_blocks in enumerate(self.stage_blocks): 
        
        for i in range(1): 
                num_filters  =  6 * 2**i
                inoutfilter = 3 * 2**i
                self.bev_fc1.append(nn.Sequential(
                    nn.Conv2d(inoutfilter, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                    #nn.ReLU(inplace=True)
                ))
                self.cyl_fc1 .append(nn.Sequential(
                    nn.Conv2d(inoutfilter, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                    #nn.ReLU(inplace=True)
                ))
                self.bev_fc2 .append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                ))
                self.cyl_fc2 .append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters),
                ))
         
                self.bev_att_path .append( nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                ))
                self.cyl_att_path .append(nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                ))
                ch_in = num_filters * 2
                self.fusion_transform .append( nn.Sequential(
                    nn.Conv2d(ch_in, inoutfilter, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(inoutfilter),
                    nn.ReLU(inplace=True),
                ))
                self.fusion_transform_MFB .append(nn.Sequential(
                    nn.Conv2d(ch_in, num_filters, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(num_filters)
                ))
         
                self.combine_trans.append( nn.Sequential(
                    nn.Conv2d(num_filters, inoutfilter, KernelS, stride=1, padding=pds, bias=False),
                    nn.BatchNorm2d(inoutfilter)
                    #nn.ReLU(inplace=True)
                ))

        if self.modality_gates:
            if self.modality_gates_act == "sigmoid":
                gate_act = nn.Sigmoid()
            elif self.modality_gates_act == "relu":
                gate_act = nn.ReLU(inplace=False)
            elif self.modality_gates_act == "noact":
                gate_act = 'noact'
            if self.separate_gates:
                if self.gate_mode == "enhanced":
                    self.gate_rgb = self.__create_conv_gate(gate_act=gate_act)
                    self.gate_thermal = self.__create_conv_gate(gate_act=gate_act)
                elif self.gate_mode == "separate":
                    self.gate_rgb = self.__create_conv_gate(gate_act=gate_act)
                    self.gate_thermal = self.__create_conv_gate(gate_act=gate_act)
                elif self.gate_mode == 'rgb_only':
                    self.r_gate_1x1 = self.__create_conv_1x1(
                        in_channels=1, out_channels=1, act=gate_act
                    )
                    self.g_gate_1x1 = self.__create_conv_1x1(
                        in_channels=1, out_channels=1, act=gate_act
                    )
                    self.b_gate_1x1 = self.__create_conv_1x1(
                        in_channels=1, out_channels=1, act=gate_act
                    )
            else:
                self.gate = self.__create_conv_gate(gate_act=gate_act)

    def __create_conv_1x1(self, in_channels, out_channels, act):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
                bias=True,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            act,
        )

    def __create_conv_gate(self, gate_act):
        if gate_act == 'noact':
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    padding_mode="zeros",
                    bias=True,
                ),
                nn.BatchNorm2d(num_features=3),
            )
        return nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
                bias=True,
            ),
            nn.BatchNorm2d(num_features=3),
            gate_act,
        )

    @property
    def pnorm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.pnorm1_name)

    def forward(self, x):
        if type(x) is list:
            tensor = x[0].reshape(1,6,800,960)
            for i in range(1, len(x)):
                tensor = torch.cat((tensor, x[i].reshape(1,6,800,960)), 0)
            x=tensor.float()
        else:
            x = x.reshape(1,6,800,960).float()
        #if(x.shape[1]==6):
        #x1=x[0]        
        #x2=x[1]
        x1=x[:,:3]
        x2=x[:,3:6]

        if self.modality_gates:
            if self.separate_gates:
                if self.gate_mode == "enhanced":
                    x1 = self.gate_rgb(x1) * x1
                    x2 = self.gate_thermal(x2) * x2
                elif self.gate_mode == "separate":
                    x1 = self.gate_rgb(x1)
                    x2 = self.gate_thermal(x2)
                elif self.gate_mode == "rgb_only":
                    r = self.r_gate_1x1(x1[:,0:1])
                    g = self.g_gate_1x1(x1[:,1:2])
                    b = self.b_gate_1x1(x1[:,2:3])

                    x1 = x1 * torch.cat((r,g,b),dim=1)
                    x2 = x2
            else:
                x1 = self.gate(x1)
                x2 = self.gate(x2)

        if self.Fusion == 'thermal':
            x = x1
        elif self.Fusion == 'rgb':
            x = x2
        elif self.Fusion == 'ADD':
            x = torch.add(x1, x2)
            v = 1
        elif self.Fusion == 'MUL':
            x = torch.mul(x1, x2)        
        elif self.Fusion == 'CONCAT':
            x = x
        elif self.Fusion == 'MFB':
            x = self.MFBv2(x[:,:3],x[:,3:],0)              
        elif self.Fusion == 'BGF':
            x = self.trad_fusion(x[:,:3],x[:,3:],0)
        elif self.Fusion is None:
            x =x1
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()


    def trad_fusion(self, RGB, LIDAR,nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
        att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))
        cyl_x2 = self.cyl_fc2[nm](cyl_x1)
        bev_x2 = self.bev_fc2[nm](bev_x1)
        pt_cyl_pre_fusion = (cyl_x1 + att_bev_to_cyl * cyl_x2)
        pt_bev_pre_fusion = (bev_x1 + att_cyl_to_bev * bev_x2)
        point_features = torch.cat([pt_cyl_pre_fusion, pt_bev_pre_fusion], dim=1)
        conv_features = self.fusion_transform[nm](point_features)
        #conv_features = conv_features.squeeze(0).transpose(0, 1)
        return conv_features
    def MFB(self, RGB, LIDAR,nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
        att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))
        cyl_x2 = self.cyl_fc2[nm](cyl_x1)
        bev_x2 = self.bev_fc2[nm](bev_x1)
        pt_cyl_pre_fusion =  (cyl_x1 + att_bev_to_cyl * cyl_x2)
        pt_bev_pre_fusion =  (bev_x1 + att_cyl_to_bev * bev_x2)
        feat_aux = torch.add(pt_cyl_pre_fusion, pt_bev_pre_fusion)
        inter = torch.mul(pt_cyl_pre_fusion, pt_bev_pre_fusion)
        combined = self.combine_trans[nm](inter)
        point_features = torch.cat([combined, feat_aux], dim=1)
        out = self.fusion_transform_MFB[nm](point_features)
        out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(nn.ReLU(inplace=True)(-out))
        power_normed = torch.mul(out, torch.sign(out))
        l2_normed = power_normed/torch.norm(power_normed)
        #l2_normed = l2_normed.squeeze(0).transpose(0, 1)
        return l2_normed
    def MFBv2(self, RGB, LIDAR,nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        F_aux = torch.add(cyl_x1, bev_x1)
        inter = torch.mul(cyl_x1, bev_x1)
        inter = nn.Dropout(p=0.1)(inter)
        F = self.combine_trans[nm](inter)
        #m = nn.AvgPool2d(2, stride=1, divisor_override=1)
        out =  F
        out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(nn.ReLU(inplace=True)(-out))
        l2_normed = torch.nn.functional.normalize( out) 
        return l2_normed
    def MFBv3(self, RGB, LIDAR,nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
        att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))
        cyl_x2 = self.cyl_fc2[nm](cyl_x1)
        bev_x2 = self.bev_fc2[nm](bev_x1)
 
        feat_aux = torch.add(att_cyl_to_bev, att_bev_to_cyl)
        inter = torch.mul(att_cyl_to_bev, att_cyl_to_bev)
        combined = self.combine_trans[nm](inter)
        point_features = torch.cat([combined, feat_aux], dim=1)
        out = self.fusion_transform_MFB[nm](point_features)
        out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(nn.ReLU(inplace=True)(-out))
        l2_normed = torch.nn.functional.normalize( out)  
        return l2_normed           
    def MFBFUS(self, feats1, feats2):
        *input_size, _ = feats1.size() # (b, m, h)
        feats1 = feats1.contiguous().view(-1, self.hidden_size) # (b*m, h)
        feats2 = feats2.contiguous().view(-1, self.hidden_size) # (b*m, h)
        feats1 = self.fc1(feats1)  # (b*m, h*factor)
        feats2 = self.fc2(feats2)  # (b*m, h*factor)
        feats = torch.mul(feats1, feats2)  # (b*m, h*factor)
        #feats = self.drop_out(feats)  # (b*m, h*factor)
        feats = feats.view(-1, self.hidden_size, self.MFB_FACTOR_NUM) # (b*m, h, factor)
        feats = torch.sum(feats, 2)  # sum pool, (b*m, h)
        feats = torch.sqrt(torch.nn.functional.relu(feats)) - torch.sqrt(torch.nn.functional.relu(-feats))  # signed sqrt, (b*m, h)
        feats = torch.nn.functional.normalize(feats) # (b*m, h)
        feats = feats.view(*input_size, self.hidden_size) # (b, m, h)
        return feats