# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
import torch

from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, "Not implemented yet."
        assert plugins is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False
        )
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ["after_conv1", "after_conv2", "after_conv3"]
            assert all(p["position"] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv1"
            ]
            self.after_conv2_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv2"
            ]
            self.after_conv3_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv3"
            ]

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3
        )

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins
            )
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins
            )
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins
            )

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin, in_channels=in_channels, postfix=plugin.pop("postfix", "")
            )
            assert not hasattr(self, name), f"duplicate plugin {name}"
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        Fusion=None,
        fusion_channels=3,
        attention_augmented=False,
        modality_gates=False,
        separate_gates=False,
        modality_gates_act="relu",
        gate_mode="separate",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        block_init_cfg = None
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be specified at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type="Constant", val=0, override=dict(name="norm2")
                        )
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type="Constant", val=0, override=dict(name="norm3")
                        )
        else:
            raise TypeError("pretrained must be a str or None")

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        self.Fusion = Fusion
        self.fusion_channels = fusion_channels
        self.attention_augmented = attention_augmented
        self.modality_gates = modality_gates
        self.separate_gates = separate_gates
        self.modality_gates_act = modality_gates_act
        self.gate_mode = gate_mode
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.in_channels = in_channels

        self._make_stem_layer(self.in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        num_filters = self.in_channels

        self.feat_dim = (
            self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)
        )

        pds = 1
        KernelS = 3
        self.bev_fc1 = nn.ModuleList()
        self.cyl_fc1 = nn.ModuleList()
        self.bev_fc2 = nn.ModuleList()
        self.cyl_fc2 = nn.ModuleList()
        self.bev_att_path = nn.ModuleList()
        self.cyl_att_path = nn.ModuleList()
        self.fusion_transform = nn.ModuleList()
        self.fusion_transform_MFB = nn.ModuleList()
        self.combine_trans = nn.ModuleList()
        # for i, num_blocks in enumerate(self.stage_blocks):

        for i in range(1):
            NUM_CHANNELS = self.fusion_channels
            num_filters = 2 * NUM_CHANNELS * 2**i
            inoutfilter = NUM_CHANNELS * 2**i
            self.bev_fc1.append(
                nn.Sequential(
                    nn.Conv2d(
                        inoutfilter,
                        num_filters,
                        KernelS,
                        stride=1,
                        padding=pds,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_filters),
                    # nn.ReLU(inplace=True)
                )
            )

            self.cyl_fc1.append(
                nn.Sequential(
                    nn.Conv2d(
                        inoutfilter,
                        num_filters,
                        KernelS,
                        stride=1,
                        padding=pds,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_filters),
                    # nn.ReLU(inplace=True)
                )
            )

            self.bev_fc2.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_filters,
                        num_filters,
                        KernelS,
                        stride=1,
                        padding=pds,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_filters),
                )
            )

            self.cyl_fc2.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_filters,
                        num_filters,
                        KernelS,
                        stride=1,
                        padding=pds,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_filters),
                )
            )

            self.bev_att_path.append(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                )
            )

            self.cyl_att_path.append(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_filters, num_filters, KernelS, stride=1, padding=pds),
                )
            )

            ch_in = num_filters * 2
            self.fusion_transform.append(
                nn.Sequential(
                    nn.Conv2d(
                        ch_in, inoutfilter, KernelS, stride=1, padding=pds, bias=False
                    ),
                    nn.BatchNorm2d(inoutfilter),
                    nn.ReLU(inplace=True),
                )
            )
            self.fusion_transform_MFB.append(
                nn.Sequential(
                    nn.Conv2d(
                        ch_in, num_filters, KernelS, stride=1, padding=pds, bias=False
                    ),
                    nn.BatchNorm2d(num_filters),
                )
            )

            self.combine_trans.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_filters,
                        inoutfilter,
                        KernelS,
                        stride=1,
                        padding=pds,
                        bias=False,
                    ),
                    nn.BatchNorm2d(inoutfilter)
                    # nn.ReLU(inplace=True)
                )
            )

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

        if self.Fusion == "BON":
            self.r_conv_1x1 = self.__create_conv_1x1(
                in_channels=1, out_channels=3, act=nn.ReLU(inplace=True)
            )
            self.g_conv_1x1 = self.__create_conv_1x1(
                in_channels=1, out_channels=3, act=nn.ReLU(inplace=True)
            )
            self.b_conv_1x1 = self.__create_conv_1x1(
                in_channels=1, out_channels=3, act=nn.ReLU(inplace=True)
            )

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

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def pnorm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.pnorm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1
            )
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        INPUT_CHANNELS = 6 # first 3 thermal, next 3 rgb, last 1 thermal_grayscale
        if type(x) is list:
            tensor = x[0].reshape(1, INPUT_CHANNELS, 800, 960)
            for i in range(1, len(x)):
                tensor = torch.cat((tensor, x[i].reshape(1, INPUT_CHANNELS, 800, 960)), 0)
            x = tensor.float()
        else:
            x = x.reshape(1, INPUT_CHANNELS, 800, 960).float()
        # if(x.shape[1]==6):
        # x1=x[0]
        # x2=x[1]

        # print("------------------------------------")
        # print("x.shape", x.shape)
        # print("x.min()", x.min())
        # print("x.max()", x.max())

        x1 = x[:, :3]
        x2 = x[:, 3:6]
        # x3 = x[:, 6:]

        # print('x3.shape', x3.shape, self.Fusion, 'min', x3.min(), 'max', x3.max())

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

        # print("------------------------------------")
        # print("x1.shape", x1.shape)
        # print("x1.min()", x1.min())
        # print("x1.max()", x1.max())
        # print("x2.shape", x2.shape)
        # print("x2.min()", x2.min())
        # print("x2.max()", x2.max())

        if self.Fusion == "thermal":
            # c1, c2, c3 = torch.split(x1, 1, dim=1)
            # print('c1', 'min', c1.cpu().numpy().min(), 'max', c1.cpu().numpy().max(), 'mean', c1.cpu().numpy().mean(), 'std', c1.cpu().numpy().std())
            # print('c2', 'min', c2.cpu().numpy().min(), 'max', c2.cpu().numpy().max(), 'mean', c2.cpu().numpy().mean(), 'std', c2.cpu().numpy().std())
            # print('c3', 'min', c3.cpu().numpy().min(), 'max', c3.cpu().numpy().max(), 'mean', c3.cpu().numpy().mean(), 'std', c3.cpu().numpy().std())
            # print('x1', 'min', x1.cpu().numpy().min(), 'max', x1.cpu().numpy().max(), 'mean', x1.cpu().numpy().mean(), 'std', x1.cpu().numpy().std())

            x = x1
        elif self.Fusion == "rgb":
            x = x2
            # print('x', 'min', x.cpu().numpy().min(), 'max', x.cpu().numpy().max(), 'mean', x.cpu().numpy().mean(), 'std', x.cpu().numpy().std())
        elif self.Fusion == "ADD":
            x = torch.add(x1, x2)
            v = 1
        elif self.Fusion == "MUL":
            x = torch.mul(x1, x2)
        elif self.Fusion == "CONCAT":
            x = x
        elif self.Fusion == "MFB":
            x = self.MFBv2(x[:, :3], x[:, 3:], 0)
        elif self.Fusion == "BGF":
            x = self.trad_fusion(x[:, :3], x[:, 3:], 0)
        elif self.Fusion == "BON":
            x = self.bonferroni(x1, x2)
        elif self.Fusion is None:
            x = x1
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def bonferroni(self, rgb, thermal):
        r = self.r_conv_1x1(rgb[:, 0:1])
        g = self.g_conv_1x1(rgb[:, 1:2])
        b = self.b_conv_1x1(rgb[:, 2:3])

        inputs = [rgb, r, g, b, thermal]
        a = []

        for i in range(len(inputs)):
            a.append(
                torch.stack(inputs[0:i] + inputs[i + 1 :], dim=1).sum(dim=1)
                * 0.25
                * inputs[i]
            )

        fused = torch.stack(a, dim=1).sum(dim=1) * 0.2
        # fused = torch.relu(fused) ** 0.5

        # print(fused.min(), fused.max())

        return fused

    def trad_fusion(self, RGB, LIDAR, nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
        att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))
        cyl_x2 = self.cyl_fc2[nm](cyl_x1)
        bev_x2 = self.bev_fc2[nm](bev_x1)
        pt_cyl_pre_fusion = cyl_x1 + att_bev_to_cyl * cyl_x2
        pt_bev_pre_fusion = bev_x1 + att_cyl_to_bev * bev_x2
        point_features = torch.cat([pt_cyl_pre_fusion, pt_bev_pre_fusion], dim=1)
        conv_features = self.fusion_transform[nm](point_features)
        # conv_features = conv_features.squeeze(0).transpose(0, 1)
        return conv_features

    def MFB(self, RGB, LIDAR, nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        att_cyl_to_bev = torch.sigmoid(self.cyl_att_path[nm](cyl_x1))
        att_bev_to_cyl = torch.sigmoid(self.bev_att_path[nm](bev_x1))
        cyl_x2 = self.cyl_fc2[nm](cyl_x1)
        bev_x2 = self.bev_fc2[nm](bev_x1)
        pt_cyl_pre_fusion = cyl_x1 + att_bev_to_cyl * cyl_x2
        pt_bev_pre_fusion = bev_x1 + att_cyl_to_bev * bev_x2
        feat_aux = torch.add(pt_cyl_pre_fusion, pt_bev_pre_fusion)
        inter = torch.mul(pt_cyl_pre_fusion, pt_bev_pre_fusion)
        combined = self.combine_trans[nm](inter)
        point_features = torch.cat([combined, feat_aux], dim=1)
        out = self.fusion_transform_MFB[nm](point_features)
        out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(
            nn.ReLU(inplace=True)(-out)
        )
        power_normed = torch.mul(out, torch.sign(out))
        l2_normed = power_normed / torch.norm(power_normed)
        # l2_normed = l2_normed.squeeze(0).transpose(0, 1)
        return l2_normed

    def MFBv2(self, RGB, LIDAR, nm):
        cyl_x1 = self.cyl_fc1[nm](RGB)
        bev_x1 = self.bev_fc1[nm](LIDAR)
        F_aux = torch.add(cyl_x1, bev_x1)
        inter = torch.mul(cyl_x1, bev_x1)
        inter = nn.Dropout(p=0.1)(inter)
        F = self.combine_trans[nm](inter)
        # m = nn.AvgPool2d(2, stride=1, divisor_override=1)
        out = F
        out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(
            nn.ReLU(inplace=True)(-out)
        )
        l2_normed = torch.nn.functional.normalize(out)
        return l2_normed

    def MFBv3(self, RGB, LIDAR, nm):
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
        out = torch.sqrt(nn.ReLU(inplace=True)(out)) + torch.sqrt(
            nn.ReLU(inplace=True)(-out)
        )
        l2_normed = torch.nn.functional.normalize(out)
        return l2_normed

    def MFBFUS(self, feats1, feats2):
        *input_size, _ = feats1.size()  # (b, m, h)
        feats1 = feats1.contiguous().view(-1, self.hidden_size)  # (b*m, h)
        feats2 = feats2.contiguous().view(-1, self.hidden_size)  # (b*m, h)
        feats1 = self.fc1(feats1)  # (b*m, h*factor)
        feats2 = self.fc2(feats2)  # (b*m, h*factor)
        feats = torch.mul(feats1, feats2)  # (b*m, h*factor)
        # feats = self.drop_out(feats)  # (b*m, h*factor)
        feats = feats.view(
            -1, self.hidden_size, self.MFB_FACTOR_NUM
        )  # (b*m, h, factor)
        feats = torch.sum(feats, 2)  # sum pool, (b*m, h)
        feats = torch.sqrt(torch.nn.functional.relu(feats)) - torch.sqrt(
            torch.nn.functional.relu(-feats)
        )  # signed sqrt, (b*m, h)
        feats = torch.nn.functional.normalize(feats)  # (b*m, h)
        feats = feats.view(*input_size, self.hidden_size)  # (b, m, h)
        return feats


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(deep_stem=True, avg_down=True, **kwargs)


class AugmentedConv(nn.Module):
    """
    file:///Y:/uOttawa/Publications/PBVS'23/2107.14391.pdf
    https://ieeexplore.ieee.org/document/9165005
    https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/1ce94a3072c2d9aabe258313b3a17c974d987411/AA-Wide-ResNet/attention_augmented_conv.py#L9
    https://arxiv.org/pdf/1904.09925v5.pdf
    https://paperswithcode.com/method/attention-augmented-convolution#:~:text=Attention%2Daugmented%20Convolution%20is%20a,head%20attention%20as%20with%20Transformers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dk,
        dv,
        Nh,
        shape=0,
        relative=False,
        stride=1,
    ):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(
            self.in_channels,
            self.out_channels - self.dv,
            self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )
            self.key_rel_h = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(
            x, self.dk, self.dv, self.Nh
        )
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(
            attn_out, (batch, self.Nh, self.dv // self.Nh, height, width)
        )
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh**-0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(
            torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h"
        )

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum("bhxyd,md->bhxym", q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = (
                torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
            )
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x
