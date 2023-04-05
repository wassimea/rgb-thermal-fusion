_base_ = [
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/schedule_2x.py",
    "../_base_/default_runtime.py",
]
model = dict(
    type="TOOD",
    # pretrained=None,
    init_cfg=dict(type="Pretrained", checkpoint="https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_1x_coco/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth"),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        Fusion="ADD",
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style="pytorch",
        # init_cfg=None,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        type="TOODHead",
        num_classes=1,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type="anchor_free",
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        initial_loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_cls=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
    ),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type="ATSSAssigner", topk=9),
        assigner=dict(type="TaskAlignedAssigner", topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)

# custom hooks
custom_hooks = [dict(type="SetEpochInfoHook")]

batch_size: 1

runner = dict(type="EpochBasedRunner", max_epochs=12)

default_mean = [123.675, 116.28, 103.53]
default_std = [58.395, 57.12, 57.375]
rgb_mean = [93.59200385, 93.71329788, 93.73038564]
rgb_std = [87.42839991, 87.07733285, 87.25576906]#[74.2528371, 73.89367641, 74.10238738]#
thermal_mean = [135.93606275, 135.93606275, 135.93606275]
thermal_std = [49.13574314, 49.13574314, 49.13574314]
min_max = False

# Modify dataset related settings
dataset_type = "COCODataset"
classes = ("car",)
data = dict(
    train=dict(
        img_prefix="/media/yalaa/Storage/datasets/Thermal_RGB_SC/thermal/train/",
        classes=classes,
        ann_file="/media/yalaa/Storage/datasets/Thermal_RGB_SC/coco_train.json",
    ),
    val=dict(
        img_prefix="/media/yalaa/Storage/datasets/Thermal_RGB_SC/thermal/val/",
        classes=classes,
        ann_file="/media/yalaa/Storage/datasets/Thermal_RGB_SC/coco_val.json",
    ),
    test=dict(
        img_prefix="/media/yalaa/Storage/datasets/Thermal_RGB_SC/thermal/val/",
        classes=classes,
        ann_file="/media/yalaa/Storage/datasets/Thermal_RGB_SC/coco_val.json",
    ),
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
