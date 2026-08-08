_base_ = [
    '../_base_/schedules/schedule_2x_unfrozen.py',
    '../_base_/models/gsformer_fsdv2.py',
    '../_base_/datasets/nusc-10class-images_augmentation.py',
    '../_base_/default_runtime.py',
]

# =========== data config ==============
find_unused_parameters=True#Added to avoid unused parameters error in distributed training
input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
val_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
train_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
# =========== misc config ==============
# optimizer = dict(
#     optimizer = dict(
#         type="AdamW", lr=4e-4, weight_decay=0.01,
#     ),
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.1),
#             # 'fsdv2_cfg': dict(lr_mult=0.1),
#             }
#     )
# )
# grad_max_norm = 35


optimizer = dict(
    type='AdamW',
    lr=4e-4,  # Default learning rate
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

num_classes = len(class_names)
group1 = ['car']
group2 = ['truck', 'construction_vehicle']
group3 = ['bus', 'trailer']
group4 = ['barrier']
group5 = ['motorcycle', 'bicycle']
group6 = ['pedestrian', 'traffic_cone']
group_names=[group1, group2, group3, group4, group5, group6]

seg_score_thresh = [0.2, ] * 3 + [0.1, ] * 3
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]

head_group1 = class_names[:5]
head_group2 = class_names[5:]
tasks=[
    dict(class_names=head_group1),
    dict(class_names=head_group2),
]

# ========= model config ===============
embed_dims = 128
num_decoder = 4
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
fsdv2_input_point_cloud_range = [-50.0, -50.0, -4.99, 50.0, 50.0, 2.99]
# fsdv2_input_point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]


# scale_range = [0.01, 2.5]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
# load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
load_from = 'ckpts/converted_gsformer_prob_256.pth'
# load_from = 'ckpts/merged_gsformer_fsdv2.pth'


semantics = True
semantic_dim = 17


workflow = [('train', 1)]
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)


model = dict(
    type='GaussianFormerFSDV2MultiFrame',
    prev_samples_num=3,
    ts_index=27,
    point_cloud_range=fsdv2_input_point_cloud_range,
    
    freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # frozen_stages=4, # freeze all layers
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        start_level=1),
    pts_voxel_encoder=dict(
        type='GaussianLifterV2',#GaussianLifterV2
        num_anchor=19200,#4000/6400,
        embed_dims=embed_dims,
        anchor_grad=False,
        feat_grad=False,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                # frozen_stages=4, # freeze all layers
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                # upsample_strides=[0.5, 1, 2, 4]
                upsample_strides=[0.5, 1, 2, 4]

                )),
        initializer_img_downsample=None,
        pretrained_path="out/prob/init/init.pth",
        deterministic=False,#False
        random_samples=6400),#2400
    pts_middle_encoder=dict(
        type='GaussianOccEncoder',
        freeze_encoder= False,  # ✅ added flag
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),#returns embeddings of the gaussians of ouptut shape: torch.Size([1, 6400, 128])
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            _delete_=True,
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False,
        ),#returns features of the same ouptut shape as the embeddings: torch.Size([1, 6400, 128])
        deformable_model=dict(
            embed_dims=embed_dims,
            residual_mode="none",
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0,
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[1.0, 1.0, 1.0],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True,
        ),
        num_decoder=num_decoder,
        operation_order=[
            "identity",
            "deformable",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",

            "identity",
            "spconv",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",
            
            "refine",
        ] * num_decoder,
    ),


    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.05,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500
        )
    )
)

# =========== dataset pipeline override ==============
_extra_meta_keys = (
    'scene_token',
    'sample_token',
    'prev_sample_tokens',
    'prev_sensor2lidar_R',
    'prev_sensor2lidar_t',
    'timestamp',
    'ego2global_rotation',
    'ego2global_translation',
    'lidar2ego_rotation',
    'lidar2ego_translation',
)

_train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='ResizeCropFlipImage',
         H=900, W=1600, final_dim=(864, 1600),
         bot_pct_lim=(0.0, 0.0), resize_lim=(1.0, 1.0),
         rot_lim=(0.0, 0.0), rand_flip=True, training=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage",
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'lidar2img'],
         extra_meta_keys=_extra_meta_keys),
]

_val_test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='ResizeCropFlipImage',
         H=900, W=1600, final_dim=(864, 1600),
         bot_pct_lim=(0.0, 0.0), resize_lim=(1.0, 1.0),
         rot_lim=(0.0, 0.0), rand_flip=False, training=False),
    dict(type="NormalizeMultiviewImage",
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D',
                  keys=['img', 'lidar2img'],
                  extra_meta_keys=_extra_meta_keys),
         ]),
]

data = dict(
    use_seq_code=True,
    training_shuffle=False,
    train=dict(
        _delete_=True,
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        load_interval=1,
        pipeline=_train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        box_type_3d='LiDAR',
    ),
    val=dict(pipeline=_val_test_pipeline),
    test=dict(pipeline=_val_test_pipeline),
)
