# ================== model ========================
embed_dims = 128
num_groups = 4
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # setup.py needs to be executed
num_levels = 4
drop_out = 0.1
pc_range = [5.0, 1.20, 0, 60.0, 1.95, 2 * 3.1415926535]
scale_range = [0.1, 0.6]

include_opa = True
semantics = True
semantic_dim = 17
phi_activation = 'loop'
xyz_coordinate = 'polar'

model = dict(
    type="GaussianFormerFSDV2",
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    pts_voxel_encoder=dict(
        type='GaussianLifter',
        num_anchor=3600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
    ),
    pts_middle_encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=False,
            unit_xyz=None,
            refine_manual=None,
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softmax',
        ),
        spconv_layer=None,
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=None,
    ),
    # FSDV2 detector config plugged into GaussianFormerFSDV2
    fsdv2_cfg=dict(
        type='SingleStageFSDV2',

        segmentor=dict(
            type='VoteSegmentor',
            tanh_dims=[],
            voxel_layer=dict(
                voxel_size=(0.2, 0.2, 0.2),
                max_num_points=-1,
                point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                max_voxels=(-1, -1)
            ),
            voxel_encoder=dict(
                type='DynamicScatterVFE',
                in_channels=28,
                feat_channels=[64, 64],
                voxel_size=(0.2, 0.2, 0.2),
                with_cluster_center=True,
                with_voxel_center=True,
                point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
                unique_once=True,
            ),
            middle_encoder=dict(
                type='PseudoMiddleEncoderForSpconvFSD',
            ),
            backbone=dict(
                type='SimpleSparseUNet',
                in_channels=64,
                sparse_shape=[40, 512, 512],
                order=('conv', 'norm', 'act'),
                norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
                base_channels=64,
                output_channels=128,
                encoder_channels=((128, ), (128, 128, ), (128, 128, ), (128, 128, 128), (256, 256, 256), (256, 256, 256)),
                encoder_paddings=((1, ), (1, 1, ), (1, 1, ), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),
                decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)),
                return_multiscale_features=True,
            ),
            decode_neck=dict(
                type='Voxel2PointScatterNeck',
                voxel_size=(0.2, 0.2, 0.2),
                point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            ),
            segmentation_head=dict(
                type='VoteSegHead',
                in_channel=67 + 64,
                hidden_dims=[128, 128],
                num_classes=10,
                dropout_ratio=0.0,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='naiveSyncBN1d'),
                act_cfg=dict(type='ReLU'),
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight=[1.0, ] * 10 + [0.1,],
                    loss_weight=10.0),
                loss_vote=dict(
                    type='L1Loss',
                    loss_weight=1.0),
            ),
            train_cfg=dict(
                point_loss=True,
                score_thresh=[0.2, ] * 3 + [0.1, ] * 3,
                class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                             'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'],
                group_names=[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
                group_lens=[1, 2, 2, 1, 2, 2],
            ),
        ),

        virtual_point_projector=dict(
            in_channels=106 + 64,
            hidden_dims=[64, 64],
            norm_cfg=dict(type='naiveSyncBN1d'),
            ori_in_channels=67 + 64,
            ori_hidden_dims=[64, 64],
        ),

        multiscale_cfg=dict(
            multiscale_levels=[0, 1, 2],
            projector_hiddens=[[256, 128], [128, 128], [128, 128]],
            fusion_mode='avg',
            target_sparse_shape=[20, 256, 256],
            norm_cfg=dict(type='naiveSyncBN1d'),
        ),

        voxel_encoder=dict(
            type='DynamicScatterVFE',
            in_channels=67,
            feat_channels=[64, 128],
            voxel_size=(0.4, 0.4, 0.4),
            with_cluster_center=True,
            with_voxel_center=True,
            point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
            unique_once=True,
        ),

        backbone=dict(
            type='VirtualVoxelMixer',
            in_channels=128,
            sparse_shape=[20, 256, 256],
            order=('conv', 'norm', 'act'),
            norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
            base_channels=64,
            output_channels=128,
            encoder_channels=((64, ), (64, 64, ), (64, 64, ), ),
            encoder_paddings=((1, ), (1, 1,), (1, 1,), ),
            decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),
            decoder_paddings=((1, 1), (1, 1), (1, 1),),
        ),

        bbox_head=dict(
            type='FSDV2Head',
            num_classes=10,
            bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=4.0),
            loss_center=dict(type='L1Loss', loss_weight=0.5),
            loss_size=dict(type='L1Loss', loss_weight=0.5),
            loss_rot=dict(type='L1Loss', loss_weight=0.2),
            loss_vel=dict(type='L1Loss', loss_weight=0.2),
            in_channel=128,
            shared_mlp_dims=[256, 256],
            train_cfg=None,
            test_cfg=None,
            norm_cfg=dict(type='naiveSyncBN1d'),
            tasks=[
                dict(class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle']),
                dict(class_names=['bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']),
            ],
            class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'],
            common_attrs=dict(
                center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128)
            ),
            num_cls_layer=2,
            cls_hidden_dim=128,
            separate_head=dict(
                type='FSDSeparateHead',
                norm_cfg=dict(type='naiveSyncBN1d'),
                act='relu',
            ),
        ),
        
        train_cfg=dict(
            score_thresh=[0.2, ] * 3 + [0.1, ] * 3,
            sync_reg_avg_factor=True,
            batched_group_sample=True,
            offset_weight='max',
            class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'],
            group_names=[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
            centroid_assign=True,
            disable_pretrain=True,
            disable_pretrain_topks=[500, ] * 10,
        ),
        test_cfg=dict(
            score_thresh=[0.2, ] * 3 + [0.1, ] * 3,
            batched_group_sample=True,
            offset_weight='max',
            class_names=['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'],
            group_names=[['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.05,
            score_thr=0.05,#0.05
            min_bbox_size=0,
            max_num=500,#500/250
            all_task_max_num=500,
        ),
    ),
)


