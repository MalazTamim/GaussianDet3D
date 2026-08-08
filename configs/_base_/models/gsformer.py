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
    type="GaussianFormer",
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
    # Use these for NuScenes!
pointpillars_cfg=dict(
      type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=[0.25, 0.25, 8],
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=19,#dimensions of the point features
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.25, 0.25, 8],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    # pts_neck=dict(
    #     type='FPN',
    #     norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
    #     act_cfg=dict(type='ReLU'),
    #     in_channels=[64, 128, 256],
    #     out_channels=256,
    #     start_level=0,
    #     num_outs=3),
    pts_neck=dict(
        # _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=384,#256
        feat_channels=384,#256
        use_direction_classifier=True,
        # anchor_generator=dict(
        #     type='AlignedAnchor3DRangeGenerator',
        #     ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
        #     scales=[1, 2, 4],
        #     sizes=[
        #         [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
        #         [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
        #         [1., 1., 1.],
        #         [0.4, 0.4, 1],
        #     ],
        #     custom_values=[0, 0],
        #     rotations=[0, 1.57],
        #     reshape_out=True),
        anchor_generator=dict(
            # _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
            ],
            sizes=[
                [4.60718145, 1.95017717, 1.72270761],  # car
                [6.73778078, 2.4560939, 2.73004906],  # truck
                [12.01320693, 2.87427237, 3.81509561],  # trailer
                [1.68452161, 0.60058911, 1.27192197],  # bicycle
                [0.7256437, 0.66344886, 1.75748069],  # pedestrian
                [0.40359262, 0.39694519, 1.06232151],  # traffic_cone
                [0.48578221, 2.49008838, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
  

  # ✅ THESE TWO BLOCKS FIX THE CRASH
      train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        pts=dict(
            # use_rotate_nms=True,
            # nms_across_levels=False,
            # nms_thr=0.2,
            # score_thr=0.05,
            # min_bbox_size=0,
            # max_num=300
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500
        )
    )

    
),
    # pts_bbox_head=dict(
    #     type='GaussianHead',
        # apply_loss_type=None,
        # num_classes=17,
        # empty_args=None,
        # with_empty=False,
        # cuda_kwargs=None,
    # )

   
        
)
