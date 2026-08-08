import torch
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from .centerpoint import CenterPoint


@DETECTORS.register_module()
class GaussianFormerFSDV2MultiFrame(CenterPoint):
    """GaussianFormer + FSDV2 with rolling Gaussian buffer for temporal fusion.

    Processing model (streaming / scene-sequential):
      - Frames are fed in temporal order within each scene (enforced by the
        NuScenesSceneSequentialSampler in seq_training_apis.py).
      - After each forward pass the current frame's Gaussians are stored in a
        small CPU-side rolling buffer keyed by sample_token (max 3 entries).
      - On the next forward pass the model looks up the prev tokens from
        img_metas['prev_sample_tokens'] and retrieves the cached Gaussians
        instantly — **zero extra encoder calls**.
      - The backbone + lifter + encoder run exactly ONCE per sample per epoch.

    Buffer layout:
        self._gaussian_buffer: dict[sample_token → (gaussians_cpu_fp16, pose_meta)]
        pose_meta holds lidar2ego_* and ego2global_* for that frame so that
        ego-motion compensation can be applied when the entry is consumed.
    """

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        fsdv2_cfg=None,
        prev_samples_num=3,
        ts_index=27,
        point_cloud_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if fsdv2_cfg is not None:
            self.fsdv2 = builder.build_detector(fsdv2_cfg)
        else:
            raise ValueError("FSDV2 configuration must be provided.")

        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.prev_samples_num = prev_samples_num
        self.ts_index = ts_index
        self.point_cloud_range = point_cloud_range

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.pts_voxel_encoder.requires_grad_(False)
            if hasattr(self.pts_voxel_encoder, "random_anchors"):
                self.pts_voxel_encoder.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            from mmdet3d.models import build_backbone
            self.extra_img_backbone = build_backbone(extra_img_backbone)

        # Rolling Gaussian buffer: sample_token → (tensor_fp16, pose_meta)
        self._gaussian_buffer: dict = {}
        self._last_scene_token: str = None

    # ------------------------------------------------------------------
    # Image feature extraction
    # ------------------------------------------------------------------

    def extract_img_feat(self, imgs, **kwargs):
        result = {}
        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)
        if imgs.ndim == 6:
            imgs = imgs.squeeze(1)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)

        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result

    def forward_extra_img_backbone(self, imgs, **kwargs):
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        out = []
        for feat in img_feats_backbone:
            BN, C, H, W = feat.size()
            out.append(feat.view(B, int(BN / B), C, H, W))
        return out

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_mat(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.stack([
            1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
                2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
                2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ]).reshape(3, 3)

    @staticmethod
    def _lidar2global(rot_l2e, trans_l2e, rot_e2g, trans_e2g, device):
        import numpy as np
        rot_l2e = np.array(rot_l2e)
        rot_e2g = np.array(rot_e2g)

        t_l2e = torch.tensor(trans_l2e, dtype=torch.float64, device=device)
        t_e2g = torch.tensor(trans_e2g, dtype=torch.float64, device=device)
        q_l2e = torch.tensor(rot_l2e, dtype=torch.float64, device=device)
        q_e2g = torch.tensor(rot_e2g, dtype=torch.float64, device=device)

        R_l2e = GaussianFormerFSDV2MultiFrame._quat_to_mat(q_l2e) if q_l2e.shape == torch.Size([4]) else q_l2e.reshape(3, 3)
        R_e2g = GaussianFormerFSDV2MultiFrame._quat_to_mat(q_e2g) if q_e2g.shape == torch.Size([4]) else q_e2g.reshape(3, 3)

        R = R_e2g @ R_l2e
        t = R_e2g @ t_l2e + t_e2g
        return R.float(), t.float()

    def _transform_prev_to_cur(self, prev_pts, cur_meta, prev_pose_meta, device,
                                s2l_R=None, s2l_t=None):
        """Transform Gaussians xyz from prev LiDAR frame → current LiDAR frame.

        If `s2l_R` and `s2l_t` (precomputed `sensor2lidar_rotation/translation`
        from the current sample's `.pkl` info) are provided, use them directly —
        same single-composite-transform math as offline LoadPointsFromMultiSamples.
        Otherwise fall back to recomposing via lidar2ego/ego2global at runtime.
        """
        pts = prev_pts.to(device).clone()
        xyz = pts[:, :3].double()

        if s2l_R is not None and s2l_t is not None:
            R = torch.as_tensor(s2l_R, dtype=torch.float64, device=device)
            t = torch.as_tensor(s2l_t, dtype=torch.float64, device=device)
            xyz_cur_lidar = xyz @ R.T + t
        else:
            R_cur, t_cur = self._lidar2global(
                cur_meta['lidar2ego_rotation'],       cur_meta['lidar2ego_translation'],
                cur_meta['ego2global_rotation'],      cur_meta['ego2global_translation'],
                device,
            )
            R_prev, t_prev = self._lidar2global(
                prev_pose_meta['lidar2ego_rotation'],  prev_pose_meta['lidar2ego_translation'],
                prev_pose_meta['ego2global_rotation'], prev_pose_meta['ego2global_translation'],
                device,
            )
            xyz_global    = xyz @ R_prev.double().T + t_prev.double()
            xyz_cur_lidar = (xyz_global - t_cur.double()) @ R_cur.double()

        pts[:, :3] = xyz_cur_lidar.float()
        return pts

    # ------------------------------------------------------------------
    # Gaussian extraction from encoder output
    # ------------------------------------------------------------------

    def _gaussians_from_representation(self, results):
        """Pack encoder output into (N, D) tensors, one per batch sample.
        Timestamp dim (ts_index) is set to 0.0 (= current frame).
        """
        g = results["representation"][-1]["gaussian"]
        imgs = results['imgs']
        device = imgs[0].device if isinstance(imgs, list) else imgs.device

        means  = g.means.to(device)
        opas   = g.opacities.to(device)
        scales = g.scales.to(device)
        rots   = g.rotations.to(device)
        sems   = g.semantics.to(device)

        pseudo = torch.cat([means, opas, scales, rots, sems], dim=-1).detach()
        pseudo[:, :, self.ts_index] = 0.0
        return [pseudo[b] for b in range(pseudo.shape[0])]

    # ------------------------------------------------------------------
    # Rolling Gaussian buffer
    # ------------------------------------------------------------------

    def _buffer_write(self, sample_token, gaussians, pose_meta):
        """Store current-frame Gaussians (CPU fp32) in the rolling buffer.

        Kept at fp32 to match offline `.bin` precision; fp16 was found to lose
        ~0.01-0.05 m of xyz precision and degrade FSDv2 detection.
        """
        self._gaussian_buffer[sample_token] = (gaussians.detach().cpu(), pose_meta)

    def _buffer_read(self, prev_sample_tokens, device):
        """Retrieve prev-frame Gaussians from the buffer, ordered t-1 first.

        Missing entries (first frame of scene or warm-up) are silently skipped.
        """
        results = []
        for token in prev_sample_tokens:
            if token in self._gaussian_buffer:
                g_cpu, pose = self._gaussian_buffer[token]
                results.append((g_cpu.to(device), pose))
        return results

    # ------------------------------------------------------------------
    # Multiframe point cloud assembly
    # ------------------------------------------------------------------

    def _build_multiframe_points(self, results, img_metas):
        """Assemble per-sample multiframe Gaussian point clouds.

        1. Extract current Gaussians from encoder output.
        2. Write them to the rolling buffer.
        3. Read prev Gaussians from buffer (zero extra encoder calls).
        4. Apply ego-motion compensation and concatenate.
        5. PCR filter.
        """
        imgs = results['imgs']
        device = imgs[0].device if isinstance(imgs, list) else imgs.device
        cur_list = self._gaussians_from_representation(results)

        multiframe = []
        for b, cur_pts in enumerate(cur_list):
            meta = img_metas[b]
            sample_token       = meta['sample_token']
            prev_sample_tokens = meta.get('prev_sample_tokens', [])
            ts_cur             = meta.get('timestamp', 0.0)

            scene_token = meta.get('scene_token', None)
            if scene_token and scene_token != self._last_scene_token:
                self._gaussian_buffer.clear()
                self._last_scene_token = scene_token

            pose_meta = {
                'lidar2ego_rotation':    meta['lidar2ego_rotation'],
                'lidar2ego_translation': meta['lidar2ego_translation'],
                'ego2global_rotation':   meta['ego2global_rotation'],
                'ego2global_translation':meta['ego2global_translation'],
                'timestamp':             ts_cur,
            }

            # Read prev entries WITH ordering preserved so we can index into
            # the per-prev sensor2lidar lists from img_metas.
            prev_s2l_R = meta.get('prev_sensor2lidar_R', [])
            prev_s2l_t = meta.get('prev_sensor2lidar_t', [])
            prev_entries = []
            for k_idx, token in enumerate(prev_sample_tokens):
                if token in self._gaussian_buffer:
                    g_cpu, pose = self._gaussian_buffer[token]
                    s2l_R = prev_s2l_R[k_idx] if k_idx < len(prev_s2l_R) else None
                    s2l_t = prev_s2l_t[k_idx] if k_idx < len(prev_s2l_t) else None
                    prev_entries.append((g_cpu.to(device), pose, s2l_R, s2l_t))
            self._buffer_write(sample_token, cur_pts, pose_meta)

            all_pts = [cur_pts]
            prev_counts = []
            for g_prev, prev_pose, s2l_R, s2l_t in prev_entries:
                g_prev = self._transform_prev_to_cur(
                    g_prev, meta, prev_pose, device,
                    s2l_R=s2l_R, s2l_t=s2l_t,
                )
                g_prev[:, self.ts_index] = float(ts_cur - prev_pose['timestamp'])
                g_prev = g_prev[g_prev[:, 3] > 0.01]
                all_pts.append(g_prev)
                prev_counts.append(g_prev.shape[0])

            # Pad with current-frame copies when no prev data is available
            # (matches LoadPointsFromMultiSamples pad_empty_samples=True behavior)
            if len(prev_entries) == 0:
                for _ in range(self.prev_samples_num):
                    pad = cur_pts.clone()
                    pad[:, self.ts_index] = 0.0
                    all_pts.append(pad)
                    prev_counts.append(pad.shape[0])

            combined = torch.cat(all_pts, dim=0)

            if self.point_cloud_range is not None:
                pcr = self.point_cloud_range
                xyz = combined[:, :3]
                mask = (
                    (xyz[:, 0] > pcr[0]) & (xyz[:, 0] < pcr[3]) &
                    (xyz[:, 1] > pcr[1]) & (xyz[:, 1] < pcr[4]) &
                    (xyz[:, 2] > pcr[2]) & (xyz[:, 2] < pcr[5])
                )
                combined = combined[mask]

            prev_str = '+'.join(str(c) for c in prev_counts) if prev_counts else 'none'
            xyz = combined[:, :3]
            '''

            print(f"[DBG sample={b}] cur={cur_pts.shape[0]} prev={prev_str} "
                  f"after_pcr={combined.shape[0]} "
                  f"xyz_min=({xyz[:,0].min():.1f},{xyz[:,1].min():.1f},{xyz[:,2].min():.1f}) "
                  f"xyz_max=({xyz[:,0].max():.1f},{xyz[:,1].max():.1f},{xyz[:,2].max():.1f})")
            n_show = 3
            # layout: xyz[0:3] | opa[3] | scale[4:7] | rot[7:11] | sem[11:ts_index+1]
            rot_start, sem_start = 7, 11
            for label, pts in [("first", combined[:n_show]), ("last", combined[-n_show:])]:
                for i, row in enumerate(pts):
                    rot  = row[rot_start:sem_start]
                    sems = row[sem_start:self.ts_index]
                    sem_str = ' '.join(f'{v:.2f}' for v in sems.tolist())
                    print(f"  [{label}[{i}]] xyz=({row[0]:.3f},{row[1]:.3f},{row[2]:.3f}) "
                          f"opa={row[3]:.3f} "
                          f"scale=({row[4]:.3f},{row[5]:.3f},{row[6]:.3f}) "
                          f"rot=({rot[0]:.3f},{rot[1]:.3f},{rot[2]:.3f},{rot[3]:.3f}) "
                          f"sem=[{sem_str}] "
                          f"ts={row[self.ts_index]:.3f}")
            '''
            multiframe.append(combined)

        return multiframe

    # ------------------------------------------------------------------
    # Forward train / test
    # ------------------------------------------------------------------

    def forward_train(self,
                      img=None,
                      metas=None,
                      img_metas=None,
                      points=None,
                      extra_backbone=False,
                      rep_only=False,
                      **kwargs):
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=img)

        results = {'imgs': img, 'metas': metas, 'img_metas': img_metas, 'points': points}
        results.update(kwargs)

        outs = self.extract_img_feat(**results)
        results.update(outs)

        outs = self.pts_voxel_encoder(**results)
        results.update(outs)

        outs = self.pts_middle_encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)

        multiframe_points = self._build_multiframe_points(results, img_metas)
        # print(f"[DBG] multiframe_points lengths: {[p.shape[0] for p in multiframe_points]}")

        fsdv2_losses = self.fsdv2.forward_train(
            points=multiframe_points,
            img_metas=img_metas,
            gt_bboxes_3d=results["gt_bboxes_3d"],
            gt_labels_3d=results["gt_labels_3d"],
        )

        if isinstance(fsdv2_losses, dict):
            losses = {}
            for k, v in fsdv2_losses.items():
                if isinstance(v, (list, tuple)):
                    losses[k] = torch.stack(v).mean()
                elif isinstance(v, torch.Tensor):
                    losses[k] = v.mean()
                else:
                    losses[k] = torch.tensor(v, device=img.device)
            return losses
        return fsdv2_losses

    def forward_test(self,
                     img=None,
                     metas=None,
                     points=None,
                     extra_backbone=False,
                     **kwargs):
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=img)

        results = {'imgs': img, 'metas': metas, 'points': points}
        results.update(kwargs)

        img_metas = results.get('img_metas', kwargs.get('img_metas'))
        if img_metas and isinstance(img_metas[0], list):
            img_metas = [m[0] for m in img_metas]

        outs = self.extract_img_feat(**results)
        results.update(outs)

        outs = self.pts_voxel_encoder(**results)
        results.update(outs)

        outs = self.pts_middle_encoder(**results)
        results.update(outs)

        multiframe_points = self._build_multiframe_points(results, img_metas)

        predictions = self.fsdv2.simple_test(
            points=multiframe_points,
            img_metas=img_metas,
        )
        return predictions
