import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import os
from pdb import set_trace
import yaml
import torch
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC
from collections import OrderedDict


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        # print("img  ", img)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        # print("results['img']  len",len(results['img']))
        # print("results['img']0  shape",results['img'][0].shape)


        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam_intrinsic'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


'''
@PIPELINES.register_module()
class LoadPointsFromMultiSamples(object):
    """Load points from previous keyframe samples (not all sweeps).

    Only loads keyframe LiDAR scans (usually spaced 0.5s apart in nuScenes).

    Args:
        samples_num (int): Number of keyframe samples. Defaults to 3.
        load_dim (int): Dimensionality of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimensions to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): File client args for loading point files.
        pad_empty_samples (bool): Whether to pad with current frame if not enough.
        remove_close (bool): Whether to remove points close to ego vehicle.
        test_mode (bool): If True, deterministically picks samples.
        ts_index (int): Index of the timestamp dimension in the point cloud. Defaults to 4.
    """

    def __init__(self,
                 samples_num=3,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_samples=False,
                 remove_close=False,
                 test_mode=False,
                 ts_index=4):
        self.load_dim = load_dim
        self.samples_num = samples_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_samples = pad_empty_samples
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.ts_index = ts_index

    def _load_points(self, pts_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        points = results['points']
        # print("points shape at the beginning", points.shape)
        points.tensor[:, self.ts_index] = 0  # set time = 0 for current frame
        sample_points_list = [points]
        ts = results['timestamp']

        # Filter only keyframe sweeps (i.e., previous samples)
        # keyframe_sweeps = [s for s in results['sweeps'] if s.get('is_key_frame', False)]
        keyframe_sweeps = [s for s in results['sweeps'] if 'samples' in s['data_path']]


        if len(keyframe_sweeps) == 0 and self.pad_empty_samples:
            # print("No keyframe sweeps found, padding with current frame.")
            for _ in range(self.samples_num):
                sample_points_list.append(points if not self.remove_close else self._remove_close(points))
        else:
            # print(f"Found {len(keyframe_sweeps)} keyframe sweeps.")
            if len(keyframe_sweeps) <= self.samples_num:
                # print("Not enough keyframe sweeps, using all available.")
                choices = np.arange(len(keyframe_sweeps))
            elif self.test_mode:
                # print("Test mode: using first samples_num keyframe sweeps.")
                choices = np.arange(self.samples_num)
            else:
                # print("Randomly selecting keyframe sweeps.")
                choices = np.random.choice(len(keyframe_sweeps), self.samples_num, replace=False)

            for idx in choices:
                # print(f"Loading keyframe sweep {idx + 1}/{len(keyframe_sweeps)}")
                sample = keyframe_sweeps[idx]
                # print("sample['data_path'] ", sample['data_path'])
                points_sample = self._load_points(sample['data_path'])
                points_sample = np.copy(points_sample).reshape(-1, self.load_dim)

                if self.remove_close:
                    points_sample = self._remove_close(points_sample)

                sample_ts = sample['timestamp'] / 1e6
                points_sample[:, :3] = points_sample[:, :3] @ sample['sensor2lidar_rotation'].T
                points_sample[:, :3] += sample['sensor2lidar_translation']
                points_sample[:, 4] = ts - sample_ts  # time gap from current frame

                points_sample = points.new_point(points_sample)
                sample_points_list.append(points_sample)

        points = points.cat(sample_points_list)
        points = points[:, self.use_dim]
        print("points shape", points.shape)# at the end of LoadMultiSamples

        results['points'] = points
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(samples_num={self.samples_num})'
'''

@PIPELINES.register_module()
class LoadPointsFromMultiSamples(object):
    """Load points from previous keyframe samples (not all sweeps).

    Only loads keyframe LiDAR scans (usually spaced 0.5s apart in nuScenes).

    Args:
        samples_num (int): Number of keyframe samples. Defaults to 3.
        load_dim (int): Dimensionality of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimensions to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): File client args for loading point files.
        pad_empty_samples (bool): Whether to pad with current frame if not enough.
        remove_close (bool): Whether to remove points close to ego vehicle.
        test_mode (bool): If True, deterministically picks samples.
        ts_index (int): Index of the timestamp dimension in the point cloud. Defaults to 4.
    """

    def __init__(self,
                 samples_num=3,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_samples=False,
                 remove_close=False,
                 test_mode=False,
                 ts_index=4):
        self.load_dim = load_dim
        self.samples_num = samples_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_samples = pad_empty_samples
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.ts_index = ts_index

    def _load_points(self, pts_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    # def _quat_to_yaw_sincos_torch(self, T):
    #     # T: torch.Tensor (N, 28)
    #     q = T[:, 7:11].to(torch.float64)          # (N,4)
    #     # normalize
    #     n = torch.linalg.norm(q, dim=1)
    #     n = torch.where(n > 0, n, torch.ones_like(n))
    #     q = q / n.unsqueeze(1)
    #     w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    #     # yaw about +Z (nuScenes: x-forward, y-left, z-up)
    #     yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    #     # print("yaw", yaw)
    #     T[:, 7] = torch.sin(yaw).to(T.dtype)
    #     T[:, 8] = torch.cos(yaw).to(T.dtype)
    #     # print("T[:, 7]", T[:, 7])
    #     # print("T[:, 8]", T[:, 8])
    #     T[:, 9:11] = 0

    def __call__(self, results):
        points = results['points']
        # self._quat_to_yaw_sincos_torch(points.tensor)#convert to yaw
        # print("points shape at the beginning", points.shape)
        points.tensor[:, self.ts_index] = 0  # set time = 0 for current frame
        sample_points_list = [points]
        ts = results['timestamp']

        # Filter only keyframe sweeps (i.e., previous samples)
        # keyframe_sweeps = [s for s in results['sweeps'] if s.get('is_key_frame', False)]
        keyframe_sweeps = [s for s in results['sweeps'] if 'samples' in s['data_path']]

        #  # Print all available keyframe paths
        # print(f"Available keyframe sweeps ({len(keyframe_sweeps)}):")
        # for i, sweep in enumerate(keyframe_sweeps):
        #     print(f"  {i}: {sweep['data_path']}")


        if len(keyframe_sweeps) == 0 and self.pad_empty_samples:
            # print("No keyframe sweeps found, padding with current frame.")
            for _ in range(self.samples_num):
                sample_points_list.append(points if not self.remove_close else self._remove_close(points))
        else:
            # print(f"Found {len(keyframe_sweeps)} keyframe sweeps.")
            if len(keyframe_sweeps) <= self.samples_num:
                # print("Not enough keyframe sweeps, using all available.")
                choices = np.arange(len(keyframe_sweeps))
            elif self.test_mode:
                # print("Test mode: using first samples_num keyframe sweeps.")
                choices = np.arange(self.samples_num)
            else:
                # print("Randomly selecting keyframe sweeps.")
                # choices = np.random.choice(len(keyframe_sweeps), self.samples_num, replace=False)
                # Select the first samples_num (most recent)
                choices = np.arange(self.samples_num)
                # print(f"Selecting the {self.samples_num} most recent keyframe sweeps.")
        

            for idx in choices:
                # print(f"Loading keyframe sweep {idx + 1}/{len(keyframe_sweeps)}")
                sample = keyframe_sweeps[idx]
                # print("sample['data_path'] ", sample['data_path'])
                points_sample = self._load_points(sample['data_path'])
                points_sample = np.copy(points_sample).reshape(-1, self.load_dim)

                if self.remove_close:
                    points_sample = self._remove_close(points_sample)

                sample_ts = sample['timestamp'] / 1e6
                points_sample[:, :3] = points_sample[:, :3] @ sample['sensor2lidar_rotation'].T
                points_sample[:, :3] += sample['sensor2lidar_translation']
                points_sample[:, self.ts_index] = ts - sample_ts  # time gap from current frame, was hard coded as 4

                # Filter points based on ts_index (4th dimension value > 0.01)
                points_sample = points_sample[points_sample[:, 3] > 0.01]

                points_sample = points.new_point(points_sample)
                sample_points_list.append(points_sample)

        points = points.cat(sample_points_list)
        points = points[:, self.use_dim]
        print("points MultiFrame shape", points.shape)# at the end of LoadMultiSamples
        t = points.tensor
        n_show, rot_start, sem_start = 3, 7, 11
        for label, pts in [("first", t[:n_show]), ("last", t[-n_show:])]:
            for i, row in enumerate(pts):
                rot  = row[rot_start:sem_start]
                sems = row[sem_start:self.ts_index]
                sem_str = ' '.join(f'{v:.2f}' for v in sems.tolist())
                print(f"  [loading {label}[{i}]] xyz=({row[0]:.3f},{row[1]:.3f},{row[2]:.3f}) "
                      f"opa={row[3]:.3f} "
                      f"scale=({row[4]:.3f},{row[5]:.3f},{row[6]:.3f}) "
                      f"rot=({rot[0]:.3f},{rot[1]:.3f},{rot[2]:.3f},{rot[3]:.3f}) "
                      f"sem=[{sem_str}] "
                      f"ts={row[self.ts_index]:.3f}")

        results['points'] = points
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(samples_num={self.samples_num})'


@PIPELINES.register_module()
class LoadPointsFromMultiSamplesWithRotation(LoadPointsFromMultiSamples):
    """Same as LoadPointsFromMultiSamples but also rotates the Gaussian
    orientation quaternion (dims 7:11) when transforming prev frames into
    the current LiDAR frame.

    The Gaussian covariance is Σ = R(q) diag(s)² R(q)ᵀ.  When the frame
    transform rotation R_s2l is applied, the new covariance becomes
        R_s2l Σ R_s2l.T = (R_s2l R(q)) diag(s)² (R_s2l R(q))T,
    so the new quaternion encodes  R_s2l @ R(q),  i.e. q_new = q_s2l ⊗ q_old.

    All other args are identical to LoadPointsFromMultiSamples.
    quat_dims (tuple): start and end (exclusive) of quaternion dims. Default (7, 11).
    """

    def __init__(self, quat_dims=(7, 11), **kwargs):
        super().__init__(**kwargs)
        self.quat_start, self.quat_end = quat_dims

    @staticmethod
    def _mat_to_quat(R):
        """3×3 rotation matrix → quaternion [w, x, y, z] (float64 numpy)."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z], dtype=np.float64)

    @staticmethod
    def _quat_mul_batch(q1, q2):
        """Hamilton product q1 ⊗ q2.
        q1: (4,) scalar quaternion [w,x,y,z]
        q2: (N, 4) batch of quaternions
        Returns: (N, 4) normalised quaternions.
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2 = q2[:, 0]; x2 = q2[:, 1]; y2 = q2[:, 2]; z2 = q2[:, 3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        out = np.stack([w, x, y, z], axis=1)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-8)
        return out

    def __call__(self, results):
        points = results['points']
        points.tensor[:, self.ts_index] = 0
        sample_points_list = [points]
        ts = results['timestamp']

        keyframe_sweeps = [s for s in results['sweeps'] if 'samples' in s['data_path']]

        if len(keyframe_sweeps) == 0 and self.pad_empty_samples:
            for _ in range(self.samples_num):
                sample_points_list.append(
                    points if not self.remove_close else self._remove_close(points))
        else:
            if len(keyframe_sweeps) <= self.samples_num:
                choices = np.arange(len(keyframe_sweeps))
            elif self.test_mode:
                choices = np.arange(self.samples_num)
            else:
                choices = np.arange(self.samples_num)

            for idx in choices:
                sample = keyframe_sweeps[idx]
                points_sample = self._load_points(sample['data_path'])
                points_sample = np.copy(points_sample).reshape(-1, self.load_dim)

                if self.remove_close:
                    points_sample = self._remove_close(points_sample)

                R = sample['sensor2lidar_rotation']   # (3, 3)

                # Transform xyz positions
                points_sample[:, :3] = points_sample[:, :3] @ R.T
                points_sample[:, :3] += sample['sensor2lidar_translation']

                # Rotate Gaussian orientation quaternions: q_new = q_R ⊗ q_old
                q_transform = self._mat_to_quat(R)    # (4,) [w,x,y,z]
                qs = self.quat_start
                qe = self.quat_end
                q_prev = points_sample[:, qs:qe].astype(np.float64)
                points_sample[:, qs:qe] = self._quat_mul_batch(q_transform, q_prev)

                sample_ts = sample['timestamp'] / 1e6
                points_sample[:, self.ts_index] = ts - sample_ts

                # Filter low-opacity Gaussians
                points_sample = points_sample[points_sample[:, 3] > 0.01]

                points_sample = points.new_point(points_sample)
                sample_points_list.append(points_sample)

        points = points.cat(sample_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 tanh_dim=None,
                 ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.tanh_dim = tanh_dim

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        print("points shape", points.shape)# at the end of LoadMultiSamples

        return points


    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        # print("points shape ", points.shape)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.tanh_dim is not None:
            # only used for SST. FSD applies tanh in the segmentation model.
            assert isinstance(self.tanh_dim, list)
            assert max(self.tanh_dim) < points.shape[1]
            assert min(self.tanh_dim) > 2
            points[:, self.tanh_dim] = np.tanh(points[:, self.tanh_dim])

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 with_speed=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype
        self.with_speed = with_speed

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        if self.with_speed:
            assert 'speed_global' in results['ann_info']
            speed_global = results['ann_info']['speed_global']
            pose = results['pose']
            speed_global = np.pad(speed_global, ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
            speed_local = np.dot(speed_global, np.linalg.inv(pose[:3, :3].T))[:, :2]
            box_7dim = results['gt_bboxes_3d'].tensor
            speed_tensor = torch.from_numpy(speed_local).to(box_7dim.dtype)
            box_with_speed = torch.cat([box_7dim, speed_tensor], -1)
            results['gt_bboxes_3d'] = type(results['gt_bboxes_3d'])(box_with_speed, box_dim=9)

        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str

@PIPELINES.register_module()
class LoadPointsFromMultiSweepsWaymo(LoadPointsFromMultiSweeps):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 close_radius=1.0,
                 t_dim=3,
                 return_list=False,
                 test_mode=False):
        super().__init__(
                 sweeps_num=sweeps_num,
                 load_dim=load_dim,
                 use_dim=use_dim,
                 file_client_args=file_client_args,
                 pad_empty_sweeps=pad_empty_sweeps,
                 remove_close=remove_close,
                 test_mode=test_mode)
        self.close_radius = close_radius
        if isinstance(self.use_dim, int):
            self.use_dim = list(range(self.use_dim))
        self.t_dim = t_dim
        self.return_list = return_list

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        r = np.linalg.norm(points_numpy[:, :2], ord=2, axis=1)
        # x_filt = np.abs(points_numpy[:, 0]) < radius
        # y_filt = np.abs(points_numpy[:, 1]) < radius
        # not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        not_close = r > radius
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']

        if self.t_dim == points.tensor.size(-1):
            padding = points.tensor.new_zeros(len(points.tensor))[:, None]
            points.tensor = torch.cat([points.tensor, padding], dim=1)
            points.points_dim += 1
        elif self.t_dim < points.tensor.size(-1):
            points.tensor[:, self.t_dim] = 0
        else:
            raise ValueError

        sweep_points_list = [points]
        # ts = results['timestamp']
        ts = None # timestamp in mmdet is wrong
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points, self.close_radius))
                else:
                    sweep_points_list.append(points)
        else:
            if hasattr(self, 'sweep_choices'):
                choices = self.sweep_choices
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            else:
                choices = np.arange(self.sweeps_num)
            # elif self.test_mode:
            #     choices = np.arange(self.sweeps_num)
            # else:
            #     choices = np.random.choice(
            #         len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                data_path = os.path.join(os.path.dirname(results['pts_filename']), os.path.basename(sweep['velodyne_path']))
                points_sweep = self._load_points(data_path)
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep, self.close_radius)
                # sweep_ts = sweep['timestamp'] / 1e6
                # points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                #     'sensor2lidar_rotation'].T
                # points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # points_sweep[:, 3] = -1 * float(idx+1)
                # points_sweep = points.new_point(points_sweep)
                curr_pose = results['pose']
                past_pose = sweep['pose']

                past2world_rot = past_pose[0:3, 0:3]
                past2world_trans = past_pose[0:3, 3]

                world2curr_pose = np.linalg.inv(curr_pose)
                world2curr_rot = world2curr_pose[0:3, 0:3]
                world2curr_trans = world2curr_pose[0:3, 3]

                past_points = points_sweep[:, :3]

                past_pc_in_world = np.einsum('ij,nj->ni', past2world_rot, past_points) + past2world_trans[None, :]
                past_pc_in_curr = np.einsum('ij,nj->ni', world2curr_rot, past_pc_in_world) + world2curr_trans[None, :]

                points_sweep[:, :3] = past_pc_in_curr
                # points_sweep[:, 3] = -1 * float(idx+1)
                points_sweep = points_sweep[:, self.use_dim]

                if self.t_dim == points_sweep.shape[-1]:
                    padding = np.zeros(len(points_sweep), dtype=points_sweep.dtype)[:, None] - float(idx+1)
                    points_sweep = np.concatenate([points_sweep, padding], axis=1)
                elif self.t_dim < points_sweep.shape[-1]:
                    points_sweep[:, self.t_dim] = -1 * float(idx+1)

                assert points.points_dim == points_sweep.shape[-1]
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        if self.return_list:
            results['points_list'] = sweep_points_list
            return results

        points = points.cat(sweep_points_list)
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@PIPELINES.register_module()
class LoadPreviousSweepsWaymo(LoadPointsFromMultiSweeps):

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 3, 4],
                 file_client_args=dict(backend='disk'),
                 ):
        super().__init__(
                 sweeps_num=sweeps_num,
                 load_dim=load_dim,
                 use_dim=use_dim,
                 file_client_args=file_client_args,
                 pad_empty_sweeps=False,
                 remove_close=False,
                 )
        if isinstance(self.use_dim, int):
            self.use_dim = list(range(self.use_dim))


    def __call__(self, results):

        cur_points = results['points']
        sweep_points_list = [cur_points]
        frame_inds_list = [np.zeros(len(cur_points.tensor), dtype=int),]
        sweeps = results['sweeps']
        real_num_sweeps = min(self.sweeps_num, len(sweeps))
        sweeps = sweeps[:real_num_sweeps]

        # pad self as latest frame. So there are at least 1 previous frame and can make a very clean diff operation.
        if real_num_sweeps < self.sweeps_num:
            sweeps = [
                dict(
                    velodyne_path=results['pts_filename'],
                    pose=results['pose']
                )
            ] + sweeps

        # ts = results['timestamp']
        ts = None # timestamp in mmdet is wrong

        for idx in range(len(sweeps)):
            sweep = sweeps[idx]
            data_path = os.path.join(os.path.dirname(results['pts_filename']), os.path.basename(sweep['velodyne_path']))
            points_sweep = self._load_points(data_path)
            points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
            curr_pose = results['pose']
            past_pose = sweep['pose']

            past2world_rot = past_pose[0:3, 0:3]
            past2world_trans = past_pose[0:3, 3]

            world2curr_pose = np.linalg.inv(curr_pose)
            world2curr_rot = world2curr_pose[0:3, 0:3]
            world2curr_trans = world2curr_pose[0:3, 3]

            past_points = points_sweep[:, :3]

            past_pc_in_world = np.einsum('ij,nj->ni', past2world_rot, past_points) + past2world_trans[None, :]
            past_pc_in_curr = np.einsum('ij,nj->ni', world2curr_rot, past_pc_in_world) + world2curr_trans[None, :]

            points_sweep[:, :3] = past_pc_in_curr
            points_sweep = points_sweep[:, self.use_dim]

            frame_inds_list.append(np.zeros(len(points_sweep), dtype=int) - idx - 1)
            points_sweep = cur_points.new_point(points_sweep)
            # vis_bev_pc('ms_01_before_cat.png', points_sweep.tensor[:, :3], [-80.88, -80.88, -2, 80.88, 80.88, 4])
            sweep_points_list.append(points_sweep)

        results['points'] = cur_points.cat(sweep_points_list)
        results['pts_frame_inds'] = np.concatenate(frame_inds_list, 0)
        results['num_frames'] = len(sweeps) + 1

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@PIPELINES.register_module()
class LoadPointsFromFileResetLast(LoadPointsFromFile):

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2, 3],
                 shift_height=False,
                 use_color=False,
                 append_last=False,
                 file_client_args=dict(backend='disk'),
                 reset_value=0):
        super().__init__(
                 coord_type,
                 load_dim,
                 use_dim,
                 shift_height,
                 use_color,
                 file_client_args,
        )
        self.reset_value = reset_value
        self.append_last = append_last

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        if self.append_last:
            points.tensor = torch.nn.functional.pad(points.tensor, (0, 1), 'constant', float(self.reset_value))
            points.points_dim += 1
        else:
            points.tensor[:, -1] = float(self.reset_value)
        results['points'] = points

        return results

@PIPELINES.register_module()
class NormalizePoints(object):

    def __init__(self,
                 std=[255,],
                 mean=[0,],
                 dims=[3,]):
        self.dims = dims
        self.std = std
        self.mean = mean

    def __call__(self, input_dict):
        """Call function to jitter all the points in the scene.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after adding noise to each point, \
                'points' key is updated in the result dict.
        """
        points = input_dict['points']
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)

        points.tensor[:, self.dims] = (points.tensor[:, self.dims] - mean[None, :]) / std[None, :]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(jitter_std={self.jitter_std},'
        repr_str += f' clip_range={self.clip_range})'
        return repr_str



@PIPELINES.register_module()
class NormalizePC28:
    """Normalize an N×28 point cloud with per-dimension mean/std.

    Args:
        mean (list[float]): length-28 list of means.
        std (list[float]): length-28 list of stds.
        dims (list[int] | None): which columns to normalize. By default all 28.
        skip_xyz (bool): if True, do not normalize columns 0,1,2 (x,y,z).
        eps (float): small value to avoid div-by-zero.
    """

    def __init__(self,
                 mean=None,
                 std=None,
                 dims=None,
                 skip_xyz=False,
                 eps=1e-6):
        # Defaults filled with your folder stats, ordered by columns 0..27
        default_mean = [
             0.0900,  -0.8306,  -0.2891,   0.0682,
             0.1399,   1.9966,   1.5259,  -0.0012,
            -0.7052,  -0.0036,   0.7088, -19.5733,
            -4.8887,  -4.6491,  -6.5724,  -1.6826,
            -5.1729,  -6.4112,  -2.4400,  -4.4674,
            -4.6998,  -3.6998,  -1.2615,  -4.1443,
            -1.4455,  -1.6835,   2.8215,   0.7328
        ]
        default_std = [
            24.1051, 26.5488,  1.8629, 0.0899,
             0.1179,  0.9848,  0.9949, 0.0053,
             0.0084,  0.0046,  0.0080, 2.0222,
             2.0643,  1.8319,  2.9277, 2.6746,
             2.5430,  2.3160,  2.0062, 2.2840,
             2.7244,  2.9749,  3.6298, 1.7960,
             2.3468,  2.2998,  2.8404, 2.9904
        ]
        self.mean = default_mean if mean is None else mean
        self.std = default_std if std is None else std
        self.eps = float(eps)

        if len(self.mean) != 28 or len(self.std) != 28:
            raise ValueError("mean and std must be length-28 lists.")

        if dims is None:
            dims = list(range(28))
        if skip_xyz:
            dims = [i for i in dims if i not in (0, 1, 2)]
        if not dims:
            raise ValueError("No dims selected for normalization.")

        self.dims = dims

    def __call__(self, input_dict):
        points = input_dict['points']  # BasePoints with .tensor of shape (N, C)
        pt = points.tensor
        device = pt.device
        dtype = pt.dtype

        mean_t = torch.as_tensor(self.mean, device=device, dtype=dtype)
        std_t = torch.as_tensor(self.std, device=device, dtype=dtype)
        std_t = torch.clamp(std_t, min=self.eps)  # avoid div by 0

        # Normalize only selected dims
        idx = torch.as_tensor(self.dims, device=device, dtype=torch.long)
        pt[:, idx] = (pt[:, idx] - mean_t[idx][None, :]) / std_t[idx][None, :]

        # write back
        points.tensor = pt
        input_dict['points'] = points
        return input_dict

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"dims={self.dims}, skip_xyz={'0,1,2' not in map(str,self.dims)}, "
                f"eps={self.eps})")
