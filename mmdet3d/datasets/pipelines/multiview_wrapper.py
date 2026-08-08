from mmcv.utils import Registry
from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class MultiViewWrapper:
    def __init__(self, transforms):
        self.transforms = [PIPELINES.build(t) for t in transforms]

    def __call__(self, results):
        assert isinstance(results['img'], list), "Expected list of multi-view images."
        imgs = []
        for img in results['img']:
            tmp = dict(img=img.copy())
            for t in self.transforms:
                tmp = t(tmp)
            imgs.append(tmp['img'])
        results['img'] = imgs
        return results
