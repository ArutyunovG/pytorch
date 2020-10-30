from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import scipy.ndimage.filters as filters
from scipy.ndimage.interpolation import shift
import unittest


class TestCTDetDecodeOp(serial.SerializedTestCase):

    @given(
        batch_size=st.sampled_from([1, 2, 5]),
        max_dets=st.sampled_from([10, 50, 100]),
        num_classes=st.sampled_from([1, 7, 23]),
        w=st.sampled_from([128, 256, 512]),
        h=st.sampled_from([128, 256, 512]),
        stride=st.sampled_from([1, 2, 4]),
        **hu.gcs_cpu_only
    )
    def test_ctdet_decode(
        self, batch_size, max_dets,
        num_classes, w, h, stride, gc, dc
    ):
        imgs = np.random.rand(batch_size, 3, h, w).astype(np.float32)
        fms_w = w // stride
        fms_h = h // stride

        hm = np.random.rand(
            batch_size, num_classes, fms_h, fms_w).astype(np.float32)
        wh = np.random.rand(batch_size, 2, fms_h, fms_w).astype(np.float32)
        of = np.random.rand(batch_size, 2, fms_h, fms_w).astype(np.float32)

        def ref(imgs, hm, wh, of):
            peaks = np.zeros_like(hm)
            for batch in range(batch_size):
                for channel in range(num_classes):
                    peaks[batch, channel, :, :] = filters.maximum_filter(
                        hm[batch, channel, :, :],
                        size=3,
                        mode='constant',
                        cval=0)
            hm[hm != peaks] = 0
            scores = np.zeros([0], dtype=np.float32)
            classes = np.zeros([0], dtype=np.float32)
            all_bboxes = np.zeros([0, 4], dtype=np.float32)
            for batch in range(batch_size):
                single_hm = hm[batch, :, :, :].ravel()
                indices = (-single_hm).argsort(kind='mergesort')[:max_dets]
                indices = indices[:np.sum(hm[batch] > 0)]
                scores = np.append(scores, single_hm[indices])
                classes = np.append(
                    classes,
                    (indices // (fms_w * fms_h)).astype(np.float32))
                indices = indices % (fms_w * fms_h)
                ys, xs = np.unravel_index(indices, (fms_h, fms_w))
                ws = wh[batch, 0].ravel()[indices] * stride
                hs = wh[batch, 1].ravel()[indices] * stride
                xs = (xs + of[batch, 0].ravel()[indices]) * stride
                ys = (ys + of[batch, 1].ravel()[indices]) * stride

                bboxes = np.array(
                    [xs - ws / 2, ys - hs / 2, xs + ws / 2, ys + hs / 2])
                all_bboxes = np.concatenate((all_bboxes, bboxes.transpose()))

            return scores, all_bboxes, classes + 1.0

        op = core.CreateOperator(
            'CTDetDecode',
            ['IMAGES', 'HEATMAP', 'WH', 'OFFSETS'],
            ['SCORES', 'BBOXES', 'CLASSES'],
            max_detections=max_dets)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[imgs, hm, wh, of],
            reference=ref,
        )

if __name__ == "__main__":
    import random
    random.seed(2603)
    unittest.main()
