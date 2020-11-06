from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import scipy.ndimage.filters as filters
from scipy.ndimage.interpolation import shift
import unittest


def iou(box_1, box_2):
    overlap_w = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])
    overlap_h = min(box_1[3], box_2[3]) - max(box_1[1], box_2[1])
    if overlap_w <= 0 or overlap_h <= 0:
        overlap_area = 0
    else:
        overlap_area = overlap_w * overlap_h
    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])
    union_area = box_1_area + box_2_area - overlap_area
    if union_area == 0:
        return 0
    return overlap_area / (union_area + 1e-6)


class TestYoloDetectionOutputOp(serial.SerializedTestCase):

    @given(
        top_k=st.sampled_from([10, 50, 100]),
        num_classes=st.sampled_from([1, 7, 23]),
        h=st.sampled_from([16, 24, 32]),
        w=st.sampled_from([16, 24, 32]),
        im_scale_factor=st.sampled_from([1, 4, 32]),
        num_fms=st.sampled_from([1, 2, 3]),
        anchors_per_fm=st.sampled_from([1, 2, 4]),
        confidence_threshold=st.sampled_from([0.5]),
        nms_threshold=st.sampled_from([0.5]),
        **hu.gcs_cpu_only,
    )
    @settings(deadline=None)
    def test_yolo_detection_output(
        self, top_k, num_classes,
        w, h, im_scale_factor, num_fms, anchors_per_fm,
        confidence_threshold, nms_threshold, gc, dc
    ):
        fms = []
        anchors = []
        num_features = (1 + num_classes + 4) * anchors_per_fm
        image_shape = np.array([1, 3, h * im_scale_factor, w * im_scale_factor], dtype=np.int64)
        for i in range(num_fms):
            anchors.append(np.random.randint(1, 100, (anchors_per_fm * 2)))
            fms.append(np.random.uniform(-4, 4, (1, num_features, h // 2 ** i, w // 2 ** i)).astype(np.float32))

        fm_names = ['FM{}'.format(i) for i in range(num_fms)]
        op = core.CreateOperator(
            'YoloDetectionOutput',
            ['IMAGE_SHAPE'] + fm_names,
            ['SCORES', 'BBOXES', 'CLASSES', 'BATCH_SPLITS'],
            top_k=top_k,
            num_classes=num_classes,
            anchors=[float(i) for fm_anchors in anchors for i in fm_anchors],
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        def ref(image_shape, *fms):
            current_anchor = 0
            bboxes = []
            anchor_shift = 1 + num_classes + 4
            for fm, fm_anchors in zip(fms, anchors):
                num_fm_anchors = fm.shape[1] // anchor_shift
                for i in range(num_fm_anchors):
                    fm_begin = i * anchor_shift
                    fm_end = fm_begin + anchor_shift
                    scores_begin = fm_begin + 4
                    fm[:, fm_begin:fm_begin + 2, :, :] = 1.0 / (1.0 + np.exp(-fm[:, fm_begin:fm_begin + 2, :, :]))
                    fm[:, scores_begin:fm_end, :, :] = 1.0 / (1.0 + np.exp(-fm[:, scores_begin:fm_end, :, :]))
                    fm[:, fm_begin + 2:scores_begin, :, :] = np.exp(fm[:, fm_begin + 2:scores_begin, :, :])
                    for r, c in np.ndindex(fm.shape[2], fm.shape[3]):
                        objectness = fm[0, scores_begin, r, c]
                        class_probs = fm[0, scores_begin + 1:fm_end, r, c]
                        scores = objectness * class_probs
                        label = np.argmax(objectness * class_probs)
                        score = scores[label]
                        if score < confidence_threshold:
                            continue
                        x_center = (c + fm[0, fm_begin, r, c]) / fm.shape[3] * image_shape[3]
                        y_center = (r + fm[0, fm_begin + 1, r, c]) / fm.shape[2] * image_shape[2]
                        w = fm[0, fm_begin + 2, r, c] * fm_anchors[i * 2]
                        h = fm[0, fm_begin + 3, r, c] * fm_anchors[i * 2 + 1]
                        x_min = x_center - w / 2
                        x_max = x_center + w / 2
                        y_min = y_center - h / 2
                        y_max = y_center + h / 2
                        bboxes.append([x_min, y_min, x_max, y_max, score, label + 1])

                current_anchor += num_fm_anchors
            bboxes.sort(key=lambda x: -x[4])

            for i in range(len(bboxes)):
                if bboxes[i][4] == 0:
                    continue
                for j in range(i + 1, len(bboxes)):
                    if iou(bboxes[i], bboxes[j]) > nms_threshold:
                        bboxes[j][4] = 0
            bboxes = np.array([b for b in bboxes if b[4] > confidence_threshold][:top_k])
            
            return [bboxes[:, 4], bboxes[:, :4], bboxes[:, 5], [bboxes.shape[0]]]


        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[image_shape] + fms,
            reference=ref,
        )


if __name__ == "__main__":
    import random
    random.seed(2603)
    unittest.main()
