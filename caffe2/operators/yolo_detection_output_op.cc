#include "caffe2/operators/yolo_detection_output_op.h"

#include <cmath>
#include <queue>

namespace caffe2 {

namespace {

  struct Bbox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;
    int label;
  };

  float iou(const Bbox & a, const Bbox & b) {
    const float overlap_w = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    const float overlap_h = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
    if (overlap_h <= 0 || overlap_w <= 0) {
        return 0.0f;
    }
    const float overlap_area = overlap_h * overlap_w;
    const float a_area = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    const float b_area = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    const float union_area = a_area + b_area - overlap_area;

    return overlap_area / (union_area + 1e-6);
  }

  float logistic_activate(float f) {
    return 1.0f / (1.0f + expf(-f));
  }
}

template <>
bool YoloDetectionOutputOp<CPUContext>::RunOnDevice() {
  auto& image_shape = Input(IMAGE_SHAPE);
  CAFFE_ENFORCE(image_shape.template IsType<int64_t>(), image_shape.dtype().name());
  int64_t* image_shape_data = image_shape.template mutable_data<int64_t>();
  const float image_h = image_shape.dim32(0) == 4 ? image_shape_data[2] : image_shape_data[0];
  const float image_w = image_shape.dim32(0) == 4 ? image_shape_data[3] : image_shape_data[1];

  auto batch_size = Input(1).dim32(0);
  CAFFE_ENFORCE_EQ(batch_size, 1, "Batch size larger than 1 is unsupported.");
  for (int i = 2; i < InputSize(); ++i) {
    CAFFE_ENFORCE_EQ(Input(i).dim32(0), batch_size, "Different batch sizes in feature maps.");
  }

  auto* scores = Output(SCORES, vector<int64_t>{batch_size * top_k_}, at::dtype<float>());
  auto* bboxes = Output(BBOXES, vector<int64_t>{batch_size * top_k_, 4}, at::dtype<float>());
  auto* classes = Output(CLASSES, vector<int64_t>{batch_size * top_k_}, at::dtype<float>());
  Tensor* batch_splits = nullptr;
  if (OutputSize() > 3) {
    batch_splits = Output(BATCH_SPLITS, vector<int64_t>{batch_size}, at::dtype<float>());
  }

  float* scores_data = scores->template mutable_data<float>();
  float* classes_data = classes->template mutable_data<float>();
  float* bboxes_data = bboxes->template mutable_data<float>();
  float* batch_splits_data = nullptr;
  if (batch_splits != nullptr) {
    batch_splits_data = batch_splits->template mutable_data<float>();
  }

  int current_anchor = 0;
  std::vector<Bbox> decoded_bboxes;
  for (int i = 1; i < InputSize(); ++i) {
    auto& fm = Input(i);
    CAFFE_ENFORCE(fm.template IsType<float>(), fm.dtype().name());
    const auto fm_dims = fm.sizes();
    const int32_t num_fm = fm_dims[1];
    const int32_t fm_h = fm_dims[2];
    const int32_t fm_w = fm_dims[3];
    const int32_t fm_size = fm_h * fm_w;
    const int32_t fm_anchors = num_fm / (num_classes_ + 1 + 4);

    auto fm_ptr = fm.data<float>();
    for (int anchor = 0; anchor < fm_anchors; ++anchor) {
      const float anchor_x = anchors_[2 * (current_anchor + anchor)];
      const float anchor_y = anchors_[2 * (current_anchor + anchor) + 1];
      for (int pos = 0; pos < fm_size; ++pos) {
        const float objectness = logistic_activate(fm_ptr[4 * fm_size + pos]);
        int label;
        float confidence = std::numeric_limits<float>::min();
        for (int class_ = 1; class_ <= num_classes_; ++class_) {
          const float score = logistic_activate(fm_ptr[(4 + class_) * fm_size + pos]);
          if (score > confidence) {
            confidence = score;
            label = class_;
          }
        }
        confidence *= objectness;
        if (confidence < confidence_threshold_) {
          continue;
        }

        const float x = pos % fm_w;
        const float y = pos / fm_w;
        const float x_center = (x + logistic_activate(fm_ptr[pos])) / fm_w * image_w;
        const float y_center = (y + logistic_activate(fm_ptr[fm_size + pos])) / fm_h * image_h;
        const float w = (expf(fm_ptr[2 * fm_size + pos]) * anchor_x);
        const float h = (expf(fm_ptr[3 * fm_size + pos]) * anchor_y);
        const float xmin = x_center - w / 2;
        const float xmax = x_center + w / 2;
        const float ymin = y_center - h / 2;
        const float ymax = y_center + h / 2;
        decoded_bboxes.push_back({ xmin, ymin, xmax, ymax, confidence, label });
      }

      fm_ptr += fm_size * (num_classes_ + 1 + 4);
    }

    current_anchor += fm_anchors;
  }

  std::sort(
    decoded_bboxes.begin(),
    decoded_bboxes.end(),
    [] (const auto & l, const auto & r) {return l.confidence > r.confidence;});
  std::vector<bool> keep(decoded_bboxes.size(), true);
  for (int i = 0; i < decoded_bboxes.size(); ++i) {
      if (!keep[i]) {
          continue;
      }
      Bbox & bbox = decoded_bboxes[i];
      for (int j = i + 1; j < decoded_bboxes.size(); ++j) {
          if (keep[j] && iou(bbox, decoded_bboxes[j]) > nms_threshold_) {
              keep[j] = false;
          }
      }
  }

  int num_output_bboxes = 0;
  int i = 0;
  while(i < decoded_bboxes.size() && num_output_bboxes < top_k_) {
      if (!keep[i]) {
          ++i;
          continue;
      }
      scores_data[num_output_bboxes] = decoded_bboxes[i].confidence;
      classes_data[num_output_bboxes] = decoded_bboxes[i].label;
      bboxes_data[num_output_bboxes * 4] = decoded_bboxes[i].xmin;
      bboxes_data[num_output_bboxes * 4 + 1] = decoded_bboxes[i].ymin;
      bboxes_data[num_output_bboxes * 4 + 2] = decoded_bboxes[i].xmax;
      bboxes_data[num_output_bboxes * 4 + 3] = decoded_bboxes[i].ymax;
      ++num_output_bboxes;
      ++i;
  }
  if (batch_splits_data != nullptr) {
    batch_splits_data[0] = num_output_bboxes;
  }
  scores->Resize(num_output_bboxes);
  bboxes->Resize(num_output_bboxes, 4);
  classes->Resize(num_output_bboxes);

  return true;
}

REGISTER_CPU_OPERATOR(YoloDetectionOutput, YoloDetectionOutputOp<CPUContext>);
OPERATOR_SCHEMA(YoloDetectionOutput)
    .NumInputs(2, INT_MAX)
    .NumOutputs(4)
    .Arg(
        "top_k",
        "Max detections.")
    .Arg(
        "num_classes",
        "Number of detection classes.")
    .Arg(
        "anchors",
        "Bbox anchors.")
    .Arg(
        "confidence_threshold",
        "Confidence threshold.")
    .Arg(
        "nms_threshold",
        "NMS threshold.")
    .SetDoc("This op applies anchor transform for all feature maps and applies non-maxima supression for boxes.")
    .Input(
        0,
        "im_info",
        "int Tensor with input image size")
    .Input(
        1,
        "FM1, FM2, ...",
        "*(type: Tensor`<float>`)* List of input feature maps.")
    .Output(0, "scores", "Filtered scores, size (n)")
    .Output(
        1,
        "boxes",
        "Filtered boxes, size (n, 4). ")
    .Output(2, "classes", "Class id for each filtered score/box, size (n)")
    .Output(
        3,
        "batch_splits",
        "Output batch splits for scores/boxes after applying NMS")
    .Output(4, "keeps", "Optional filtered indices, size (n)");

SHOULD_NOT_DO_GRADIENT(YoloDetectionOutput);

} // namespace caffe2
