#include "caffe2/operators/ctdet_decode_op.h"

#include <queue>

namespace caffe2 {

namespace {

template <typename T>
struct ValueComp {
  bool operator()(
      const std::pair<T, int64_t>& lhs,
      const std::pair<T, int64_t>& rhs) const {
    return lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second);
  }
};

template <typename T>
struct MaximaChecker {
  MaximaChecker(T * data, int64_t w, int64_t h): data_(data), w_(w), h_(h) {
  }

  bool operator() (int64_t index) const {
    T val = data_[index];
    int64_t single_channel_index = index % (w_ * h_);
    int64_t x = single_channel_index % w_;
    int64_t y = single_channel_index / w_;

    if (y == 0 || y == h_ - 1)
    {
      int64_t stride_y = y == 0 ? w_ : -w_;
      if (x == 0 || x == w_ - 1)
      {
        int64_t stride_x = x == 0 ? 1 : -1;
        return val >= data_[index + stride_x]
          && val >= data_[index + stride_y]
          && val >= data_[index + stride_y + stride_x];
      }

      return val >= data_[index - 1]
        && val >= data_[index + 1]
        && val >= data_[index + stride_y - 1]
        && val >= data_[index + stride_y]
        && val >= data_[index + stride_y + 1];
    }
    if (x == 0 || x == w_ - 1)
    {
      int64_t stride_x = x == 0 ? 1 : -1;

      return val >= data_[index - w_ + stride_x]
        && val >= data_[index - w_]
        && val >= data_[index + stride_x]
        && val >= data_[index + w_ + stride_x]
        && val >= data_[index + w_];
    }

    return val >= data_[index - w_ - 1]
      && val >= data_[index - w_]
      && val >= data_[index - w_ + 1]
      && val >= data_[index - 1]
      && val >= data_[index + 1]
      && val >= data_[index + w_ - 1]
      && val >= data_[index + w_]
      && val >= data_[index + w_ + 1];
  }

private:
  T * data_;
  int64_t w_;
  int64_t h_;
};

} // namespace

template <>
bool CTDetDecodeOp<CPUContext>::RunOnDevice() {
  auto& images = Input(IMAGES);
  auto& heatmap = Input(HEATMAP);
  auto& wh = Input(WH);
  auto& offsets = Input(OFFSETS);

  CAFFE_ENFORCE(heatmap.template IsType<float>(), heatmap.dtype().name());
  CAFFE_ENFORCE(wh.template IsType<float>(), wh.dtype().name());
  CAFFE_ENFORCE(offsets.template IsType<float>(), offsets.dtype().name());

  CAFFE_ENFORCE_EQ(images.dim(), 4);
  CAFFE_ENFORCE_EQ(heatmap.dim(), 4);
  CAFFE_ENFORCE_EQ(wh.dim(), 4);
  CAFFE_ENFORCE_EQ(offsets.dim(), 4);

  const int32_t image_h = images.dim(2);
  const int32_t image_w = images.dim(3);

  const auto heatmap_dims = heatmap.sizes();
  const int32_t batch_size = heatmap_dims[0];
  const int32_t num_classes = heatmap_dims[1];
  const int32_t h = heatmap_dims[2];
  const int32_t w = heatmap_dims[3];

  const auto wh_dims = wh.sizes();
  CAFFE_ENFORCE_EQ(wh_dims[0], batch_size);
  CAFFE_ENFORCE_EQ(wh_dims[1], 2);
  CAFFE_ENFORCE_EQ(wh_dims[2], h);
  CAFFE_ENFORCE_EQ(wh_dims[3], w);

  const auto offsets_dims = offsets.sizes();
  CAFFE_ENFORCE_EQ(offsets_dims[0], batch_size);
  CAFFE_ENFORCE_EQ(offsets_dims[1], 2);
  CAFFE_ENFORCE_EQ(offsets_dims[2], h);
  CAFFE_ENFORCE_EQ(offsets_dims[3], w);

  const auto scale_x = static_cast<float>(image_w) / w;
  const auto scale_y = static_cast<float>(image_h) / h;

  const auto size = w * h;
  const auto heatmap_size = num_classes * size;

  auto* scores = Output(SCORES, vector<int64_t>{batch_size * max_detections_}, at::dtype<float>());
  auto* bboxes = Output(BBOXES, vector<int64_t>{batch_size * max_detections_, 4}, at::dtype<float>());
  auto* classes = Output(CLASSES, vector<int64_t>{batch_size * max_detections_}, at::dtype<float>());
  Tensor* batch_splits = nullptr;
  if (OutputSize() > 3)
  {
    batch_splits = Output(BATCH_SPLITS, vector<int64_t>{batch_size}, at::dtype<float>());
  }

  float* scores_data = scores->template mutable_data<float>();
  float* classes_data = classes->template mutable_data<float>();
  float* bboxes_data = bboxes->template mutable_data<float>();
  float* batch_splits_data = nullptr;
  if (batch_splits != nullptr)
  {
    batch_splits_data = batch_splits->template mutable_data<float>();
  }

  int64_t offset = 0;
  for (int32_t batch = 0; batch < batch_size; ++batch) {
    const auto hm_ptr = heatmap.data<float>() + batch * heatmap_size;
    const auto wh_ptr = wh.data<float>() + batch * 2 * size;
    const auto offsets_ptr = offsets.data<float>() + batch * 2 * size;

    MaximaChecker<float> checker(hm_ptr, w, h);

    std::vector<std::pair<float, int64_t>> heap_data;
    heap_data.reserve(max_detections_);

    int hm_index = 0;
    for (int64_t i = 0; i < max_detections_ && hm_index < heatmap_size; ++hm_index) {
      if (checker(hm_index)) {
        heap_data.emplace_back(hm_ptr[hm_index], hm_index);
        ++i;
      }
    }

    std::priority_queue<
      std::pair<float, int64_t>,
      std::vector<std::pair<float, int64_t>>,
      ValueComp<float>>
      pq(ValueComp<float>(), std::move(heap_data));
    for (int64_t i = hm_index; i < heatmap_size; ++i) {
      if (checker(i) && pq.top().first < hm_ptr[i]) {
        pq.pop();
        pq.emplace(hm_ptr[i], i);
      }
    }

    const auto num_detections = pq.size();
    int64_t dst_pos = num_detections - 1;
    while (!pq.empty()) {
      const auto& item = pq.top();
      const auto index = item.second % size;
      scores_data[offset + dst_pos] = item.first;
      classes_data[offset + dst_pos] = item.second / size + 1.0f;
      const float offset_x = offsets_ptr[index];
      const float offset_y = offsets_ptr[index + size];
      const float half_bbox_w = wh_ptr[index] / 2.0f;
      const float half_bbox_h = wh_ptr[index + size] / 2.0f;
      const float center_x = index % w + offset_x;
      const float center_y = index / w + offset_y;
      bboxes_data[offset * 4 + dst_pos * 4] = (center_x - half_bbox_w) * scale_x;
      bboxes_data[offset * 4 + dst_pos * 4 + 1] = (center_y - half_bbox_h) * scale_y;
      bboxes_data[offset * 4 + dst_pos * 4 + 2] = (center_x + half_bbox_w) * scale_x;
      bboxes_data[offset * 4 + dst_pos * 4 + 3] = (center_y + half_bbox_h) * scale_y;

      pq.pop();
      --dst_pos;
    }

    if (batch_splits_data != nullptr)
    {
      batch_splits_data[batch] = num_detections;
    }

    offset += num_detections;
  }

  scores->Resize(offset);
  bboxes->Resize(offset, 4);
  classes->Resize(offset);

  return true;
}

REGISTER_CPU_OPERATOR(CTDetDecode, CTDetDecodeOp<CPUContext>);
OPERATOR_SCHEMA(CTDetDecode)
    .NumInputs(4)
    .NumOutputs(3, 4)
    .Arg(
        "max_detections",
        "Max detections.")
    .SetDoc("Decoder for CenterNet.")
    .Input(
        0,
        "images",
        "float Tensor sized (batch_size, num_classes, input_height, input_width)")
    .Input(
        1,
        "heatmap",
        "float Tensor sized (batch_size, num_classes, height, width)")
    .Input(
        2,
        "wh",
        "float Tensor sized (batch_size, 2, height, width) containing widths and heights of detected bboxes")
    .Input(
        3,
        "offsets",
        "float Tensor sized [batch_size, 2, height, width] containing centers offsets")
    .Output(0, "scores", "Filtered scores, size (num_detections)")
    .Output(
        1,
        "boxes",
        "Filtered boxes, size (num_detections, 4)")
    .Output(2, "classes", "Class id for each filtered score/box, size (num_detections)")
    .Output(
        3,
        "batch_splits",
        "Tensor of shape (batch_size) with each element denoting the number "
        "of detections belonging to the corresponding image in batch");
SHOULD_NOT_DO_GRADIENT(CTDetDecode);

} // namespace caffe2
