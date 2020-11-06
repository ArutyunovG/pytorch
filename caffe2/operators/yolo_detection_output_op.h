#ifndef CAFFE2_OPERATORS_YOLO_DETECTION_OUTPUT_OP_H_
#define CAFFE2_OPERATORS_YOLO_DETECTION_OUTPUT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class YoloDetectionOutputOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  YoloDetectionOutputOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    num_classes_ =
        this->template GetSingleArgument<int>("num_classes", 0);
    confidence_threshold_ =
        this->template GetSingleArgument<float>("confidence_threshold", 0.0f);
    nms_threshold_ =
        this->template GetSingleArgument<float>("nms_threshold", 1.0f);
    anchors_ =
        this->template GetRepeatedArgument<float>("anchors");
    top_k_ =
        this->template GetSingleArgument<int>("top_k", std::numeric_limits<int>::max());

    CAFFE_ENFORCE_GT(num_classes_, 0);
  }

  bool RunOnDevice() override;

 protected:
  int num_classes_;
  float confidence_threshold_;
  float nms_threshold_;
  int top_k_;
  std::vector<float> anchors_;
  INPUT_TAGS(IMAGE_SHAPE);
  OUTPUT_TAGS(SCORES, BBOXES, CLASSES, BATCH_SPLITS);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_YOLO_DETECTION_OUTPUT_OP_H_
