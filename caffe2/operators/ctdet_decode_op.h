#ifndef CAFFE2_OPERATORS_CTDET_DECODE_OP_H_
#define CAFFE2_OPERATORS_CTDET_DECODE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class CTDetDecodeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CTDetDecodeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    max_detections_ =
        this->template GetSingleArgument<int>("max_detections", 100);
  }

  bool RunOnDevice() override;

 protected:
  int max_detections_;
  INPUT_TAGS(IMAGES, HEATMAP, WH, OFFSETS);
  OUTPUT_TAGS(SCORES, BBOXES, CLASSES, BATCH_SPLITS);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CTDET_DECODE_OP_H_
