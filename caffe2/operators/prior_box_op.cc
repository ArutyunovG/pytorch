#include "caffe2/operators/prior_box_op.h"

namespace caffe2 {

template<>
bool PriorBoxOp<CPUContext>::RunOnDevice() {
    return DispatchHelper<TensorTypes<int, int64_t, float>>::call(this, Input(0));
}

template <>
template <typename T>
bool PriorBoxOp<CPUContext>::DoRunWithType() {
    const auto& fm = Input(FM);
    const auto& image = Input(IMAGE);

    int layer_width = 0;
    int layer_height = 0;
    int image_width = 0;
    int image_height = 0;

    if (fm.dim() == 1) {
        CAFFE_ENFORCE_EQ(fm.dim32(0), 2);
        const T* fm_shape_data = fm.template data<T>();
        layer_height = fm_shape_data[0];
        layer_width = fm_shape_data[1];
    }
    else {
        CAFFE_ENFORCE_EQ(fm.dim(), 4);
        layer_width = fm.dim32(3);
        layer_height = fm.dim32(2);
    }

    if (image.dim() == 1) {
        CAFFE_ENFORCE_EQ(image.dim32(0), 2);
        const T* image_shape_data = image.template data<T>();
        image_height = image_shape_data[0];
        image_width = image_shape_data[1];
    }
    else {
        CAFFE_ENFORCE_EQ(image.dim(), 4);
        image_width = image.dim32(3);
        image_height = image.dim32(2);
    }

    CAFFE_ENFORCE_GT(layer_width, 0);
    CAFFE_ENFORCE_GT(layer_height, 0);
    CAFFE_ENFORCE_GT(image_width, 0);
    CAFFE_ENFORCE_GT(image_height, 0);

    float step_w = .0f, step_h = .0f;

    if (std::fabs(step_w_) < std::numeric_limits<float>::epsilon() ||
        std::fabs(step_h_) < std::numeric_limits<float>::epsilon()) {
        step_w = static_cast<float>(image_width) / layer_width;
        step_h = static_cast<float>(image_height) / layer_height;
    } else {
        step_w = step_w_;
        step_h = step_h_;
    }

    auto* priors = Output(0, {layer_width * layer_height * num_priors_ * 4}, at::dtype<float>());
    auto* prior_data = priors->template mutable_data<float>();
    int idx = 0;

    for (auto h = 0; h < layer_height; ++h) {
        for (auto w = 0; w < layer_width; ++w) {

            // compute current anchors' 
            // center position in the scale of input image
            const float center_x = (w + offset_) * step_w;
            const float center_y = (h + offset_) * step_h;

            for (auto prior_idx = 0; prior_idx < num_priors_; ++prior_idx) {

                // current anchor sizes in pixels
                const float box_width = widths_[prior_idx];
                const float box_height = heights_[prior_idx];

                // output normalized anchor [xmin, ymin, xmax, ymax]

                // xmin
                prior_data[idx++] = (center_x - box_width / 2.0f) / image_width;
                // ymin
                prior_data[idx++] = (center_y - box_height / 2.0f) / image_height;
                // xmax
                prior_data[idx++] = (center_x + box_width / 2.0f) / image_width;
                // ymax
                prior_data[idx++] = (center_y + box_height / 2.0f) / image_height;

            }

        }
    }

    // clip coordidates of the anchors to be within [0, 1]
    if (clip_) {
        for (auto d = 0; d < priors->numel(); ++d) {
            prior_data[d] = std::min<float>(std::max<float>(prior_data[d], float(0)), float(1));
        }
    }

    return true;
}


REGISTER_CPU_OPERATOR(PriorBox, PriorBoxOp<CPUContext>);

OPERATOR_SCHEMA(PriorBox)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("widths", "List of widths of anchors in pixels.")
    .Arg("heights", "List of heights of anchors in pixels.")
    .Arg("step_w", "Step between adjacent anchors in pixels along width."
        "If this is set, step_h must be set as well.")
    .Arg("step_h", "Step between adjacent anchors in pixels along height."
        "If this is set, step_w must be set as well.")
    .Arg("step", "Step between adjacent anchors in pixels along both width and height.")
    .Arg("clip", "bool (default false) Clip anchors to be within [0, 1]")
    .Arg("offset", "float (default 0.5) Offset to the top left corner of each anchor.")
    .SetDoc(R"DOC(
Takes widths and heights of anchor boxes (prior boxes) and performs their tiling over the input feature map.
Tiling step size is computed either based on explicit parameters (step or step_h and step_w) or via ratio of
input image over feature map spatial sizes. All pixel-scaled parameters are in the scale of the input image.
Inputs can be feature map and input image itself as well as their spatial sizes in 1D tensor.
)DOC")
.Input(0, "FM", "Input feature map or its spatial shape.")
.Input(1, "image", "Input image or its spatial shape.")
.Output(0, "priors", "Output priors tensor.");

SHOULD_NOT_DO_GRADIENT(PriorBox);

} // namespace caffe2

