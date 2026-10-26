// Graph integration only; Host shape/tiling and Device kernels are upstream.
#include "register/register.h"
namespace domi {
Status ParseParetoPilotBatchMatmul(const ge::Operator &, ge::Operator &dst) {
    dst.SetAttr("adj_x1", false);
    dst.SetAttr("adj_x2", false);
    dst.SetAttr("offset_x", static_cast<int64_t>(0));
    dst.SetAttr("enable_hf32", false);
    return SUCCESS;
}
REGISTER_CUSTOM_OP("BatchMatMulV3")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::13::ParetoPilotOfficialBatchMatMul")
    .ParseParamsByOperatorFn(ParseParetoPilotBatchMatmul)
    .ImplyType(ImplyType::TVM);
}
