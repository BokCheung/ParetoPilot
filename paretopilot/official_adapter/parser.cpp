// ParetoPilot integration; the MatMul kernel and shape/tiling code are upstream.
#include "register/register.h"
namespace domi {
Status ParseParetoPilotMatmul(const ge::Operator &, ge::Operator &dst) {
    dst.SetAttr("transpose_x1", false);
    dst.SetAttr("transpose_x2", false);
    dst.SetAttr("offset_x", static_cast<int64_t>(0));
    dst.SetAttr("opImplMode", static_cast<int64_t>(1));
    return SUCCESS;
}
REGISTER_CUSTOM_OP("MatMulV3")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::13::ParetoPilotOfficialMatMul")
    .ParseParamsByOperatorFn(ParseParetoPilotMatmul)
    .ImplyType(ImplyType::TVM);
}
