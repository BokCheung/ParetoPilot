#include "pp_add_tiling.h"
#include "paretopilot_params.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>
#include <cstdint>

namespace optiling {
static ge::graphStatus PpAddTiling(gert::TilingContext *ctx) {
    const auto &a = ctx->GetInputShape(0)->GetOriginShape();
    const auto &b = ctx->GetInputShape(1)->GetOriginShape();
    if (a.GetDimNum() != b.GetDimNum()) return ge::GRAPH_FAILED;
    for (size_t i=0; i<a.GetDimNum(); ++i) if (a.GetDim(i)!=b.GetDim(i)) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (PP_BLOCK_DIM > platform.GetCoreNumAiv() || 3ULL*PP_BUFFERS*PP_TILE_ELEMS*2 > ub) return ge::GRAPH_FAILED;
    const int64_t length = a.GetShapeSize();
    if (length <= 0 || length > UINT32_MAX) return ge::GRAPH_FAILED;
    PpAddTilingData data;
    data.set_length(static_cast<uint32_t>(length));
    ctx->SetBlockDim(PP_BLOCK_DIM);
    ctx->SetTilingKey(1);
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    ctx->GetWorkspaceSizes(1)[0] = 0;
    std::fprintf(stderr, "PARETOPILOT_PARAMS=%s\n", PP_PARAMS_HASH);
    return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class PpAdd : public OpDef {
public:
    explicit PpAdd(const char *name) : OpDef(name) {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape([](gert::InferShapeContext *ctx) { *ctx->GetOutputShape(0) = *ctx->GetInputShape(0); return ge::GRAPH_SUCCESS; });
        this->SetInferDataType([](gert::InferDataTypeContext *ctx) { ctx->SetOutputDataType(0, ctx->GetInputDataType(0)); return ge::GRAPH_SUCCESS; });
        this->AICore().SetTiling(optiling::PpAddTiling).AddConfig("ascend910b");
    }
};
OP_ADD(PpAdd);
}
