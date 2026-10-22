#include "pp_matmul_tiling.h"
#include "paretopilot_params.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>
#include <climits>

namespace optiling {
static ge::graphStatus PpMatmulTiling(gert::TilingContext *ctx) {
    const auto &a=ctx->GetInputShape(0)->GetOriginShape();
    const auto &b=ctx->GetInputShape(1)->GetOriginShape();
    if (a.GetDimNum()!=2 || b.GetDimNum()!=2 || a.GetDim(1)!=b.GetDim(0)) return ge::GRAPH_FAILED;
    const int64_t m=a.GetDim(0), n=b.GetDim(1), k=a.GetDim(1);
    if (m<=0 || n<=0 || k<=0 || m>INT_MAX || n>INT_MAX || k>INT_MAX || m%16 || n%16 || k%16) return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    if (PP_BLOCK_DIM>platform.GetCoreNumAic()) return ge::GRAPH_FAILED;
    matmul_tiling::MatmulApiTiling api(platform);
    using namespace matmul_tiling;
    api.SetAType(AscendC::TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    api.SetBType(AscendC::TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    api.SetCType(AscendC::TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    if (api.SetOrgShape(m, n, k)==-1 || api.SetShape(PP_TILE_M, PP_TILE_N, k)==-1) return ge::GRAPH_FAILED;
    if (api.SetFixSplit(PP_TILE_M, PP_TILE_N, PP_TILE_K)==-1) return ge::GRAPH_FAILED;
    api.EnableBias(false);
    api.SetBufferSpace(-1, -1, -1);
    PpMatmulTilingData data;
    if (api.GetTiling(data.cube)==-1) return ge::GRAPH_FAILED;
    data.set_m(m); data.set_n(n); data.set_k(k);
    ctx->SetBlockDim(PP_BLOCK_DIM);
    ctx->SetTilingKey(1);
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    ctx->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
    std::fprintf(stderr, "PARETOPILOT_PARAMS=%s\n", PP_PARAMS_HASH);
    return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class PpMatmul : public OpDef {
public:
    explicit PpMatmul(const char *name) : OpDef(name) {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape([](gert::InferShapeContext *ctx) {
            const auto *a=ctx->GetInputShape(0); const auto *b=ctx->GetInputShape(1);
            if (a->GetDimNum()!=2 || b->GetDimNum()!=2) return ge::GRAPH_FAILED;
            auto *out=ctx->GetOutputShape(0); out->SetDimNum(2); out->SetDim(0,a->GetDim(0)); out->SetDim(1,b->GetDim(1));
            return ge::GRAPH_SUCCESS;
        });
        this->SetInferDataType([](gert::InferDataTypeContext *ctx) { ctx->SetOutputDataType(0,ctx->GetInputDataType(0)); return ge::GRAPH_SUCCESS; });
        this->AICore().SetTiling(optiling::PpMatmulTiling).AddConfig("ascend910b");
    }
};
OP_ADD(PpMatmul);
}
