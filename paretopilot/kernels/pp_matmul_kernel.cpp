#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "paretopilot_params.h"
using namespace AscendC;
using MT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
extern "C" __global__ __aicore__ void pp_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GET_TILING_DATA(data, tiling);
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr()==nullptr) return;
    TPipe pipe;
    Matmul<MT, MT, MT, MT> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &data.cube);
    GlobalTensor<half> ga, gb, gc;
    ga.SetGlobalBuffer((__gm__ half*)a, static_cast<uint64_t>(data.m)*data.k);
    gb.SetGlobalBuffer((__gm__ half*)b, static_cast<uint64_t>(data.k)*data.n);
    gc.SetGlobalBuffer((__gm__ half*)c, static_cast<uint64_t>(data.m)*data.n);
    const uint32_t rows=(data.m+PP_TILE_M-1)/PP_TILE_M, cols=(data.n+PP_TILE_N-1)/PP_TILE_N;
    for (uint64_t tile=GetBlockIdx(); tile<static_cast<uint64_t>(rows)*cols; tile+=GetBlockNum()) {
        const uint32_t row=(tile/cols)*PP_TILE_M, col=(tile%cols)*PP_TILE_N;
        const uint32_t tm=data.m-row<PP_TILE_M ? data.m-row : PP_TILE_M;
        const uint32_t tn=data.n-col<PP_TILE_N ? data.n-col : PP_TILE_N;
        mm.SetTail(tm,tn,data.k);
        mm.SetTensorA(ga[static_cast<uint64_t>(row)*data.k]);
        mm.SetTensorB(gb[col]);
        mm.IterateAll(gc[static_cast<uint64_t>(row)*data.n+col]);
        mm.End();
    }
}
