#include "kernel_operator.h"
#include "paretopilot_params.h"
using namespace AscendC;
namespace {
class AddKernel {
    TPipe pipe;
    TQue<QuePosition::VECIN, PP_BUFFERS> qa, qb;
    TQue<QuePosition::VECOUT, PP_BUFFERS> qc;
    GlobalTensor<half> a, b, c;
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        a.SetGlobalBuffer((__gm__ half*)x, length);
        b.SetGlobalBuffer((__gm__ half*)y, length);
        c.SetGlobalBuffer((__gm__ half*)z, length);
        pipe.InitBuffer(qa, PP_BUFFERS, PP_TILE_ELEMS*sizeof(half));
        pipe.InitBuffer(qb, PP_BUFFERS, PP_TILE_ELEMS*sizeof(half));
        pipe.InitBuffer(qc, PP_BUFFERS, PP_TILE_ELEMS*sizeof(half));
        const uint64_t tiles = (static_cast<uint64_t>(length)+PP_TILE_ELEMS-1)/PP_TILE_ELEMS;
        for (uint64_t tile=GetBlockIdx(); tile<tiles; tile+=GetBlockNum()) {
            const uint32_t offset=tile*PP_TILE_ELEMS;
            const uint32_t count = length-offset < PP_TILE_ELEMS ? length-offset : PP_TILE_ELEMS;
            DataCopyExtParams cp = {1, count*static_cast<uint32_t>(sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> pad = {false, 0, 0, 0};
            auto la=qa.AllocTensor<half>(); auto lb=qb.AllocTensor<half>();
            DataCopyPad(la, a[offset], cp, pad);
            DataCopyPad(lb, b[offset], cp, pad);
            qa.EnQue(la); qb.EnQue(lb);
            la=qa.DeQue<half>(); lb=qb.DeQue<half>();
            auto lc=qc.AllocTensor<half>();
            Add(lc, la, lb, count);
            qc.EnQue(lc); qa.FreeTensor(la); qb.FreeTensor(lb);
            lc=qc.DeQue<half>();
            DataCopyPad(c[offset], lc, cp);
            qc.FreeTensor(lc);
        }
    }
};
}
extern "C" __global__ __aicore__ void pp_add(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(data, tiling);
    // Advertise the key selected by Host tiling to the CANN kernel compiler.
    if (TILING_KEY_IS(1)) {
        AddKernel kernel;
        kernel.Run(a, b, c, data.length);
    }
}
