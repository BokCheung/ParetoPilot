#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(PpMatmulTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cube);
    TILING_DATA_FIELD_DEF(uint32_t, m);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, k);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PpMatmul, PpMatmulTilingData)
}
