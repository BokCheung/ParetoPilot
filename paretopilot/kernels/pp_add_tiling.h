#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(PpAddTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, length);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PpAdd, PpAddTilingData)
}
