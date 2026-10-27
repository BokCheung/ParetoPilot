#pragma once
#include <cstddef>
#include <cstdio>
#include <string>

namespace paretopilot {
// Record the whole API plan, not just its base dimensions. No fields are changed.
template<class CubeTiling>
inline void PrintCubePlan(const char *stage, const CubeTiling &plan)
{
    static constexpr char hex[] = "0123456789abcdef";
    const auto *bytes = reinterpret_cast<const unsigned char *>(&plan);
    std::string raw;
    raw.reserve(sizeof(plan) * 2);
    for (std::size_t i = 0; i < sizeof(plan); ++i) {
        raw.push_back(hex[bytes[i] >> 4]);
        raw.push_back(hex[bytes[i] & 15]);
    }
    std::fprintf(stderr,
        "PARETOPILOT_%s_CUBE_PLAN={\"raw\":\"%s\",\"used_core_num\":%u,"
        "\"single_core_m\":%u,\"single_core_n\":%u,\"single_core_k\":%u,"
        "\"base_m\":%u,\"base_n\":%u,\"base_k\":%u,"
        "\"step_m\":%u,\"step_n\":%u,\"step_ka\":%u,\"step_kb\":%u,"
        "\"depth_a1\":%u,\"depth_b1\":%u}\n",
        stage, raw.c_str(), static_cast<unsigned>(plan.usedCoreNum),
        static_cast<unsigned>(plan.singleCoreM), static_cast<unsigned>(plan.singleCoreN),
        static_cast<unsigned>(plan.singleCoreK), static_cast<unsigned>(plan.baseM),
        static_cast<unsigned>(plan.baseN), static_cast<unsigned>(plan.baseK),
        static_cast<unsigned>(plan.stepM), static_cast<unsigned>(plan.stepN),
        static_cast<unsigned>(plan.stepKa), static_cast<unsigned>(plan.stepKb),
        static_cast<unsigned>(plan.depthA1), static_cast<unsigned>(plan.depthB1));
}
}
