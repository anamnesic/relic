#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "hardware_profile.h"
#include "../model.h"

struct TensorPlacementDecision {
    std::string tensor_name;
    MemoryTier target_tier;
    BackendDeviceType target_device;
    bool keep_resident_in_vram;
    bool enable_async_prefetch;
};

struct ExecutionPlan {
    size_t vram_required_bytes;
    size_t host_ram_required_bytes;
    int num_layers_fully_offloaded;
    int num_layers_sublayer_offloaded;
    std::unordered_map<std::string, TensorPlacementDecision> tensor_placements;
};

class AdaptivePlanner {
public:
    static ExecutionPlan generate_plan(
        const LlamaModel &model,
        const HardwareProfile &hardware,
        size_t vram_budget_bytes
    );
};
