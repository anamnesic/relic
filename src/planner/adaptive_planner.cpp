#include "adaptive_planner.h"
#include <algorithm>
#include <cstdio>

ExecutionPlan AdaptivePlanner::generate_plan(
    const LlamaModel &model,
    const HardwareProfile &hardware,
    size_t vram_budget_bytes
) {
    ExecutionPlan plan;
    plan.vram_required_bytes = 0;
    plan.host_ram_required_bytes = 0;
    plan.num_layers_fully_offloaded = 0;
    plan.num_layers_sublayer_offloaded = 0;

    BackendDeviceType primary_device = BackendDeviceType::NVIDIA_GPU;
    if (hardware.devices.empty()) {
        primary_device = BackendDeviceType::CPU_AVX2;
    }

    size_t total_model_bytes = 0;
    for (const auto &kv : model.tensors) {
        total_model_bytes += kv.second.data.size();
    }

    // Heuristic: If model fits within VRAM budget, place 100% in VRAM
    if (total_model_bytes <= vram_budget_bytes) {
        for (const auto &kv : model.tensors) {
            TensorPlacementDecision dec;
            dec.tensor_name = kv.first;
            dec.target_tier = MemoryTier::TIER0_DEDICATED_VRAM;
            dec.target_device = primary_device;
            dec.keep_resident_in_vram = true;
            dec.enable_async_prefetch = false;

            plan.tensor_placements[kv.first] = dec;
            plan.vram_required_bytes += kv.second.data.size();
        }
        plan.num_layers_fully_offloaded = (int)model.n_layer;
        return plan;
    }

    // Sub-Layer Tensor-Level Placement (Constrained Memory Mode)
    size_t current_vram_usage = 0;

    // 1. Critical Phase: Pin all Norms, Embeddings, Attention & SSM states first
    for (const auto &kv : model.tensors) {
        const std::string &name = kv.first;
        bool is_critical = (name.find("norm") != std::string::npos) ||
                           (name.find("attn_") != std::string::npos) ||
                           (name.find("ssm_") != std::string::npos) ||
                           (name.find("embed") != std::string::npos);

        if (is_critical) {
            TensorPlacementDecision dec;
            dec.tensor_name = name;
            dec.target_tier = MemoryTier::TIER0_DEDICATED_VRAM;
            dec.target_device = primary_device;
            dec.keep_resident_in_vram = true;
            dec.enable_async_prefetch = false;

            plan.tensor_placements[name] = dec;
            plan.vram_required_bytes += kv.second.data.size();
            current_vram_usage += kv.second.data.size();
        }
    }

    // 2. FFN Phase: Allocate FFN layers in VRAM until budget is reached, then offload with Async Prefetch
    for (int64_t l = 0; l < model.n_layer; l++) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        std::string ffn_gate = prefix + "ffn_gate.weight";
        std::string ffn_up   = prefix + "ffn_up.weight";
        std::string ffn_down = prefix + "ffn_down.weight";

        size_t layer_ffn_bytes = 0;
        auto it_g = model.tensors.find(ffn_gate);
        auto it_u = model.tensors.find(ffn_up);
        auto it_d = model.tensors.find(ffn_down);
        if (it_g != model.tensors.end()) layer_ffn_bytes += it_g->second.data.size();
        if (it_u != model.tensors.end()) layer_ffn_bytes += it_u->second.data.size();
        if (it_d != model.tensors.end()) layer_ffn_bytes += it_d->second.data.size();

        if (current_vram_usage + layer_ffn_bytes <= vram_budget_bytes) {
            // Layer FFN fits in VRAM
            if (it_g != model.tensors.end()) plan.tensor_placements[ffn_gate] = {ffn_gate, MemoryTier::TIER0_DEDICATED_VRAM, primary_device, true, false};
            if (it_u != model.tensors.end()) plan.tensor_placements[ffn_up]   = {ffn_up, MemoryTier::TIER0_DEDICATED_VRAM, primary_device, true, false};
            if (it_d != model.tensors.end()) plan.tensor_placements[ffn_down] = {ffn_down, MemoryTier::TIER0_DEDICATED_VRAM, primary_device, true, false};

            current_vram_usage += layer_ffn_bytes;
            plan.vram_required_bytes += layer_ffn_bytes;
            plan.num_layers_fully_offloaded++;
        } else {
            // Sub-layer offload: Offload FFN to Pinned Host RAM with Async Prefetch
            if (it_g != model.tensors.end()) plan.tensor_placements[ffn_gate] = {ffn_gate, MemoryTier::TIER2_HOST_PINNED_RAM, BackendDeviceType::CPU_AVX2, false, true};
            if (it_u != model.tensors.end()) plan.tensor_placements[ffn_up]   = {ffn_up, MemoryTier::TIER2_HOST_PINNED_RAM, BackendDeviceType::CPU_AVX2, false, true};
            if (it_d != model.tensors.end()) plan.tensor_placements[ffn_down] = {ffn_down, MemoryTier::TIER2_HOST_PINNED_RAM, BackendDeviceType::CPU_AVX2, false, true};

            plan.host_ram_required_bytes += layer_ffn_bytes;
            plan.num_layers_sublayer_offloaded++;
        }
    }

    return plan;
}
