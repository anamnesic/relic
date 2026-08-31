#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "hardware_profile.h"
#include "../model.h"

struct TensorCostEstimate
{
    std::string tensor_name;
    BackendDeviceType device;
    GgmlType representation;
    size_t memory_bytes;
    double dma_transfer_time_ms;
    double compute_time_ms;
    double total_latency_ms;
    double benefit_per_byte;
};

struct TensorPlacementDecision
{
    std::string tensor_name;
    MemoryTier target_tier;
    BackendDeviceType target_device;
    GgmlType representation;
    size_t resident_bytes;
    bool keep_resident_in_vram;
    bool enable_async_prefetch;
    double estimated_benefit_per_byte;
    double data_movement_cost_ms;
};

struct ExecutionPlan
{
    size_t vram_required_bytes = 0;
    size_t host_ram_required_bytes = 0;
    size_t total_model_uncompressed_bytes = 0;
    double vram_footprint_reduction_pct = 0.0;
    double memory_traffic_reduction_pct = 0.0;
    double estimated_dma_overlap_efficiency = 0.0;
    double total_data_movement_cost_ms = 0.0;
    int num_layers_fully_offloaded = 0;
    int num_layers_sublayer_offloaded = 0;
    BackendDeviceType primary_device = BackendDeviceType::UNKNOWN;
    BackendDeviceType secondary_device = BackendDeviceType::UNKNOWN;
    std::unordered_map<std::string, TensorPlacementDecision> tensor_placements;
    std::vector<TensorCostEstimate> cost_rankings;
};

class AdaptivePlanner
{
public:
    static size_t get_tensor_vram_size(const LlamaModel::Tensor &tensor, BackendDeviceType target_device, GgmlType target_type);

    static ExecutionPlan generate_plan(
        const LlamaModel &model,
        const HardwareProfile &hardware,
        size_t vram_budget_bytes);
};
