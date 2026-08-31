#include "adaptive_planner.h"
#include <algorithm>
#include <cstdio>
#include <cmath>

GgmlType AdaptivePlanner::choose_representation(const std::string &tensor_name, const LlamaModel::Tensor &tensor, BackendDeviceType device)
{
    // Norms and biases always stay F32 in execution for numerical precision
    if (tensor_name.find("norm") != std::string::npos || tensor_name.find("bias") != std::string::npos)
    {
        return GgmlType::F32;
    }
    // Embeddings
    if (tensor_name.find("token_embd") != std::string::npos || tensor_name.find("tok_embeddings") != std::string::npos)
    {
        return tensor.type;
    }
    // On GPU: Q8_0 weights get repacked on-the-fly to Q4_0
    if (device == BackendDeviceType::NVIDIA_GPU || device == BackendDeviceType::AMD_GPU || device == BackendDeviceType::INTEL_IGPU)
    {
        if (tensor.type == GgmlType::Q8_0)
        {
            return GgmlType::Q4_0;
        }
        return tensor.type;
    }
    // On CPU
    return tensor.type;
}

size_t AdaptivePlanner::get_tensor_vram_size(const LlamaModel::Tensor &tensor, BackendDeviceType target_device, GgmlType target_type)
{
    int64_t ne = 1;
    for (auto d : tensor.dims)
        ne *= d;
    if (ne <= 0 && !tensor.data.empty())
        return tensor.data.size();

    // If repacking on-the-fly to Q4_0 on GPU
    if (target_type == GgmlType::Q4_0)
    {
        int64_t n_blocks = (ne + 31) / 32;
        return (size_t)n_blocks * 18;
    }
    else if (target_type == GgmlType::Q8_0 || tensor.type == GgmlType::Q8_0)
    {
        int64_t n_blocks = (ne + 31) / 32;
        return (size_t)n_blocks * 34;
    }
    else if (target_type == GgmlType::F16 || tensor.type == GgmlType::F16)
    {
        return (size_t)ne * 2;
    }
    else if (target_type == GgmlType::F32 || tensor.type == GgmlType::F32)
    {
        return (size_t)ne * 4;
    }

    return tensor.data.size();
}

ExecutionPlan AdaptivePlanner::generate_plan(
    const LlamaModel &model,
    const HardwareProfile &hardware,
    size_t vram_budget_bytes)
{
    ExecutionPlan plan;
    plan.vram_required_bytes = 0;
    plan.host_ram_required_bytes = 0;
    plan.num_layers_fully_offloaded = 0;
    plan.num_layers_sublayer_offloaded = 0;

    // 1. Dynamic Device Selection from HardwareProfile
    BackendDeviceType primary_device = BackendDeviceType::CPU_AVX2;
    BackendDeviceType secondary_device = BackendDeviceType::CPU_AVX2;
    double dgpu_bw = 128.0; // GB/s
    double dma_bw = 11.5;   // GB/s PCIe
    double igpu_bw = 25.6;  // GB/s
    double cpu_bw = 25.6;   // GB/s DDR4

    for (const auto &dev : hardware.devices)
    {
        if (dev.type == BackendDeviceType::NVIDIA_GPU || dev.type == BackendDeviceType::AMD_GPU)
        {
            primary_device = dev.type;
            dgpu_bw = dev.memory_bandwidth_gbs > 0 ? dev.memory_bandwidth_gbs : 128.0;
            dma_bw = dev.dma_transfer_bandwidth_gbs > 0 ? dev.dma_transfer_bandwidth_gbs : 11.5;
        }
        else if (dev.type == BackendDeviceType::INTEL_IGPU)
        {
            if (primary_device == BackendDeviceType::CPU_AVX2)
            {
                primary_device = dev.type;
            }
            else
            {
                secondary_device = dev.type;
            }
            igpu_bw = dev.memory_bandwidth_gbs > 0 ? dev.memory_bandwidth_gbs : 25.6;
        }
    }

    plan.primary_device = primary_device;
    plan.secondary_device = secondary_device;

    // 2. Compute total uncompressed bytes & estimate costs per tensor
    size_t total_uncompressed_bytes = 0;
    std::vector<TensorCostEstimate> estimates;

    for (const auto &kv : model.tensors)
    {
        const auto &tensor = kv.second;
        size_t raw_bytes = tensor.data.size();
        total_uncompressed_bytes += raw_bytes;

        GgmlType rep = choose_representation(kv.first, tensor, primary_device);
        size_t vram_bytes = get_tensor_vram_size(tensor, primary_device, rep);
        int64_t ne = 1;
        for (auto d : tensor.dims)
            ne *= d;
        double flops = (double)ne * 2.0;

        // Latency in milliseconds: Data Transfer + Compute
        double t_dma_ms = (double)vram_bytes / (dma_bw * 1e9) * 1000.0;
        double t_gpu_comp_ms = ((double)vram_bytes / (dgpu_bw * 1e9) + flops / (2.98e12)) * 1000.0;
        double t_cpu_comp_ms = ((double)raw_bytes / (cpu_bw * 1e9) + flops / (0.15e12)) * 1000.0;

        double delta_latency = t_cpu_comp_ms - t_gpu_comp_ms;
        double bpb = (vram_bytes > 0) ? (delta_latency / (double)vram_bytes) : 0.0;

        TensorCostEstimate est;
        est.tensor_name = kv.first;
        est.device = primary_device;
        est.representation = rep;
        est.memory_bytes = vram_bytes;
        est.dma_transfer_time_ms = t_dma_ms;
        est.compute_time_ms = t_gpu_comp_ms;
        est.total_latency_ms = t_gpu_comp_ms;
        est.benefit_per_byte = bpb;

        estimates.push_back(est);
    }

    plan.total_model_uncompressed_bytes = total_uncompressed_bytes;

    // Sort tensors by benefit_per_byte (critical structures like norms, embeddings, attention first)
    std::sort(estimates.begin(), estimates.end(), [](const TensorCostEstimate &a, const TensorCostEstimate &b)
              {
        // Pin norms and attention high
        bool a_crit = (a.tensor_name.find("norm") != std::string::npos) || (a.tensor_name.find("attn_") != std::string::npos);
        bool b_crit = (b.tensor_name.find("norm") != std::string::npos) || (b.tensor_name.find("attn_") != std::string::npos);
        if (a_crit != b_crit) return a_crit;
        return a.benefit_per_byte > b.benefit_per_byte; });

    plan.cost_rankings = estimates;

    // 3. Tensor Placement Decision Loop under VRAM budget
    size_t current_vram_usage = 0;
    double total_compute_time_ms = 0.0;
    double total_transfer_time_ms = 0.0;

    for (const auto &est : estimates)
    {
        auto it = model.tensors.find(est.tensor_name);
        if (it == model.tensors.end())
            continue;

        TensorPlacementDecision dec;
        dec.tensor_name = est.tensor_name;
        dec.representation = est.representation;
        dec.estimated_benefit_per_byte = est.benefit_per_byte;

        if (current_vram_usage + est.memory_bytes <= vram_budget_bytes)
        {
            // Allocate in dedicated VRAM
            dec.target_tier = MemoryTier::TIER0_DEDICATED_VRAM;
            dec.target_device = primary_device;
            dec.resident_bytes = est.memory_bytes;
            dec.keep_resident_in_vram = true;
            dec.enable_async_prefetch = false;
            dec.data_movement_cost_ms = 0.0; // Resident in VRAM: 0 PCIe transfer per token

            current_vram_usage += est.memory_bytes;
            plan.vram_required_bytes += est.memory_bytes;
            plan.gtx_vram_weights_bytes += est.memory_bytes;
            total_compute_time_ms += est.compute_time_ms;
        }
        else
        {
            // Sub-layer offload to Pinned Host RAM with Async Prefetch
            dec.target_tier = (secondary_device == BackendDeviceType::INTEL_IGPU) ? MemoryTier::TIER1_SHARED_IGPU : MemoryTier::TIER2_HOST_PINNED_RAM;
            dec.target_device = (secondary_device == BackendDeviceType::INTEL_IGPU) ? secondary_device : BackendDeviceType::CPU_AVX2;
            size_t host_bytes = !it->second.data.empty() ? it->second.data.size() : est.memory_bytes;
            dec.resident_bytes = host_bytes;
            dec.keep_resident_in_vram = false;
            dec.enable_async_prefetch = true;
            dec.data_movement_cost_ms = est.dma_transfer_time_ms;

            if (secondary_device == BackendDeviceType::INTEL_IGPU)
            {
                plan.intel_uhd_weights_bytes += host_bytes;
            }
            else
            {
                plan.pinned_streamed_weights_bytes += host_bytes;
            }

            plan.host_ram_required_bytes += host_bytes;
            total_transfer_time_ms += est.dma_transfer_time_ms;
        }

        plan.tensor_placements[est.tensor_name] = dec;
    }

    // Accounted Peak GTX VRAM includes: resident weights + activations + SSM states + Conv1D + KV cache + staging buffer (64 MB)
    int64_t n_embd_val = model.architecture.n_embd > 0 ? model.architecture.n_embd : 2048;
    size_t state_overhead_bytes = (size_t)model.n_layer * (16 * 128 * 128 * sizeof(float) + 3 * 2048 * sizeof(float))
                                + (size_t)2048 * n_embd_val * sizeof(float) * 2
                                + (size_t)64 * 1024 * 1024;
    plan.actual_gtx_allocation_peak_bytes = plan.gtx_vram_weights_bytes + state_overhead_bytes;

    // 4. Calculate Layer Offload Counts & Form Contiguous Device Islands
    int64_t n_embd = model.architecture.n_embd > 0 ? model.architecture.n_embd : 2048;
    double single_act_transfer_ms = (double)(n_embd * sizeof(float)) / (dma_bw * 1e9) * 1000.0;

    BackendDeviceType current_island_dev = BackendDeviceType::UNKNOWN;
    MemoryTier current_island_tier = MemoryTier::TIER0_DEDICATED_VRAM;
    int island_start = 0;

    for (int64_t l = 0; l < model.n_layer; l++)
    {
        std::string prefix = "blk." + std::to_string(l) + ".";
        std::string ffn_down = prefix + "ffn_down.weight";
        auto it = plan.tensor_placements.find(ffn_down);
        BackendDeviceType layer_dev = primary_device;
        MemoryTier layer_tier = MemoryTier::TIER0_DEDICATED_VRAM;

        if (it != plan.tensor_placements.end())
        {
            layer_dev = it->second.target_device;
            layer_tier = it->second.target_tier;
            if (it->second.keep_resident_in_vram)
            {
                plan.num_layers_fully_offloaded++;
            }
            else
            {
                plan.num_layers_sublayer_offloaded++;
            }
        }

        if (l == 0)
        {
            current_island_dev = layer_dev;
            current_island_tier = layer_tier;
            island_start = 0;
        }
        else if (layer_dev != current_island_dev)
        {
            plan.islands.push_back({island_start, (int)l - 1, current_island_dev, current_island_tier});
            plan.activation_boundary_cost_ms += single_act_transfer_ms * 2.0; // Two-way activation boundary bridge
            island_start = (int)l;
            current_island_dev = layer_dev;
            current_island_tier = layer_tier;
        }
    }
    if (model.n_layer > 0)
    {
        plan.islands.push_back({island_start, (int)model.n_layer - 1, current_island_dev, current_island_tier});
    }

    // 5. Data Movement & Footprint Reduction Metrics
    if (total_uncompressed_bytes > 0)
    {
        plan.vram_footprint_reduction_pct = ((double)total_uncompressed_bytes - (double)plan.vram_required_bytes) / (double)total_uncompressed_bytes * 100.0;
        if (plan.vram_footprint_reduction_pct < 0.0)
            plan.vram_footprint_reduction_pct = 0.0;
    }

    // PCIe / Host<->Device Traffic reduction compares resident execution vs streaming all weights every token
    double streaming_traffic_bytes = (double)total_uncompressed_bytes;
    double actual_traffic_bytes = (double)plan.host_ram_required_bytes;
    plan.pcie_traffic_reduction_pct = (streaming_traffic_bytes > 0) ? ((streaming_traffic_bytes - actual_traffic_bytes) / streaming_traffic_bytes * 100.0) : 0.0;

    // Overlap efficiency: compute(N) + transfer(N+1)
    if (total_transfer_time_ms > 0.0)
    {
        plan.estimated_dma_overlap_efficiency = std::min(1.0, total_compute_time_ms / total_transfer_time_ms) * 100.0;
    }
    else
    {
        plan.estimated_dma_overlap_efficiency = 100.0; // 100% resident, zero transfer wait
    }
    plan.total_data_movement_cost_ms = total_transfer_time_ms;

    return plan;
}
