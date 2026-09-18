#include "qwen35_weights_manager.h"
#include "quantization/quant_utils.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

Qwen35WeightsManager::Qwen35WeightsManager(OpenClBackend *backend)
    : cl_(backend)
{
}

bool Qwen35WeightsManager::init(const ArchitectureSpec &arch, const ExecutionPlan *plan)
{
    arch_ = arch;
    plan_ = plan;

    if (!cl_ || !cl_->initialized)
        return false;

    int64_t n_embd = arch_.n_embd;
    int64_t n_ff = arch_.n_ff;

    gpu_act_dst_.alloc(cl_->dev.context, (size_t)(std::max(n_embd, n_ff) * sizeof(float)));
    gpu_weights_.alloc(cl_->dev.context, (size_t)(4096 * std::max(n_embd, n_ff) * sizeof(float)));
    staging_weights_.resize((size_t)std::max(n_embd * (int64_t)4096, n_embd * n_ff * 4), 0.0f);

    pinned_pool_ = std::make_unique<PinnedHostPool>(128 * 1024 * 1024);
    prefetcher_ = std::make_unique<AsyncPrefetcher>(cl_->dev.context, cl_->dev.queue);
    prefetcher_->initialize(64 * 1024 * 1024);

    intel_backend_ = std::make_unique<IntelUhdBackend>();
    if (intel_backend_->initialize())
    {
        uhd_in_buf_ = intel_backend_->allocate(n_embd * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
        uhd_dst_buf_ = intel_backend_->allocate(std::max(n_embd, n_ff) * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
    }
    else
    {
        intel_backend_.reset();
    }

    return true;
}

void Qwen35WeightsManager::ensure_weights_uploaded(const LlamaModel &model)
{
    if (!cl_ || !cl_->initialized || weights_uploaded_)
        return;

    fprintf(stdout, "Uploading model weights to GPU VRAM (ExecutionPlan guided with On-the-Fly Q4 Repacking)...\n");
    fflush(stdout);
    size_t total_uploaded = 0;
    size_t total_original = 0;
    size_t total_offloaded_to_host = 0;

    // 1. Fuse QKV and Gate weights for recurrent layers
    for (int64_t l = 0; l < arch_.n_layer; l++)
    {
        bool is_full_attn = (arch_.full_attention_interval > 0) &&
                            ((l + 1) % arch_.full_attention_interval == 0);
        if (is_full_attn) continue;

        std::string prefix = "blk." + std::to_string(l) + ".";
        auto it_qkv = model.tensors.find(prefix + "attn_qkv.weight");
        auto it_gate = model.tensors.find(prefix + "attn_gate.weight");
        if (it_qkv != model.tensors.end() && it_gate != model.tensors.end())
        {
            const auto &t_qkv = it_qkv->second;
            const auto &t_gate = it_gate->second;
            if (t_qkv.type == t_gate.type)
            {
                std::vector<uint8_t> fused_data;
                fused_data.reserve(t_qkv.data.size() + t_gate.data.size());
                fused_data.insert(fused_data.end(), t_qkv.data.begin(), t_qkv.data.end());
                fused_data.insert(fused_data.end(), t_gate.data.begin(), t_gate.data.end());

                int64_t total_elems = t_qkv.nelements() + t_gate.nelements();
                size_t uploaded_bytes = 0;
                std::string fused_name = prefix + "attn_qkv_gate.weight";
                if (gpu_store_.upload_auto_q4(cl_->dev.context, cl_->dev.queue, fused_name,
                                             fused_data.data(), fused_data.size(), t_qkv.type,
                                             total_elems, uploaded_bytes))
                {
                    total_uploaded += uploaded_bytes;
                }
            }
        }
    }

    for (const auto &kv : model.tensors)
    {
        total_original += kv.second.data.size();

        size_t blk_pos = kv.first.find("blk.");
        if (blk_pos != std::string::npos)
        {
            size_t dot_pos = kv.first.find('.', blk_pos + 4);
            if (dot_pos != std::string::npos)
            {
                std::string prefix = kv.first.substr(0, dot_pos + 1);
                if ((kv.first == prefix + "attn_qkv.weight" || kv.first == prefix + "attn_gate.weight") &&
                    gpu_store_.get(prefix + "attn_qkv_gate.weight") != nullptr)
                {
                    continue; // Already resident in fused GPU VRAM buffer
                }
            }
        }

        size_t uploaded_bytes = 0;

        BackendDeviceType target_dev = BackendDeviceType::NVIDIA_GPU;
        bool keep_in_vram = true;
        if (plan_)
        {
            auto it = plan_->tensor_placements.find(kv.first);
            if (it != plan_->tensor_placements.end())
            {
                target_dev = it->second.target_device;
                keep_in_vram = it->second.keep_resident_in_vram;
            }
        }

        if (target_dev == BackendDeviceType::INTEL_IGPU && intel_backend_)
        {
            auto uhd_buf = intel_backend_->allocate(kv.second.data.size(), MemoryTier::TIER1_SHARED_IGPU);
            if (uhd_buf)
            {
                intel_backend_->upload(*uhd_buf, kv.second.data.data(), kv.second.data.size());
                intel_tensors_[kv.first] = std::move(uhd_buf);
                total_uploaded += kv.second.data.size();
            }
        }
        else if (target_dev == BackendDeviceType::NVIDIA_GPU && keep_in_vram)
        {
            if (kv.first.find("norm") != std::string::npos || kv.first.find("bias") != std::string::npos)
            {
                // Pre-dequantize all norm weights to resident F32 in VRAM once!
                std::vector<float> norm_f32(kv.second.nelements());
                model.dequantize_to_f32(kv.second, norm_f32.data());
                if (gpu_store_.upload_auto_q4(cl_->dev.context, cl_->dev.queue, kv.first, norm_f32.data(), norm_f32.size() * sizeof(float), GgmlType::F32, kv.second.nelements(), uploaded_bytes))
                {
                    total_uploaded += uploaded_bytes;
                }
            }
            else if (gpu_store_.upload_auto_q4(cl_->dev.context, cl_->dev.queue, kv.first, kv.second.data.data(), kv.second.data.size(), kv.second.type, kv.second.nelements(), uploaded_bytes))
            {
                total_uploaded += uploaded_bytes;
            }
        }
        else if (target_dev == BackendDeviceType::CPU_AVX2 || !keep_in_vram)
        {
            if (pinned_pool_)
            {
                void *p = pinned_pool_->allocate_pinned(kv.second.data.size());
                if (p)
                {
                    memcpy(p, kv.second.data.data(), kv.second.data.size());
                    pinned_tensor_ptrs_[kv.first] = p;
                    total_offloaded_to_host += kv.second.data.size();
                }
            }
        }
    }
    clFinish(cl_->dev.queue);
    double footprint_red = (total_original > 0) ? ((double)total_original - (double)total_uploaded) / (double)total_original * 100.0 : 0.0;
    double pcie_red = (plan_) ? plan_->pcie_traffic_reduction_pct : ((total_original > 0) ? ((double)total_uploaded / (double)total_original * 100.0) : 100.0);

    fprintf(stdout, "VRAM Residency active: %.2f MB resident in GPU memory (Footprint Reduction: %.1f%%, PCIe Traffic Reduction: %.1f%%).\n",
            (double)total_uploaded / (1024.0 * 1024.0), footprint_red, pcie_red);
    if (total_offloaded_to_host > 0)
    {
        fprintf(stdout, "[ExecutionPlan Sub-Layer Offload] %.2f MB managed in Pinned Host RAM via AsyncPrefetcher staging.\n",
                (double)total_offloaded_to_host / (1024.0 * 1024.0));
    }
    fflush(stdout);
    weights_uploaded_ = true;
}

ClBuffer *Qwen35WeightsManager::get_tensor_buffer(const std::string &name)
{
    return gpu_store_.get(name);
}

GgmlType Qwen35WeightsManager::get_tensor_type(const std::string &name, GgmlType def)
{
    return gpu_store_.get_type(name, def);
}

void Qwen35WeightsManager::dispatch_gemv(ClBuffer &dst, ClBuffer &in, const std::string &name,
                                        const LlamaModel::Tensor &t, int64_t N, int64_t K, int64_t layer_idx)
{
    // 1. Check if resident in dedicated VRAM
    ClBuffer *w_buf = gpu_store_.get(name);
    GgmlType actual_type = gpu_store_.get_type(name, t.type);
    if (w_buf)
    {
        if (actual_type == GgmlType::Q4_0)
        {
            cl_->gemv_q4_0(dst, in, *w_buf, N, K);
            return;
        }
        else if (actual_type == GgmlType::Q8_0)
        {
            cl_->gemv_q8_0(dst, in, *w_buf, N, K);
            return;
        }
        else if (actual_type == GgmlType::F32)
        {
            cl_->gemv_f32_nt(dst, in, *w_buf, N, K);
            return;
        }
    }

    // 2. Check if resident on Intel UHD shared memory
    auto uhd_it = intel_tensors_.find(name);
    if (uhd_it != intel_tensors_.end() && intel_backend_ && uhd_in_buf_ && uhd_dst_buf_)
    {
        if (staging_weights_.size() < (size_t)std::max(N, K))
            staging_weights_.resize((size_t)std::max(N, K));

        clEnqueueReadBuffer(cl_->dev.queue, in.mem, CL_TRUE, 0, (size_t)(K * sizeof(float)), staging_weights_.data(), 0, nullptr, nullptr);
        intel_backend_->upload(*uhd_in_buf_, staging_weights_.data(), (size_t)(K * sizeof(float)));
        if (t.type == GgmlType::Q4_0)
        {
            intel_backend_->gemv_q4_0(*uhd_dst_buf_, *uhd_in_buf_, *(uhd_it->second), N, K);
        }
        else if (t.type == GgmlType::Q8_0)
        {
            intel_backend_->gemv_q8_0(*uhd_dst_buf_, *uhd_in_buf_, *(uhd_it->second), N, K);
        }
        intel_backend_->synchronize();
        intel_backend_->download(staging_weights_.data(), *uhd_dst_buf_, (size_t)(N * sizeof(float)));
        clEnqueueWriteBuffer(cl_->dev.queue, dst.mem, CL_TRUE, 0, (size_t)(N * sizeof(float)), staging_weights_.data(), 0, nullptr, nullptr);
        return;
    }

    // 3. Check if stored in Pinned Host Pool (AsyncPrefetcher DMA Stream)
    auto pin_it = pinned_tensor_ptrs_.find(name);
    if (pin_it != pinned_tensor_ptrs_.end() && prefetcher_)
    {
        int slot = (int)(layer_idx % 2);
        size_t raw_bytes = t.data.size();
        bool overlap_enabled = plan_ ? plan_->enable_dma_overlap : true;

        if (overlap_enabled)
        {
            prefetcher_->prefetch_async(pin_it->second, raw_bytes, slot);
            prefetcher_->wait_ready(slot);
            cl_mem staging_mem = prefetcher_->get_staging_mem(slot);
            if (staging_mem)
            {
                ClBuffer staging_buf = ClBuffer::borrow(staging_mem, raw_bytes);
                if (t.type == GgmlType::Q4_0)
                {
                    cl_->gemv_q4_0(dst, in, staging_buf, N, K);
                    return;
                }
                else if (t.type == GgmlType::Q8_0)
                {
                    cl_->gemv_q8_0(dst, in, staging_buf, N, K);
                    return;
                }
            }
        }
        else
        {
            cl_mem staging_mem = prefetcher_->get_staging_mem(slot);
            if (staging_mem)
            {
                clEnqueueWriteBuffer(cl_->dev.queue, staging_mem, CL_TRUE, 0, raw_bytes, pin_it->second, 0, nullptr, nullptr);
                clFinish(cl_->dev.queue);
                ClBuffer staging_buf = ClBuffer::borrow(staging_mem, raw_bytes);
                if (t.type == GgmlType::Q4_0)
                {
                    cl_->gemv_q4_0(dst, in, staging_buf, N, K);
                    return;
                }
                else if (t.type == GgmlType::Q8_0)
                {
                    cl_->gemv_q8_0(dst, in, staging_buf, N, K);
                    return;
                }
            }
        }
    }

    // 4. Fallback: chunked dequantization to staging buffer
    const int64_t chunk_n = 4096;
    for (int64_t start_n = 0; start_n < N; start_n += chunk_n)
    {
        int64_t cur_n = std::min(chunk_n, N - start_n);
        int64_t row_size = t.dims.empty() ? t.nelements() : t.dims[0];
        int64_t total_rows = t.dims.size() > 1 ? t.dims[1] : (row_size > 0 ? t.nelements() / row_size : 1);
        int64_t valid_rows = std::min(cur_n, total_rows - start_n);
        relic::quant::dequantize_rows(t.type, t.data.data(), row_size, start_n, valid_rows, staging_weights_.data());

        clEnqueueWriteBuffer(cl_->dev.queue, gpu_weights_.mem, CL_FALSE, 0, (size_t)(cur_n * K * sizeof(float)), staging_weights_.data(), 0, nullptr, nullptr);
        cl_->matmul_f32_nt(gpu_act_dst_, in, gpu_weights_, 1, cur_n, K);
        clEnqueueCopyBuffer(cl_->dev.queue, gpu_act_dst_.mem, dst.mem, 0, (size_t)(start_n * sizeof(float)), (size_t)(cur_n * sizeof(float)), 0, nullptr, nullptr);
    }
}
