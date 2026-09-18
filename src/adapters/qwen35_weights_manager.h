#pragma once

#include "model.h"
#include "opencl_backend.h"
#include "planner/adaptive_planner.h"
#include "memory/async_prefetcher.h"
#include "memory/pinned_host_pool.h"
#include "backends/intel_uhd_backend.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class Qwen35WeightsManager
{
public:
    explicit Qwen35WeightsManager(OpenClBackend *backend);
    ~Qwen35WeightsManager() = default;

    bool init(const ArchitectureSpec &arch, const ExecutionPlan *plan);
    void ensure_weights_uploaded(const LlamaModel &model);
    void dispatch_gemv(ClBuffer &dst, ClBuffer &in, const std::string &name,
                       const LlamaModel::Tensor &t, int64_t N, int64_t K, int64_t layer_idx);

    ClBuffer *get_tensor_buffer(const std::string &name);
    GgmlType get_tensor_type(const std::string &name, GgmlType def = GgmlType::F32);
    bool is_uploaded() const { return weights_uploaded_; }

private:
    OpenClBackend *cl_ = nullptr;
    const ExecutionPlan *plan_ = nullptr;
    ArchitectureSpec arch_;
    bool weights_uploaded_ = false;

    GpuTensorStore gpu_store_;
    ClBuffer gpu_act_dst_;
    ClBuffer gpu_weights_;
    std::vector<float> staging_weights_;

    std::unique_ptr<PinnedHostPool> pinned_pool_;
    std::unique_ptr<AsyncPrefetcher> prefetcher_;
    std::unique_ptr<IntelUhdBackend> intel_backend_;
    std::shared_ptr<BackendBuffer> uhd_in_buf_;
    std::shared_ptr<BackendBuffer> uhd_dst_buf_;
    std::unordered_map<std::string, void *> pinned_tensor_ptrs_;
    std::unordered_map<std::string, std::shared_ptr<BackendBuffer>> intel_tensors_;
};
