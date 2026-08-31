#include "kernel_registry.h"
#include <algorithm>
#include <cstdio>

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry reg;
    return reg;
}

KernelRegistry::KernelRegistry() {
    // Default fallback configurations
    // NVIDIA GPU (Pascal/Turing) - 32-thread warp optimized
    KernelConfig nv_gemv_q4{"gemv_q4_0", 32, 4, 16, false, true};
    KernelConfig nv_gemv_ffn{"gemv_q4_0_ffn_swiglu", 32, 4, 8, false, true};
    KernelConfig nv_norm{"rms_norm_f32", 1, 4, 1, false, false};
    KernelConfig nv_add_norm{"add_rms_norm_f32", 1, 4, 1, false, false};

    register_config(BackendDeviceType::NVIDIA_GPU, "gemv_q4_0", nv_gemv_q4);
    register_config(BackendDeviceType::NVIDIA_GPU, "gemv_q4_0_ffn_swiglu", nv_gemv_ffn);
    register_config(BackendDeviceType::NVIDIA_GPU, "rms_norm_f32", nv_norm);
    register_config(BackendDeviceType::NVIDIA_GPU, "add_rms_norm_f32", nv_add_norm);

    // Intel iGPU (Gen9/Gen11/Xe) - Subgroup 16/32 SIMD optimized
    KernelConfig intel_gemv_q4{"gemv_q4_0", 64, 4, 8, true, true};
    KernelConfig intel_gemv_ffn{"gemv_q4_0_ffn_swiglu", 64, 4, 4, true, true};
    KernelConfig intel_norm{"rms_norm_f32", 16, 4, 1, true, false};
    KernelConfig intel_add_norm{"add_rms_norm_f32", 16, 4, 1, true, false};

    register_config(BackendDeviceType::INTEL_IGPU, "gemv_q4_0", intel_gemv_q4);
    register_config(BackendDeviceType::INTEL_IGPU, "gemv_q4_0_ffn_swiglu", intel_gemv_ffn);
    register_config(BackendDeviceType::INTEL_IGPU, "rms_norm_f32", intel_norm);
    register_config(BackendDeviceType::INTEL_IGPU, "add_rms_norm_f32", intel_add_norm);

    // AMD GPU (GCN / RDNA) - Wavefront 64 / 32
    KernelConfig amd_gemv_q4{"gemv_q4_0", 64, 4, 16, false, true};
    KernelConfig amd_gemv_ffn{"gemv_q4_0_ffn_swiglu", 64, 4, 8, false, true};
    register_config(BackendDeviceType::AMD_GPU, "gemv_q4_0", amd_gemv_q4);
    register_config(BackendDeviceType::AMD_GPU, "gemv_q4_0_ffn_swiglu", amd_gemv_ffn);
}

void KernelRegistry::register_config(BackendDeviceType dev_type, const std::string &kernel_name, const KernelConfig &cfg) {
    std::string key = std::to_string((int)dev_type) + ":" + kernel_name;
    configs_[key] = cfg;
}

KernelConfig KernelRegistry::get_best_config(const std::string &kernel_name, const DeviceProfile &profile) {
    std::string key = std::to_string((int)profile.type) + ":" + kernel_name;
    auto it = configs_.find(key);
    if (it != configs_.end()) {
        KernelConfig cfg = it->second;
        // Clamp to device limits
        if (cfg.workgroup_size > profile.max_workgroup_size && profile.max_workgroup_size > 0) {
            cfg.workgroup_size = profile.max_workgroup_size;
        }
        return cfg;
    }

    // Generic fallback
    KernelConfig fallback;
    fallback.kernel_name = kernel_name;
    fallback.workgroup_size = std::min<size_t>(profile.max_workgroup_size > 0 ? profile.max_workgroup_size : 32, 32);
    fallback.vector_width = profile.vector_width > 0 ? profile.vector_width : 4;
    return fallback;
}

void KernelRegistry::autotune(const DeviceProfile &profile) {
    // Autotuner evaluates workgroup sizing against device subgroup & compute units
    if (profile.type == BackendDeviceType::INTEL_IGPU) {
        KernelConfig tuned_gemv{"gemv_q4_0", (size_t)(profile.subgroup_size > 0 ? profile.subgroup_size * 2 : 32), 4, 8, profile.subgroups_supported, true};
        register_config(profile.type, "gemv_q4_0", tuned_gemv);
    } else if (profile.type == BackendDeviceType::NVIDIA_GPU) {
        KernelConfig tuned_gemv{"gemv_q4_0", 32, 4, 16, false, true};
        register_config(profile.type, "gemv_q4_0", tuned_gemv);
    }
}
