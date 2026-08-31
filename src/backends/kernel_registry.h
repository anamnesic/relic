#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "backend.h"

struct KernelConfig {
    std::string kernel_name;
    size_t workgroup_size = 32;
    size_t vector_width = 4;
    size_t rows_per_workgroup = 1;
    bool use_subgroups = false;
    bool use_local_reduction = true;
};

class KernelRegistry {
public:
    static KernelRegistry& instance();

    KernelConfig get_best_config(const std::string &kernel_name, const DeviceProfile &profile);
    void register_config(BackendDeviceType dev_type, const std::string &kernel_name, const KernelConfig &cfg);
    void autotune(const DeviceProfile &profile);

private:
    KernelRegistry();
    std::unordered_map<std::string, KernelConfig> configs_;
};
