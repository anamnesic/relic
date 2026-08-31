#pragma once
#include <string>
#include <vector>
#include "../backends/backend.h"

struct HardwareProfile {
    std::string timestamp;
    std::string cpu_brand;
    int cpu_cores;
    size_t host_ram_total_bytes;
    size_t host_ram_available_bytes;
    std::vector<DeviceStats> devices;

    static HardwareProfile probe_system();
    bool save_to_file(const std::string &path) const;
    static HardwareProfile load_from_file(const std::string &path);
};
