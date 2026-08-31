#include "hardware_profile.h"
#include <CL/cl.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <windows.h>

HardwareProfile HardwareProfile::probe_system() {
    HardwareProfile prof;

    // 1. Host Memory
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        prof.host_ram_total_bytes = mem_info.ullTotalPhys;
        prof.host_ram_available_bytes = mem_info.ullAvailPhys;
    }

    // 2. CPU
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    prof.cpu_cores = sys_info.dwNumberOfProcessors;
    prof.cpu_brand = "Intel Core i5-11400H / x86_64";

    // 3. OpenCL / GPU Devices
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms > 0) {
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        for (auto p : platforms) {
            cl_uint num_devs = 0;
            clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devs);
            if (num_devs == 0) continue;

            std::vector<cl_device_id> devs(num_devs);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, num_devs, devs.data(), nullptr);

            for (auto d : devs) {
                DeviceStats st;
                char name[256] = {0};
                clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
                st.device_name = name;

                cl_ulong gmem = 0;
                clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
                st.total_memory_bytes = (size_t)gmem;
                st.free_memory_bytes = (size_t)gmem;

                cl_ulong max_alloc = 0;
                clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
                st.max_alloc_bytes = (size_t)max_alloc;

                cl_uint cu = 0;
                clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                st.compute_units = (int)cu;

                if (st.device_name.find("GeForce") != std::string::npos || st.device_name.find("NVIDIA") != std::string::npos || st.device_name.find("Quadro") != std::string::npos) {
                    st.type = BackendDeviceType::NVIDIA_GPU;
                    st.memory_bandwidth_gbs = 128.0; // GTX 1650 GDDR6
                    st.dma_transfer_bandwidth_gbs = 11.5; // PCIe 3.0 x16 / x8
                    st.tflops_fp32 = 2.98;
                } else if (st.device_name.find("Intel") != std::string::npos || st.device_name.find("UHD") != std::string::npos || st.device_name.find("Iris") != std::string::npos) {
                    st.type = BackendDeviceType::INTEL_IGPU;
                    st.memory_bandwidth_gbs = 25.6; // Dual channel DDR4 shared
                    st.dma_transfer_bandwidth_gbs = 25.6; // Unified direct host access
                    st.tflops_fp32 = 0.55;
                } else {
                    st.type = BackendDeviceType::UNKNOWN;
                }
                prof.devices.push_back(st);
            }
        }
    }

    return prof;
}

bool HardwareProfile::save_to_file(const std::string &path) const {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "{\n";
    out << "  \"cpu\": {\n";
    out << "    \"brand\": \"" << cpu_brand << "\",\n";
    out << "    \"cores\": " << cpu_cores << ",\n";
    out << "    \"ram_total_mb\": " << (host_ram_total_bytes / (1024 * 1024)) << ",\n";
    out << "    \"ram_avail_mb\": " << (host_ram_available_bytes / (1024 * 1024)) << "\n";
    out << "  },\n";
    out << "  \"devices\": [\n";
    for (size_t i = 0; i < devices.size(); i++) {
        const auto &d = devices[i];
        out << "    {\n";
        out << "      \"name\": \"" << d.device_name << "\",\n";
        out << "      \"memory_mb\": " << (d.total_memory_bytes / (1024 * 1024)) << ",\n";
        out << "      \"max_alloc_mb\": " << (d.max_alloc_bytes / (1024 * 1024)) << ",\n";
        out << "      \"compute_units\": " << d.compute_units << ",\n";
        out << "      \"bandwidth_gbs\": " << d.memory_bandwidth_gbs << ",\n";
        out << "      \"dma_transfer_gbs\": " << d.dma_transfer_bandwidth_gbs << "\n";
        out << "    }" << (i + 1 < devices.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}
