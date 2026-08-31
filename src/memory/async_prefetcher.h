#pragma once
#include <CL/cl.h>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <memory>
#include "../backends/backend.h"

struct StagingSlot {
    cl_mem mem;
    size_t capacity;
    bool is_in_use;
    cl_event ready_event;
};

class AsyncPrefetcher {
public:
    AsyncPrefetcher(cl_context context, cl_command_queue compute_queue);
    ~AsyncPrefetcher();

    bool initialize(size_t staging_slot_capacity_bytes);

    // Initiates non-blocking DMA transfer of next layer weights to GPU staging slot
    bool prefetch_async(const void *host_src, size_t bytes, int slot_idx);

    // Ensures the transfer into slot_idx is completed before compute starts
    void wait_ready(int slot_idx);

    cl_mem get_staging_mem(int slot_idx);

private:
    cl_context context_;
    cl_command_queue compute_queue_;
    cl_command_queue dma_queue_;
    StagingSlot slots_[2];
    bool initialized_;
};
