#include "async_prefetcher.h"
#include <cstdio>

AsyncPrefetcher::AsyncPrefetcher(cl_context context, cl_command_queue compute_queue)
    : context_(context), compute_queue_(compute_queue), dma_queue_(nullptr), initialized_(false) {
    for (int i = 0; i < 2; i++) {
        slots_[i] = {nullptr, 0, false, nullptr};
    }
}

AsyncPrefetcher::~AsyncPrefetcher() {
    for (int i = 0; i < 2; i++) {
        if (slots_[i].ready_event) {
            clReleaseEvent(slots_[i].ready_event);
            slots_[i].ready_event = nullptr;
        }
        if (slots_[i].mem) {
            clReleaseMemObject(slots_[i].mem);
            slots_[i].mem = nullptr;
        }
    }
    if (dma_queue_) {
        clReleaseCommandQueue(dma_queue_);
        dma_queue_ = nullptr;
    }
}

bool AsyncPrefetcher::initialize(size_t staging_slot_capacity_bytes) {
    if (initialized_) return true;

    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(compute_queue_, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
    if (!dev) return false;

    cl_int err;
    dma_queue_ = clCreateCommandQueue(context_, dev, 0, &err);
    if (err != CL_SUCCESS || !dma_queue_) return false;

    for (int i = 0; i < 2; i++) {
        slots_[i].mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, staging_slot_capacity_bytes, nullptr, &err);
        if (err != CL_SUCCESS || !slots_[i].mem) return false;
        slots_[i].capacity = staging_slot_capacity_bytes;
        slots_[i].is_in_use = false;
        slots_[i].ready_event = nullptr;
    }

    initialized_ = true;
    return true;
}

bool AsyncPrefetcher::prefetch_async(const void *host_src, size_t bytes, int slot_idx) {
    if (!initialized_ || slot_idx < 0 || slot_idx >= 2) return false;
    if (bytes > slots_[slot_idx].capacity) return false;

    if (slots_[slot_idx].ready_event) {
        clReleaseEvent(slots_[slot_idx].ready_event);
        slots_[slot_idx].ready_event = nullptr;
    }

    // Non-blocking DMA write into the staging buffer
    cl_int err = clEnqueueWriteBuffer(
        dma_queue_,
        slots_[slot_idx].mem,
        CL_FALSE, // non-blocking!
        0,
        bytes,
        host_src,
        0,
        nullptr,
        &slots_[slot_idx].ready_event
    );

    if (err != CL_SUCCESS) return false;

    // Flush DMA queue to ensure transfer is dispatched to PCIe controller immediately
    clFlush(dma_queue_);
    return true;
}

void AsyncPrefetcher::wait_ready(int slot_idx) {
    if (!initialized_ || slot_idx < 0 || slot_idx >= 2) return;
    if (slots_[slot_idx].ready_event) {
        clWaitForEvents(1, &slots_[slot_idx].ready_event);
        clReleaseEvent(slots_[slot_idx].ready_event);
        slots_[slot_idx].ready_event = nullptr;
    }
}

cl_mem AsyncPrefetcher::get_staging_mem(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= 2) return nullptr;
    return slots_[slot_idx].mem;
}
