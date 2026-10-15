// Self-test: verify OpenCL 1.2 init and kernel execution on Caicos
#include "opencl_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    fprintf(stdout, "=== Caicos RT OpenCL 1.2 Self-Test ===\n\n");

    OpenClBackend cl;
    if (!cl.init(0, 0)) {
        fprintf(stderr, "FAIL: OpenCL init failed\n");
        return 1;
    }
    fprintf(stdout, "PASS: OpenCL initialized\n\n");

    // Build all kernels
    fprintf(stdout, "Building kernels...\n");
    struct {
        const char *name;
        ClKernel knl;
        bool ok;
    } kernels[] = {
        {"rms_norm_f32"},
        {"matmul_f32"},
        {"matmul_f32_nt"},
        {"rope_f32"},
        {"softmax_f32"},
        {"silu_f32"},
        {"add_f32"},
        {"mul_f32"},
        {"copy_f32"},
        {"fill_f32"},
    };
    int n_kernels = sizeof(kernels) / sizeof(kernels[0]);

    for (int i = 0; i < n_kernels; i++) {
        kernels[i].ok = cl.build_kernel(kernels[i].knl, caicos_kernel_source, kernels[i].name);
        fprintf(stdout, "  %s: %s\n", kernels[i].name, kernels[i].ok ? "OK" : "FAIL");
    }

    // Test matmul
    fprintf(stdout, "\nRunning matmul test...\n");
    int M = 4, N = 4, K = 4;

    // Create buffers on GPU
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float A[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float B[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; // identity
    float C_expected[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float C_gpu[16] = {0};

    ClBuffer buf_a, buf_b, buf_c;
    if (!buf_a.alloc(cl.dev.context, size_a) ||
        !buf_b.alloc(cl.dev.context, size_b) ||
        !buf_c.alloc(cl.dev.context, size_c)) {
        fprintf(stderr, "FAIL: buffer allocation\n");
        return 1;
    }

    cl_int err;
    err = clEnqueueWriteBuffer(cl.dev.queue, buf_a.mem, CL_TRUE, 0, size_a, A, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(cl.dev.queue, buf_b.mem, CL_TRUE, 0, size_b, B, 0, nullptr, nullptr);

    cl.matmul_f32(buf_c, buf_a, buf_b, M, N, K);

    clFinish(cl.dev.queue);
    err |= clEnqueueReadBuffer(cl.dev.queue, buf_c.mem, CL_TRUE, 0, size_c, C_gpu, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "FAIL: matmul execution\n");
        return 1;
    }

    bool pass = true;
    for (int i = 0; i < 16; i++) {
        if (fabsf(C_gpu[i] - C_expected[i]) > 1e-4f) {
            fprintf(stderr, "  Mismatch at [%d]: got %f, expected %f\n",
                    i, C_gpu[i], C_expected[i]);
            pass = false;
        }
    }
    fprintf(stdout, "  matmul result: %s\n", pass ? "PASS" : "FAIL");

    // Test fill + add
    fprintf(stdout, "\nRunning fill+add test...\n");
    ClBuffer buf_x, buf_y;
    float ones[4] = {1,1,1,1};
    float twos[4] = {2,2,2,2};
    float result[4] = {0};

    buf_x.alloc(cl.dev.context, 4 * sizeof(float));
    buf_y.alloc(cl.dev.context, 4 * sizeof(float));

    cl.fill(buf_x, 1.0f, 4);
    cl.fill(buf_y, 1.0f, 4);
    cl.add(buf_x, buf_x, buf_y, 4);
    clFinish(cl.dev.queue);
    clEnqueueReadBuffer(cl.dev.queue, buf_x.mem, CL_TRUE, 0, 4 * sizeof(float), result, 0, nullptr, nullptr);

    pass = true;
    for (int i = 0; i < 4; i++) {
        if (fabsf(result[i] - 2.0f) > 1e-4f) {
            fprintf(stderr, "  add mismatch at [%d]: got %f, expected 2.0\n", i, result[i]);
            pass = false;
        }
    }
    fprintf(stdout, "  add result: %s\n", pass ? "PASS" : "FAIL");

    // Test RMS norm
    fprintf(stdout, "\nRunning rms_norm test...\n");
    float x_data[4] = {1, 2, 3, 4};
    float w_data[4] = {1, 1, 1, 1};
    float y_data[4] = {0};
    float eps = 1e-5f;

    // CPU reference
    float ss = 0;
    for (int i = 0; i < 4; i++) ss += x_data[i] * x_data[i];
    float s = 1.0f / sqrtf(ss / 4.0f + eps);
    float y_ref[4];
    for (int i = 0; i < 4; i++) y_ref[i] = x_data[i] * s * w_data[i];

    ClBuffer buf_xn, buf_wn, buf_yn;
    buf_xn.alloc(cl.dev.context, 4 * sizeof(float));
    buf_wn.alloc(cl.dev.context, 4 * sizeof(float));
    buf_yn.alloc(cl.dev.context, 4 * sizeof(float));

    clEnqueueWriteBuffer(cl.dev.queue, buf_xn.mem, CL_TRUE, 0, 4 * sizeof(float), x_data, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl.dev.queue, buf_wn.mem, CL_TRUE, 0, 4 * sizeof(float), w_data, 0, nullptr, nullptr);

    cl.rms_norm(buf_yn, buf_xn, buf_wn, 4, 1);
    clFinish(cl.dev.queue);
    clEnqueueReadBuffer(cl.dev.queue, buf_yn.mem, CL_TRUE, 0, 4 * sizeof(float), y_data, 0, nullptr, nullptr);

    pass = true;
    for (int i = 0; i < 4; i++) {
        if (fabsf(y_data[i] - y_ref[i]) > 1e-4f) {
            fprintf(stderr, "  rms_norm mismatch at [%d]: got %f, expected %f\n",
                    i, y_data[i], y_ref[i]);
            pass = false;
        }
    }
    fprintf(stdout, "  rms_norm result: %s\n", pass ? "PASS" : "FAIL");

    // Device info summary
    fprintf(stdout, "\n=== Device Summary ===\n");
    fprintf(stdout, "  Name: %s\n", cl.dev.name.c_str());
    fprintf(stdout, "  Platform: %s\n", cl.dev.platform_name.c_str());
    fprintf(stdout, "  OpenCL C: %d.%d\n", cl.dev.opencl_c_major, cl.dev.opencl_c_minor);
    fprintf(stdout, "  FP16: %s\n", cl.dev.fp16 ? "yes" : "no");
    fprintf(stdout, "  Global Mem: %zu MB\n", cl.dev.global_mem / (1024*1024));
    fprintf(stdout, "  Max Alloc: %zu MB\n", cl.dev.max_alloc / (1024*1024));
    fprintf(stdout, "  Compute Units: %u\n", cl.dev.compute_units);
    fprintf(stdout, "  Max WG Size: %zu\n", cl.dev.max_wg_size);

    fprintf(stdout, "\n=== All tests %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
