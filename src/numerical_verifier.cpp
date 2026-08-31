#include "numerical_verifier.h"
#include "qwen35_state.h"
#include <cmath>
#include <cstdio>
#include <numeric>
#include <algorithm>

double NumericalVerifier::compute_max_abs_error(const float *a, const float *b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = std::fabs((double)a[i] - (double)b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

double NumericalVerifier::compute_mean_abs_error(const float *a, const float *b, size_t n) {
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += std::fabs((double)a[i] - (double)b[i]);
    }
    return sum / (double)n;
}

double NumericalVerifier::compute_cosine_similarity(const float *a, const float *b, size_t n) {
    if (n == 0) return 1.0;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom <= 1e-12) return 1.0;
    return dot / denom;
}

std::vector<NumericalMetric> NumericalVerifier::verify_layers(OpenClBackend *cl_backend, IntelUhdBackend *intel_backend) {
    std::vector<NumericalMetric> results;

    if (!cl_backend || !cl_backend->initialized) return results;

    // 1. RMSNorm Verification
    {
        const int64_t n = 256;
        std::vector<float> x(n), w(n), y_cpu(n), y_gpu(n);
        for (int64_t i = 0; i < n; i++) {
            x[i] = (float)std::sin((double)i * 0.1);
            w[i] = 1.0f + 0.1f * (float)std::cos((double)i * 0.05);
        }

        // CPU reference
        float ss = 0.0f;
        for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
        float s = 1.0f / std::sqrt(ss / (float)n + 1e-5f);
        for (int64_t i = 0; i < n; i++) y_cpu[i] = x[i] * s * w[i];

        // GPU execution
        ClBuffer bx, bw, by;
        bx.alloc(cl_backend->dev.context, n * sizeof(float));
        bw.alloc(cl_backend->dev.context, n * sizeof(float));
        by.alloc(cl_backend->dev.context, n * sizeof(float));

        clEnqueueWriteBuffer(cl_backend->dev.queue, bx.mem, CL_TRUE, 0, n * sizeof(float), x.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(cl_backend->dev.queue, bw.mem, CL_TRUE, 0, n * sizeof(float), w.data(), 0, nullptr, nullptr);
        cl_backend->rms_norm(by, bx, bw, n, 1);
        clFinish(cl_backend->dev.queue);
        clEnqueueReadBuffer(cl_backend->dev.queue, by.mem, CL_TRUE, 0, n * sizeof(float), y_gpu.data(), 0, nullptr, nullptr);

        NumericalMetric m;
        m.layer_name = "layer.norm.rms_norm";
        m.max_absolute_error = compute_max_abs_error(y_cpu.data(), y_gpu.data(), n);
        m.mean_absolute_error = compute_mean_abs_error(y_cpu.data(), y_gpu.data(), n);
        m.cosine_similarity = compute_cosine_similarity(y_cpu.data(), y_gpu.data(), n);
        m.passed = (m.max_absolute_error < 1e-4 && m.cosine_similarity > 0.9999);
        results.push_back(m);
    }

    // 2. AddRMSNorm (Fused Residual + Norm) Verification
    {
        const int64_t n = 256;
        std::vector<float> res_cpu(n), branch(n), w(n), norm_cpu(n), res_gpu(n), norm_gpu(n);
        for (int64_t i = 0; i < n; i++) {
            res_cpu[i] = (float)std::cos((double)i * 0.2);
            branch[i] = (float)std::sin((double)i * 0.3);
            w[i] = 1.0f;
        }
        res_gpu = res_cpu;

        // CPU reference
        float ss = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            res_cpu[i] += branch[i];
            ss += res_cpu[i] * res_cpu[i];
        }
        float s = 1.0f / std::sqrt(ss / (float)n + 1e-5f);
        for (int64_t i = 0; i < n; i++) norm_cpu[i] = res_cpu[i] * s * w[i];

        // GPU execution
        ClBuffer b_res, b_bra, b_w, b_out;
        b_res.alloc(cl_backend->dev.context, n * sizeof(float));
        b_bra.alloc(cl_backend->dev.context, n * sizeof(float));
        b_w.alloc(cl_backend->dev.context, n * sizeof(float));
        b_out.alloc(cl_backend->dev.context, n * sizeof(float));

        clEnqueueWriteBuffer(cl_backend->dev.queue, b_res.mem, CL_TRUE, 0, n * sizeof(float), res_gpu.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(cl_backend->dev.queue, b_bra.mem, CL_TRUE, 0, n * sizeof(float), branch.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(cl_backend->dev.queue, b_w.mem, CL_TRUE, 0, n * sizeof(float), w.data(), 0, nullptr, nullptr);

        cl_backend->add_rms_norm(b_res, b_bra, b_w, b_out, n, 1e-5f);
        clFinish(cl_backend->dev.queue);
        clEnqueueReadBuffer(cl_backend->dev.queue, b_res.mem, CL_TRUE, 0, n * sizeof(float), res_gpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(cl_backend->dev.queue, b_out.mem, CL_TRUE, 0, n * sizeof(float), norm_gpu.data(), 0, nullptr, nullptr);

        NumericalMetric m;
        m.layer_name = "layer.norm.add_rms_norm";
        m.max_absolute_error = compute_max_abs_error(norm_cpu.data(), norm_gpu.data(), n);
        m.mean_absolute_error = compute_mean_abs_error(norm_cpu.data(), norm_gpu.data(), n);
        m.cosine_similarity = compute_cosine_similarity(norm_cpu.data(), norm_gpu.data(), n);
        m.passed = (m.max_absolute_error < 1e-4 && m.cosine_similarity > 0.9999);
        results.push_back(m);
    }

    // 3. DeltaNet Recurrent State Verification
    {
        ArchitectureSpec qwen;
        qwen.n_layer = 1;
        qwen.linear_conv_kernel = 2;
        qwen.linear_inner_size = 2;
        qwen.linear_key_head_dim = 1;
        qwen.linear_key_heads = 1;
        qwen.linear_value_heads = 2;

        Qwen35RecurrentState recurrent;
        recurrent.init(qwen);

        float conv_in[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float conv_weight[] = {2.0f, 3.0f, 2.0f, 3.0f, 2.0f, 3.0f, 2.0f, 3.0f};
        float conv_out[4] = {0};
        recurrent.conv1d(0, conv_in, conv_weight, conv_out);

        float ref_conv[4] = {3.0f, 6.0f, 9.0f, 12.0f};

        NumericalMetric m;
        m.layer_name = "layer.recurrent.deltanet_conv1d";
        m.max_absolute_error = compute_max_abs_error(ref_conv, conv_out, 4);
        m.mean_absolute_error = compute_mean_abs_error(ref_conv, conv_out, 4);
        m.cosine_similarity = compute_cosine_similarity(ref_conv, conv_out, 4);
        m.passed = (m.max_absolute_error < 1e-4);
        results.push_back(m);
    }

    // 4. FFN (SwiGLU) Verification
    {
        const int64_t n = 128;
        std::vector<float> gate(n), up(n), out_cpu(n), out_gpu(n);
        for (int64_t i = 0; i < n; i++) {
            gate[i] = (float)std::sin((double)i * 0.1);
            up[i] = (float)std::cos((double)i * 0.1);
            float silu_val = gate[i] / (1.0f + std::exp(-gate[i]));
            out_cpu[i] = silu_val * up[i];
        }

        ClBuffer bg, bu, bo;
        bg.alloc(cl_backend->dev.context, n * sizeof(float));
        bu.alloc(cl_backend->dev.context, n * sizeof(float));
        bo.alloc(cl_backend->dev.context, n * sizeof(float));

        clEnqueueWriteBuffer(cl_backend->dev.queue, bg.mem, CL_TRUE, 0, n * sizeof(float), gate.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(cl_backend->dev.queue, bu.mem, CL_TRUE, 0, n * sizeof(float), up.data(), 0, nullptr, nullptr);

        cl_backend->swiglu(bo, bg, bu, n);
        clFinish(cl_backend->dev.queue);
        clEnqueueReadBuffer(cl_backend->dev.queue, bo.mem, CL_TRUE, 0, n * sizeof(float), out_gpu.data(), 0, nullptr, nullptr);

        NumericalMetric m;
        m.layer_name = "layer.ffn.swiglu";
        m.max_absolute_error = compute_max_abs_error(out_cpu.data(), out_gpu.data(), n);
        m.mean_absolute_error = compute_mean_abs_error(out_cpu.data(), out_gpu.data(), n);
        m.cosine_similarity = compute_cosine_similarity(out_cpu.data(), out_gpu.data(), n);
        m.passed = (m.max_absolute_error < 1e-4 && m.cosine_similarity > 0.9999);
        results.push_back(m);
    }

    // 5. Logits Output / Argmax Verification
    {
        const int64_t n = 256;
        std::vector<float> logits(n);
        for (int64_t i = 0; i < n; i++) logits[i] = (float)std::sin((double)i * 0.05);
        logits[137] = 42.0f; // Known peak

        int ref_argmax = 137;

        ClBuffer bl, bi;
        bl.alloc(cl_backend->dev.context, n * sizeof(float));
        bi.alloc(cl_backend->dev.context, sizeof(int));

        clEnqueueWriteBuffer(cl_backend->dev.queue, bl.mem, CL_TRUE, 0, n * sizeof(float), logits.data(), 0, nullptr, nullptr);
        cl_backend->argmax(bi, bl, n);
        clFinish(cl_backend->dev.queue);

        int gpu_argmax = 0;
        clEnqueueReadBuffer(cl_backend->dev.queue, bi.mem, CL_TRUE, 0, sizeof(int), &gpu_argmax, 0, nullptr, nullptr);

        NumericalMetric m;
        m.layer_name = "layer.output.logits_argmax";
        m.max_absolute_error = (double)std::abs(ref_argmax - gpu_argmax);
        m.mean_absolute_error = m.max_absolute_error;
        m.cosine_similarity = (ref_argmax == gpu_argmax) ? 1.0 : 0.0;
        m.passed = (ref_argmax == gpu_argmax);
        results.push_back(m);
    }

    return results;
}
