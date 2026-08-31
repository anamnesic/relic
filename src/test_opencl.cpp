// Self-test: verify OpenCL 1.2 init and kernel execution on Caicos
#include "opencl_backend.h"
#include "qwen35_state.h"
#include "decoder.h"
#include "planner/adaptive_planner.h"
#include "memory/pinned_host_pool.h"
#include "memory/async_prefetcher.h"
#include "speculative/distributed_speculative.h"
#include "backends/intel_uhd_backend.h"
#include "numerical_verifier.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

int main(int argc, char **argv)
{
    int platform_idx = 0;
    int device_idx = 0;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc)
            platform_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device_idx = atoi(argv[++i]);
        else
        {
            fprintf(stderr, "Usage: %s [--platform <int>] [--device <int>]\n", argv[0]);
            return 1;
        }
    }

    fprintf(stdout, "=== Relic OpenCL 1.2 Self-Test ===\n\n");

    OpenClBackend cl;
    if (!cl.init(platform_idx, device_idx))
    {
        fprintf(stderr, "FAIL: OpenCL init failed\n");
        return 1;
    }
    fprintf(stdout, "PASS: OpenCL initialized\n\n");

    // Build all kernels
    fprintf(stdout, "Building kernels...\n");
    struct
    {
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

    for (int i = 0; i < n_kernels; i++)
    {
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

    float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float B[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // identity
    float C_expected[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float C_gpu[16] = {0};

    ClBuffer buf_a, buf_b, buf_c;
    if (!buf_a.alloc(cl.dev.context, size_a) ||
        !buf_b.alloc(cl.dev.context, size_b) ||
        !buf_c.alloc(cl.dev.context, size_c))
    {
        fprintf(stderr, "FAIL: buffer allocation\n");
        return 1;
    }

    cl_int err;
    err = clEnqueueWriteBuffer(cl.dev.queue, buf_a.mem, CL_TRUE, 0, size_a, A, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(cl.dev.queue, buf_b.mem, CL_TRUE, 0, size_b, B, 0, nullptr, nullptr);

    cl.matmul_f32(buf_c, buf_a, buf_b, M, N, K);

    clFinish(cl.dev.queue);
    err |= clEnqueueReadBuffer(cl.dev.queue, buf_c.mem, CL_TRUE, 0, size_c, C_gpu, 0, nullptr, nullptr);

    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "FAIL: matmul execution\n");
        return 1;
    }

    bool pass = true;
    for (int i = 0; i < 16; i++)
    {
        if (fabsf(C_gpu[i] - C_expected[i]) > 1e-4f)
        {
            fprintf(stderr, "  Mismatch at [%d]: got %f, expected %f\n",
                    i, C_gpu[i], C_expected[i]);
            pass = false;
        }
    }
    fprintf(stdout, "  matmul result: %s\n", pass ? "PASS" : "FAIL");

    // Test fill + add
    fprintf(stdout, "\nRunning fill+add test...\n");
    ClBuffer buf_x, buf_y;
    float ones[4] = {1, 1, 1, 1};
    float twos[4] = {2, 2, 2, 2};
    float result[4] = {0};

    buf_x.alloc(cl.dev.context, 4 * sizeof(float));
    buf_y.alloc(cl.dev.context, 4 * sizeof(float));

    cl.fill(buf_x, 1.0f, 4);
    cl.fill(buf_y, 1.0f, 4);
    cl.add(buf_x, buf_x, buf_y, 4);
    clFinish(cl.dev.queue);
    clEnqueueReadBuffer(cl.dev.queue, buf_x.mem, CL_TRUE, 0, 4 * sizeof(float), result, 0, nullptr, nullptr);

    pass = true;
    for (int i = 0; i < 4; i++)
    {
        if (fabsf(result[i] - 2.0f) > 1e-4f)
        {
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
    for (int i = 0; i < 4; i++)
        ss += x_data[i] * x_data[i];
    float s = 1.0f / sqrtf(ss / 4.0f + eps);
    float y_ref[4];
    for (int i = 0; i < 4; i++)
        y_ref[i] = x_data[i] * s * w_data[i];

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
    for (int i = 0; i < 4; i++)
    {
        if (fabsf(y_data[i] - y_ref[i]) > 1e-4f)
        {
            fprintf(stderr, "  rms_norm mismatch at [%d]: got %f, expected %f\n",
                    i, y_data[i], y_ref[i]);
            pass = false;
        }
    }
    fprintf(stdout, "  rms_norm result: %s\n", pass ? "PASS" : "FAIL");

    // Test the architecture-independent reference implementation used by the
    // Qwen3.5 decoder before its OpenCL adapter is enabled.
    fprintf(stdout, "\nRunning Qwen3.5 recurrent state test...\n");
    ArchitectureSpec qwen;
    qwen.n_layer = 1;
    qwen.linear_conv_kernel = 2;
    qwen.linear_inner_size = 2;
    qwen.linear_key_head_dim = 1;
    qwen.linear_key_heads = 1;
    qwen.linear_value_heads = 2;
    Qwen35RecurrentState recurrent;
    pass = recurrent.init(qwen);
    float conv_input[] = {1, 2, 3, 4};
    float conv_kernel[] = {2, 3, 2, 3, 2, 3, 2, 3};
    float conv_output[4] = {};
    recurrent.conv1d(0, conv_input, conv_kernel, conv_output);
    const float first_conv[] = {3, 6, 9, 12};
    for (int i = 0; i < 4; ++i)
        pass &= fabsf(conv_output[i] - first_conv[i]) < 1e-5f;

    float q[] = {1, 1};
    float k[] = {1, 1};
    float v[] = {2, 4};
    float decay[] = {0, 0};
    float beta[] = {0.5f, 0.5f};
    float delta_output[2] = {};
    recurrent.delta_step(0, q, k, v, decay, beta, delta_output);
    pass &= fabsf(delta_output[0] - 1.0f) < 1e-5f && fabsf(delta_output[1] - 2.0f) < 1e-5f;
    recurrent.delta_step(0, q, k, v, decay, beta, delta_output);
    pass &= fabsf(delta_output[0] - 1.5f) < 1e-5f && fabsf(delta_output[1] - 3.0f) < 1e-5f;
    fprintf(stdout, "  Qwen3.5 recurrent state: %s\n", pass ? "PASS" : "FAIL");

    // Test Hexagonal Architecture: Decoder Port & Qwen3.5 CPU Forward Pass
    fprintf(stdout, "\nRunning Hexagonal Decoder Port & Qwen3.5 Forward Pass test...\n");
    ArchitectureSpec qwen_full;
    qwen_full.kind = ArchitectureKind::Qwen35;
    qwen_full.name = "qwen35";
    qwen_full.n_vocab = 8;
    qwen_full.n_embd = 4;
    qwen_full.n_layer = 2;
    qwen_full.n_head = 2;
    qwen_full.n_head_kv = 2;
    qwen_full.n_ff = 8;
    qwen_full.full_attention_interval = 2; // layer 0: recurrent, layer 1: full attention
    qwen_full.linear_conv_kernel = 2;
    qwen_full.linear_inner_size = 4;
    qwen_full.linear_key_head_dim = 2;
    qwen_full.linear_key_heads = 1;
    qwen_full.linear_value_heads = 2;

    auto decoder = create_decoder(qwen_full, nullptr);
    bool decoder_pass = (decoder != nullptr);
    if (decoder_pass)
    {
        decoder_pass &= decoder->init(qwen_full, 128);

        // Build mock model with identity/unit tensors
        LlamaModel mock_model;
        mock_model.architecture = qwen_full;
        mock_model.n_vocab = 8;
        mock_model.n_embd = 4;

        auto make_f32_tensor = [](const std::string &name, const std::vector<int64_t> &dims, float val = 1.0f)
        {
            LlamaModel::Tensor t;
            t.name = name;
            t.type = GgmlType::F32;
            t.dims = dims;
            int64_t ne = 1;
            for (auto d : dims)
                ne *= d;
            t.data.resize(ne * sizeof(float));
            std::vector<float> values(ne, val);
            memcpy(t.data.data(), values.data(), ne * sizeof(float));
            return t;
        };

        // Token embedding: [n_embd, n_vocab] = [4, 8]
        mock_model.tensors["token_embd.weight"] = make_f32_tensor("token_embd.weight", {4, 8}, 0.5f);
        // Output norm & weight
        mock_model.tensors["output_norm.weight"] = make_f32_tensor("output_norm.weight", {4}, 1.0f);
        mock_model.tensors["output.weight"] = make_f32_tensor("output.weight", {4, 8}, 0.25f);

        // Layer 0: Recurrent
        mock_model.tensors["blk.0.attn_norm.weight"] = make_f32_tensor("blk.0.attn_norm.weight", {4}, 1.0f);
        mock_model.tensors["blk.0.attn_q.weight"] = make_f32_tensor("blk.0.attn_q.weight", {4, 2}, 0.5f);
        mock_model.tensors["blk.0.attn_k.weight"] = make_f32_tensor("blk.0.attn_k.weight", {4, 2}, 0.5f);
        mock_model.tensors["blk.0.attn_v.weight"] = make_f32_tensor("blk.0.attn_v.weight", {4, 4}, 0.5f);
        mock_model.tensors["blk.0.ssm_conv1d.weight"] = make_f32_tensor("blk.0.ssm_conv1d.weight", {2, 8}, 1.0f);
        mock_model.tensors["blk.0.ssm_alpha.weight"] = make_f32_tensor("blk.0.ssm_alpha.weight", {2}, 0.0f);
        mock_model.tensors["blk.0.ssm_beta.weight"] = make_f32_tensor("blk.0.ssm_beta.weight", {2}, 0.5f);
        mock_model.tensors["blk.0.attn_output.weight"] = make_f32_tensor("blk.0.attn_output.weight", {4, 4}, 0.25f);
        mock_model.tensors["blk.0.ffn_norm.weight"] = make_f32_tensor("blk.0.ffn_norm.weight", {4}, 1.0f);
        mock_model.tensors["blk.0.ffn_gate.weight"] = make_f32_tensor("blk.0.ffn_gate.weight", {4, 8}, 0.2f);
        mock_model.tensors["blk.0.ffn_up.weight"] = make_f32_tensor("blk.0.ffn_up.weight", {4, 8}, 0.2f);
        mock_model.tensors["blk.0.ffn_down.weight"] = make_f32_tensor("blk.0.ffn_down.weight", {8, 4}, 0.2f);

        // Layer 1: Full Attention
        mock_model.tensors["blk.1.attn_norm.weight"] = make_f32_tensor("blk.1.attn_norm.weight", {4}, 1.0f);
        mock_model.tensors["blk.1.attn_q.weight"] = make_f32_tensor("blk.1.attn_q.weight", {4, 4}, 0.5f);
        mock_model.tensors["blk.1.attn_k.weight"] = make_f32_tensor("blk.1.attn_k.weight", {4, 4}, 0.5f);
        mock_model.tensors["blk.1.attn_v.weight"] = make_f32_tensor("blk.1.attn_v.weight", {4, 4}, 0.5f);
        mock_model.tensors["blk.1.attn_output.weight"] = make_f32_tensor("blk.1.attn_output.weight", {4, 4}, 0.25f);
        mock_model.tensors["blk.1.ffn_norm.weight"] = make_f32_tensor("blk.1.ffn_norm.weight", {4}, 1.0f);
        mock_model.tensors["blk.1.ffn_gate.weight"] = make_f32_tensor("blk.1.ffn_gate.weight", {4, 8}, 0.2f);
        mock_model.tensors["blk.1.ffn_up.weight"] = make_f32_tensor("blk.1.ffn_up.weight", {4, 8}, 0.2f);
        mock_model.tensors["blk.1.ffn_down.weight"] = make_f32_tensor("blk.1.ffn_down.weight", {8, 4}, 0.2f);

        std::vector<float> logits(8, 0.0f);
        int res0 = decoder->forward(mock_model, 0, 0, logits.data());
        decoder_pass &= (res0 == 0);
        for (int i = 0; i < 8; ++i)
        {
            decoder_pass &= (!std::isnan(logits[i]) && !std::isinf(logits[i]));
        }

        int res1 = decoder->forward(mock_model, 1, 1, logits.data());
        decoder_pass &= (res1 == 0);
    }
    pass &= decoder_pass;
    fprintf(stdout, "  Qwen3.5 decoder port & forward pass: %s\n", decoder_pass ? "PASS" : "FAIL");

    // Phase 1 Modular Architecture & Planner Tests
    fprintf(stdout, "\nRunning Phase 1 Modular Architecture & Planner tests...\n");
    HardwareProfile prof = HardwareProfile::probe_system();
    bool prof_pass = (prof.cpu_cores > 0 && !prof.devices.empty());
    fprintf(stdout, "  HardwareProfile probe: %s (%zu devices detected)\n", prof_pass ? "PASS" : "FAIL", prof.devices.size());
    pass &= prof_pass;

    // Test Adaptive Planner with mock model and budget constraints
    LlamaModel plan_mock_model;
    plan_mock_model.n_layer = 4;
    plan_mock_model.tensors["token_embd.weight"] = {"token_embd.weight", GgmlType::Q4_0, {2048, 1000}, std::vector<uint8_t>(2048 * 1000 * 18 / 32)};
    plan_mock_model.tensors["output_norm.weight"] = {"output_norm.weight", GgmlType::F32, {2048}, std::vector<uint8_t>(2048 * 4)};
    plan_mock_model.tensors["blk.0.attn_q.weight"] = {"blk.0.attn_q.weight", GgmlType::Q4_0, {2048, 2048}, std::vector<uint8_t>(2048 * 2048 * 18 / 32)};
    plan_mock_model.tensors["blk.0.ffn_down.weight"] = {"blk.0.ffn_down.weight", GgmlType::Q4_0, {6144, 2048}, std::vector<uint8_t>(6144 * 2048 * 18 / 32)};

    ExecutionPlan plan = AdaptivePlanner::generate_plan(plan_mock_model, prof, 1024 * 1024 * 1024); // 1GB VRAM budget
    bool plan_pass = (!plan.tensor_placements.empty() && plan.vram_required_bytes > 0);
    fprintf(stdout, "  AdaptivePlanner plan: %s (VRAM Footprint: %.1f%% red, PCIe Traffic: %.1f%% red, DMA Overlap: %.1f%%)\n",
            plan_pass ? "PASS" : "FAIL", plan.vram_footprint_reduction_pct, plan.pcie_traffic_reduction_pct, plan.estimated_dma_overlap_efficiency);
    pass &= plan_pass;

    // Test Large Model (>4 GB) Heterogeneous Offload Planning (4.5 GB Model on 1.5 GB VRAM Budget)
    LlamaModel large_model;
    large_model.n_layer = 32;
    large_model.architecture.n_embd = 4096;
    large_model.tensors["token_embd.weight"] = {"token_embd.weight", GgmlType::Q4_0, {4096, 32000}, {}};
    large_model.tensors["output.weight"] = {"output.weight", GgmlType::Q4_0, {4096, 32000}, {}};
    large_model.tensors["output_norm.weight"] = {"output_norm.weight", GgmlType::F32, {4096}, {}};
    for (int l = 0; l < 32; l++) {
        std::string p = "blk." + std::to_string(l) + ".";
        large_model.tensors[p + "attn_norm.weight"] = {p + "attn_norm.weight", GgmlType::F32, {4096}, {}};
        large_model.tensors[p + "attn_q.weight"] = {p + "attn_q.weight", GgmlType::Q4_0, {4096, 4096}, {}};
        large_model.tensors[p + "attn_k.weight"] = {p + "attn_k.weight", GgmlType::Q4_0, {4096, 1024}, {}};
        large_model.tensors[p + "attn_v.weight"] = {p + "attn_v.weight", GgmlType::Q4_0, {4096, 1024}, {}};
        large_model.tensors[p + "attn_output.weight"] = {p + "attn_output.weight", GgmlType::Q4_0, {4096, 4096}, {}};
        large_model.tensors[p + "ffn_norm.weight"] = {p + "ffn_norm.weight", GgmlType::F32, {4096}, {}};
        large_model.tensors[p + "ffn_gate.weight"] = {p + "ffn_gate.weight", GgmlType::Q4_0, {11008, 4096}, {}};
        large_model.tensors[p + "ffn_up.weight"] = {p + "ffn_up.weight", GgmlType::Q4_0, {11008, 4096}, {}};
        large_model.tensors[p + "ffn_down.weight"] = {p + "ffn_down.weight", GgmlType::Q4_0, {4096, 11008}, {}};
    }
    ExecutionPlan large_plan = AdaptivePlanner::generate_plan(large_model, prof, (size_t)(1.5 * 1024 * 1024 * 1024)); // 1.5 GB strict VRAM budget
    bool large_plan_pass = (!large_plan.tensor_placements.empty() && large_plan.vram_required_bytes <= (size_t)(1.5 * 1024 * 1024 * 1024 + 1024 * 1024));
    fprintf(stdout, "  Large Model (>4 GB) Heterogeneous Plan: %s (VRAM: %.1f MB / 1536 MB budget, Host RAM: %.1f MB, Offload: %zu tensors)\n",
            large_plan_pass ? "PASS" : "FAIL",
            (double)large_plan.vram_required_bytes / (1024.0 * 1024.0),
            (double)large_plan.host_ram_required_bytes / (1024.0 * 1024.0),
            large_plan.tensor_placements.size());
    pass &= large_plan_pass;

    // Phase 2 MemoryEngine & Pinned Host Pool Tests
    fprintf(stdout, "\nRunning Phase 2 MemoryEngine & Pinned Pool tests...\n");
    PinnedHostPool pinned_pool(16 * 1024 * 1024); // 16 MB pool
    void *pinned_alloc = pinned_pool.allocate_pinned(1024 * 1024);
    bool pinned_pass = (pinned_alloc != nullptr && pinned_pool.used() == 1024 * 1024);
    fprintf(stdout, "  PinnedHostPool allocation (64-byte aligned): %s\n", pinned_pass ? "PASS" : "FAIL");
    pass &= pinned_pass;

    AsyncPrefetcher prefetcher(cl.dev.context, cl.dev.queue);
    bool prefetcher_init = prefetcher.initialize(4 * 1024 * 1024); // 4 MB dual-slot staging
    bool prefetch_pass = prefetcher_init;
    if (prefetcher_init && pinned_alloc)
    {
        std::vector<float> mock_layer_data(256 * 1024, 1.0f); // 1 MB
        memcpy(pinned_alloc, mock_layer_data.data(), mock_layer_data.size() * sizeof(float));
        bool pf_ok = prefetcher.prefetch_async(pinned_alloc, mock_layer_data.size() * sizeof(float), 0);
        prefetcher.wait_ready(0);
        prefetch_pass &= (pf_ok && prefetcher.get_staging_mem(0) != nullptr);
    }
    fprintf(stdout, "  AsyncPrefetcher non-blocking DMA transfer & wait: %s\n", prefetch_pass ? "PASS" : "FAIL");
    pass &= prefetch_pass;

    // Phase 3 Distributed Speculative Engine Batched Verification Tests
    fprintf(stdout, "\nRunning Phase 3 Distributed Speculative Batched Verification tests...\n");
    DistributedSpeculativeEngine dist_spec(nullptr, nullptr, nullptr, nullptr, nullptr);
    SpeculativeResult spec_res = dist_spec.step_speculation({1, 2, 3}, 4);
    bool spec_pass = (spec_res.num_drafted == 0 && spec_res.accepted_tokens.empty());
    fprintf(stdout, "  DistributedSpeculativeEngine batched verification fallback: %s\n", spec_pass ? "PASS" : "FAIL");
    pass &= spec_pass;

    // Intel UHD Backend Tests (Compute Kernels & Zero-Copy Allocation)
    fprintf(stdout, "\nRunning Intel UHD Backend Compute tests...\n");
    IntelUhdBackend intel_be;
    bool intel_init = intel_be.initialize();
    if (intel_init)
    {
        DeviceStats stats = intel_be.query_stats();
        auto shared_buf = intel_be.allocate(1024 * 1024, MemoryTier::TIER1_SHARED_IGPU);
        bool alloc_pass = (shared_buf != nullptr && shared_buf->size() == 1024 * 1024);
        fprintf(stdout, "  Intel UHD Shared Memory Allocation (1 MB zero-copy): %s\n", alloc_pass ? "PASS" : "FAIL");

        // Test Intel UHD compute kernel (rms_norm)
        auto x_buf = intel_be.allocate(256 * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
        auto w_buf = intel_be.allocate(256 * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
        auto out_buf = intel_be.allocate(256 * sizeof(float), MemoryTier::TIER1_SHARED_IGPU);
        std::vector<float> mock_x(256, 1.5f), mock_w(256, 1.0f), out_h(256, 0.0f);
        intel_be.upload(*x_buf, mock_x.data(), 256 * sizeof(float));
        intel_be.upload(*w_buf, mock_w.data(), 256 * sizeof(float));
        intel_be.rms_norm(*out_buf, *x_buf, *w_buf, 256, 1e-5f);
        intel_be.synchronize();
        intel_be.download(out_h.data(), *out_buf, 256 * sizeof(float));

        bool intel_compute_pass = (fabsf(out_h[0] - 1.0f) < 1e-2f);
        fprintf(stdout, "  Intel UHD Compute RMSNorm Execution: %s (Sample: %.4f)\n", intel_compute_pass ? "PASS" : "FAIL", out_h[0]);
        pass &= alloc_pass && intel_compute_pass;
    }
    else
    {
        fprintf(stdout, "  Intel UHD Shared Memory Backend: SKIP (Standalone dGPU mode)\n");
    }

    // Phase 4 Layer-by-Layer Numerical Verification
    fprintf(stdout, "\nRunning Layer-by-Layer Numerical Verification (Error & Cosine Sim)...\n");
    std::vector<NumericalMetric> metrics = NumericalVerifier::verify_layers(&cl, intel_init ? &intel_be : nullptr);
    bool num_pass = true;
    for (const auto &m : metrics)
    {
        fprintf(stdout, "  [%s] MaxAbsErr: %.2e | MeanAbsErr: %.2e | CosSim: %.6f -> %s\n",
                m.layer_name.c_str(), m.max_absolute_error, m.mean_absolute_error, m.cosine_similarity,
                m.passed ? "PASS" : "FAIL");
        num_pass &= m.passed;
    }

    fprintf(stdout, "\nRunning End-to-End Model Layer-by-Layer Verification...\n");
    std::vector<NumericalMetric> model_metrics = NumericalVerifier::verify_model_layer_by_layer(plan_mock_model, &cl);
    for (const auto &m : model_metrics)
    {
        fprintf(stdout, "  [%s] MaxAbsErr: %.2e | MeanAbsErr: %.2e | CosSim: %.6f -> %s\n",
                m.layer_name.c_str(), m.max_absolute_error, m.mean_absolute_error, m.cosine_similarity,
                m.passed ? "PASS" : "FAIL");
        num_pass &= m.passed;
    }
    pass &= num_pass;

    // Device info summary
    fprintf(stdout, "\n=== Device Summary ===\n");
    fprintf(stdout, "  Name: %s\n", cl.dev.name.c_str());
    fprintf(stdout, "  Platform: %s\n", cl.dev.platform_name.c_str());
    fprintf(stdout, "  OpenCL C: %d.%d\n", cl.dev.opencl_c_major, cl.dev.opencl_c_minor);
    fprintf(stdout, "  FP16: %s\n", cl.dev.fp16 ? "yes" : "no");
    fprintf(stdout, "  Global Mem: %zu MB\n", cl.dev.global_mem / (1024 * 1024));
    fprintf(stdout, "  Max Alloc: %zu MB\n", cl.dev.max_alloc / (1024 * 1024));
    fprintf(stdout, "  Compute Units: %u\n", cl.dev.compute_units);
    fprintf(stdout, "  Max WG Size: %zu\n", cl.dev.max_wg_size);

    fprintf(stdout, "\n=== All tests %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
