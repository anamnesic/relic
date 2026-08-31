// Self-test: verify OpenCL 1.2 init and kernel execution on Caicos
#include "opencl_backend.h"
#include "qwen35_state.h"
#include "decoder.h"
#include "planner/adaptive_planner.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    int platform_idx = 0;
    int device_idx = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc) platform_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device_idx = atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [--platform <int>] [--device <int>]\n", argv[0]);
            return 1;
        }
    }

    fprintf(stdout, "=== Relic OpenCL 1.2 Self-Test ===\n\n");

    OpenClBackend cl;
    if (!cl.init(platform_idx, device_idx)) {
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
    for (int i = 0; i < 4; ++i) pass &= fabsf(conv_output[i] - first_conv[i]) < 1e-5f;

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
    if (decoder_pass) {
        decoder_pass &= decoder->init(qwen_full, 128);

        // Build mock model with identity/unit tensors
        LlamaModel mock_model;
        mock_model.architecture = qwen_full;
        mock_model.n_vocab = 8;
        mock_model.n_embd = 4;

        auto make_f32_tensor = [](const std::string &name, const std::vector<int64_t> &dims, float val = 1.0f) {
            LlamaModel::Tensor t;
            t.name = name;
            t.type = GgmlType::F32;
            t.dims = dims;
            int64_t ne = 1;
            for (auto d : dims) ne *= d;
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
        for (int i = 0; i < 8; ++i) {
            decoder_pass &= (!std::isnan(logits[i]) && !std::isinf(logits[i]));
        }

        int res1 = decoder->forward(mock_model, 1, 1, logits.data());
        decoder_pass &= (res1 == 0);
    }
    pass &= decoder_pass;
    fprintf(stdout, "  Qwen3.5 decoder port & forward pass: %s\n", decoder_pass ? "PASS" : "FAIL");

    // Phase 1 Modular Architecture Tests
    fprintf(stdout, "\nRunning Phase 1 Modular Architecture tests...\n");
    HardwareProfile prof = HardwareProfile::probe_system();
    bool prof_pass = (prof.cpu_cores > 0 && !prof.devices.empty());
    fprintf(stdout, "  HardwareProfile probe: %s (%zu devices detected)\n", prof_pass ? "PASS" : "FAIL", prof.devices.size());
    pass &= prof_pass;

    // Test Adaptive Planner with mock model
    LlamaModel plan_mock_model;
    plan_mock_model.n_layer = 4;
    plan_mock_model.tensors["token_embd.weight"] = { "token_embd.weight", GgmlType::Q4_0, {2048, 1000}, std::vector<uint8_t>(2048 * 1000 * 18 / 32) };
    plan_mock_model.tensors["output_norm.weight"] = { "output_norm.weight", GgmlType::F32, {2048}, std::vector<uint8_t>(2048 * 4) };
    plan_mock_model.tensors["blk.0.attn_q.weight"] = { "blk.0.attn_q.weight", GgmlType::Q4_0, {2048, 2048}, std::vector<uint8_t>(2048 * 2048 * 18 / 32) };
    plan_mock_model.tensors["blk.0.ffn_down.weight"] = { "blk.0.ffn_down.weight", GgmlType::Q4_0, {6144, 2048}, std::vector<uint8_t>(6144 * 2048 * 18 / 32) };

    ExecutionPlan plan = AdaptivePlanner::generate_plan(plan_mock_model, prof, 1024 * 1024 * 1024); // 1GB VRAM budget
    bool plan_pass = (!plan.tensor_placements.empty() && plan.vram_required_bytes > 0);
    fprintf(stdout, "  AdaptivePlanner execution plan generation: %s (%zu placements)\n", plan_pass ? "PASS" : "FAIL", plan.tensor_placements.size());
    pass &= plan_pass;

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
