// Relic - High-Performance OpenCL 1.2 Kernels for LLM Inference
// Pure OpenCL C 1.2 (No FP16 extensions required, supports all GPUs)

//------------------------------------------------------------------------------
// FP16 to FP32 bitcast helper (IEEE 754 half-precision decode)
//------------------------------------------------------------------------------
inline float fp16_to_fp32(ushort h) {
    uint sign = ((uint)h >> 15) & 1;
    uint exp  = ((uint)h >> 10) & 0x1f;
    uint mant = (uint)h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3ff;
    } else if (exp == 31) {
        return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    exp = exp + (127 - 15);
    uint u = (sign << 31) | (exp << 23) | (mant << 13);
    return as_float(u);
}

//------------------------------------------------------------------------------
// GPU Embedding Lookup (128-bit SIMD in VRAM)
//------------------------------------------------------------------------------
kernel void embed_lookup_q4_0(
    global float *hidden,
    global const uchar *embd_table,
    int token_id,
    int n_embd
) {
    int i = get_global_id(0); // Vector index [0 .. n_embd/4 - 1]
    int n_blocks = n_embd / 32;
    int blk_idx = i / 8;
    int in_blk_vec = i % 8; // [0 .. 7]

    global const uchar *row_ptr = embd_table + (size_t)token_id * (size_t)(n_blocks * 18);
    global const uchar *b_blk = row_ptr + (size_t)blk_idx * 18;
    float d = fp16_to_fp32((ushort)b_blk[0] | ((ushort)b_blk[1] << 8));
    global const uchar *qs = b_blk + 2;

    if (in_blk_vec < 4) {
        uchar4 qb = vload4(in_blk_vec, qs);
        float4 v_lo = (convert_float4(qb & (uchar4)0x0F) - (float4)8.0f) * d;
        vstore4(v_lo, i, hidden);
    } else {
        uchar4 qb = vload4(in_blk_vec - 4, qs);
        float4 v_hi = (convert_float4(qb >> (uchar4)4) - (float4)8.0f) * d;
        vstore4(v_hi, i, hidden);
    }
}

//------------------------------------------------------------------------------
// Vectorized RMS Norm (128-bit SIMD)
//------------------------------------------------------------------------------
kernel void rms_norm_f32(
    global float *out,
    global const float *x,
    global const float *weight,
    int n,
    float eps
) {
    int row = get_global_id(0);
    global const float *rx = x + (size_t)row * n;
    global float *rout = out + (size_t)row * n;

    float4 ss4 = (float4)(0.0f);
    int n4 = n / 4;
    for (int i = 0; i < n4; i++) {
        float4 v = vload4(i, rx);
        ss4 += v * v;
    }
    float ss = ss4.x + ss4.y + ss4.z + ss4.w;
    float s = rsqrt(ss / (float)n + eps);

    for (int i = 0; i < n4; i++) {
        float4 v = vload4(i, rx);
        float4 w = vload4(i, weight);
        vstore4(v * s * w, i, rout);
    }
}

//------------------------------------------------------------------------------
// FUSED: Residual Add + In-Place RMS Norm (128-bit SIMD)
//------------------------------------------------------------------------------
kernel void add_rms_norm_f32(
    global float *residual,      // In/Out: residual += branch
    global const float *branch,  // In: branch activation
    global const float *weight,  // In: norm weight
    global float *norm_out,      // Out: normalized activation
    int n,
    float eps
) {
    int row = get_global_id(0);
    global float *r = residual + (size_t)row * n;
    global const float *b = branch + (size_t)row * n;
    global float *out = norm_out + (size_t)row * n;

    float4 ss4 = (float4)(0.0f);
    int n4 = n / 4;
    for (int i = 0; i < n4; i++) {
        float4 r_val = vload4(i, r) + vload4(i, b);
        vstore4(r_val, i, r);
        ss4 += r_val * r_val;
    }
    float ss = ss4.x + ss4.y + ss4.z + ss4.w;
    float s = rsqrt(ss / (float)n + eps);

    for (int i = 0; i < n4; i++) {
        float4 r_val = vload4(i, r);
        float4 w = vload4(i, weight);
        vstore4(r_val * s * w, i, out);
    }
}

//------------------------------------------------------------------------------
// Matrix Multiply (A: MxK, B: KxN, C: MxN)
//------------------------------------------------------------------------------
kernel void matmul_f32(
    global const float *a,
    global const float *b,
    global float *dst,
    int M,
    int N,
    int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    global const float *a_row = a + (size_t)row * K;
    global const float *b_col = b + col;
    for (int i = 0; i < K; i++) {
        sum += a_row[i] * b_col[(size_t)i * N];
    }
    dst[(size_t)row * N + col] = sum;
}

//------------------------------------------------------------------------------
// Matrix Multiply NT (A: MxK, B: NxK, C: MxN)
//------------------------------------------------------------------------------
kernel void matmul_f32_nt(
    global const float *a,
    global const float *b,
    global float *dst,
    int M,
    int N,
    int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    global const float *a_row = a + (size_t)row * K;
    global const float *b_row = b + (size_t)col * K;
    for (int i = 0; i < K; i++) {
        sum += a_row[i] * b_row[i];
    }
    dst[(size_t)row * N + col] = sum;
}

//------------------------------------------------------------------------------
// GEMV F32 NT (Warp-32 optimized)
//------------------------------------------------------------------------------
kernel void gemv_f32_nt(
    global const float *a,
    global const float *b,
    global float *dst,
    int N,
    int K
) {
    local float l_sum[32];
    int col = get_group_id(0);
    if (col >= N) return;
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    global const float *b_row = b + (size_t)col * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += wg_size) {
        sum += a[i] * b_row[i];
    }

    l_sum[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16) l_sum[tid] += l_sum[tid + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)  l_sum[tid] += l_sum[tid + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)  l_sum[tid] += l_sum[tid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)  l_sum[tid] += l_sum[tid + 2];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        dst[col] = l_sum[0] + l_sum[1];
    }
}

//------------------------------------------------------------------------------
// MULTI-ROW 4x GEMV Q8_0: Warp-Synchronous Execution
//------------------------------------------------------------------------------
kernel void gemv_q8_0(
    global const float *a,
    global const uchar *b,
    global float *dst,
    int N,
    int K
) {
    local float l_sum0[32];
    local float l_sum1[32];
    local float l_sum2[32];
    local float l_sum3[32];

    int row0 = get_group_id(0) * 4;
    int row1 = row0 + 1;
    int row2 = row0 + 2;
    int row3 = row0 + 3;
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    int n_blocks = K / 32;
    global const uchar *row_ptr0 = b + (size_t)row0 * (size_t)(n_blocks * 34);
    global const uchar *row_ptr1 = (row1 < N) ? (b + (size_t)row1 * (size_t)(n_blocks * 34)) : row_ptr0;
    global const uchar *row_ptr2 = (row2 < N) ? (b + (size_t)row2 * (size_t)(n_blocks * 34)) : row_ptr0;
    global const uchar *row_ptr3 = (row3 < N) ? (b + (size_t)row3 * (size_t)(n_blocks * 34)) : row_ptr0;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int blk = tid; blk < n_blocks; blk += wg_size) {
        global const float *a_blk = a + blk * 32;

        global const uchar *b_blk0 = row_ptr0 + (size_t)blk * 34;
        float d0 = fp16_to_fp32((ushort)b_blk0[0] | ((ushort)b_blk0[1] << 8));
        global const char *qs0 = (global const char *)(b_blk0 + 2);

        global const uchar *b_blk1 = row_ptr1 + (size_t)blk * 34;
        float d1 = fp16_to_fp32((ushort)b_blk1[0] | ((ushort)b_blk1[1] << 8));
        global const char *qs1 = (global const char *)(b_blk1 + 2);

        global const uchar *b_blk2 = row_ptr2 + (size_t)blk * 34;
        float d2 = fp16_to_fp32((ushort)b_blk2[0] | ((ushort)b_blk2[1] << 8));
        global const char *qs2 = (global const char *)(b_blk2 + 2);

        global const uchar *b_blk3 = row_ptr3 + (size_t)blk * 34;
        float d3 = fp16_to_fp32((ushort)b_blk3[0] | ((ushort)b_blk3[1] << 8));
        global const char *qs3 = (global const char *)(b_blk3 + 2);

        for (int i = 0; i < 8; i++) {
            float4 a_v = vload4(i, a_blk);

            char4 q0 = vload4(i, qs0);
            char4 q1 = vload4(i, qs1);
            char4 q2 = vload4(i, qs2);
            char4 q3 = vload4(i, qs3);

            sum0 += dot(a_v, convert_float4(q0) * d0);
            sum1 += dot(a_v, convert_float4(q1) * d1);
            sum2 += dot(a_v, convert_float4(q2) * d2);
            sum3 += dot(a_v, convert_float4(q3) * d3);
        }
    }

    l_sum0[tid] = sum0; l_sum1[tid] = sum1; l_sum2[tid] = sum2; l_sum3[tid] = sum3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16) {
        l_sum0[tid] += l_sum0[tid + 16]; l_sum1[tid] += l_sum1[tid + 16];
        l_sum2[tid] += l_sum2[tid + 16]; l_sum3[tid] += l_sum3[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8) {
        l_sum0[tid] += l_sum0[tid + 8]; l_sum1[tid] += l_sum1[tid + 8];
        l_sum2[tid] += l_sum2[tid + 8]; l_sum3[tid] += l_sum3[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4) {
        l_sum0[tid] += l_sum0[tid + 4]; l_sum1[tid] += l_sum1[tid + 4];
        l_sum2[tid] += l_sum2[tid + 4]; l_sum3[tid] += l_sum3[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2) {
        l_sum0[tid] += l_sum0[tid + 2]; l_sum1[tid] += l_sum1[tid + 2];
        l_sum2[tid] += l_sum2[tid + 2]; l_sum3[tid] += l_sum3[tid + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        dst[row0] = l_sum0[0] + l_sum0[1];
        if (row1 < N) dst[row1] = l_sum1[0] + l_sum1[1];
        if (row2 < N) dst[row2] = l_sum2[0] + l_sum2[1];
        if (row3 < N) dst[row3] = l_sum3[0] + l_sum3[1];
    }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// MULTI-ROW 8x GEMV Q4_0: 2-Warp Tiled Execution with Dynamic Local Activation Cache
//------------------------------------------------------------------------------
kernel void gemv_q4_0(
    global const float *a,
    global const uchar *b,
    global float *dst,
    int N,
    int K,
    local float *l_a
) {
    local float l_sum0[2][32];
    local float l_sum1[2][32];
    local float l_sum2[2][32];
    local float l_sum3[2][32];

    int tid = get_local_id(0);
    int wg_size = get_local_size(0);
    int warp_id = tid / 32;
    int lane = tid % 32;

    int base_row = get_group_id(0) * 8 + warp_id * 4;
    int row0 = base_row;
    int row1 = base_row + 1;
    int row2 = base_row + 2;
    int row3 = base_row + 3;

    // Cache vector 'a' into on-chip shared memory dynamically sized to K
    for (int i = tid; i < K; i += wg_size) {
        l_a[i] = a[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_blocks = K / 32;
    global const uchar *row_ptr0 = b + (size_t)row0 * (size_t)(n_blocks * 18);
    global const uchar *row_ptr1 = (row1 < N) ? (b + (size_t)row1 * (size_t)(n_blocks * 18)) : row_ptr0;
    global const uchar *row_ptr2 = (row2 < N) ? (b + (size_t)row2 * (size_t)(n_blocks * 18)) : row_ptr0;
    global const uchar *row_ptr3 = (row3 < N) ? (b + (size_t)row3 * (size_t)(n_blocks * 18)) : row_ptr0;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    for (int blk = lane; blk < n_blocks; blk += 32) {
        global const uchar *b_blk0 = row_ptr0 + (size_t)blk * 18;
        float d0 = fp16_to_fp32((ushort)b_blk0[0] | ((ushort)b_blk0[1] << 8));
        global const uchar *qs0 = b_blk0 + 2;

        global const uchar *b_blk1 = row_ptr1 + (size_t)blk * 18;
        float d1 = fp16_to_fp32((ushort)b_blk1[0] | ((ushort)b_blk1[1] << 8));
        global const uchar *qs1 = b_blk1 + 2;

        global const uchar *b_blk2 = row_ptr2 + (size_t)blk * 18;
        float d2 = fp16_to_fp32((ushort)b_blk2[0] | ((ushort)b_blk2[1] << 8));
        global const uchar *qs2 = b_blk2 + 2;

        global const uchar *b_blk3 = row_ptr3 + (size_t)blk * 18;
        float d3 = fp16_to_fp32((ushort)b_blk3[0] | ((ushort)b_blk3[1] << 8));
        global const uchar *qs3 = b_blk3 + 2;

        // Issue 128-bit vector loads upfront into private registers
        uchar16 qb0 = vload16(0, qs0);
        uchar16 qb1 = vload16(0, qs1);
        uchar16 qb2 = vload16(0, qs2);
        uchar16 qb3 = vload16(0, qs3);

        local const float *a_blk = l_a + blk * 32;
        float4 a0 = vload4(0, a_blk);
        float4 a1 = vload4(1, a_blk);
        float4 a2 = vload4(2, a_blk);
        float4 a3 = vload4(3, a_blk);
        float4 a4 = vload4(4, a_blk);
        float4 a5 = vload4(5, a_blk);
        float4 a6 = vload4(6, a_blk);
        float4 a7 = vload4(7, a_blk);

        float sum_a = (a0.x + a0.y + a0.z + a0.w) +
                      (a1.x + a1.y + a1.z + a1.w) +
                      (a2.x + a2.y + a2.z + a2.w) +
                      (a3.x + a3.y + a3.z + a3.w) +
                      (a4.x + a4.y + a4.z + a4.w) +
                      (a5.x + a5.y + a5.z + a5.w) +
                      (a6.x + a6.y + a6.z + a6.w) +
                      (a7.x + a7.y + a7.z + a7.w);

        // Row 0
        uchar4 q0_0 = qb0.s0123, q0_1 = qb0.s4567, q0_2 = qb0.s89ab, q0_3 = qb0.scdef;
        float block_acc0 = dot(a0, convert_float4(q0_0 & (uchar4)0x0F)) + dot(a4, convert_float4(q0_0 >> (uchar4)4))
                         + dot(a1, convert_float4(q0_1 & (uchar4)0x0F)) + dot(a5, convert_float4(q0_1 >> (uchar4)4))
                         + dot(a2, convert_float4(q0_2 & (uchar4)0x0F)) + dot(a6, convert_float4(q0_2 >> (uchar4)4))
                         + dot(a3, convert_float4(q0_3 & (uchar4)0x0F)) + dot(a7, convert_float4(q0_3 >> (uchar4)4));

        // Row 1
        uchar4 q1_0 = qb1.s0123, q1_1 = qb1.s4567, q1_2 = qb1.s89ab, q1_3 = qb1.scdef;
        float block_acc1 = dot(a0, convert_float4(q1_0 & (uchar4)0x0F)) + dot(a4, convert_float4(q1_0 >> (uchar4)4))
                         + dot(a1, convert_float4(q1_1 & (uchar4)0x0F)) + dot(a5, convert_float4(q1_1 >> (uchar4)4))
                         + dot(a2, convert_float4(q1_2 & (uchar4)0x0F)) + dot(a6, convert_float4(q1_2 >> (uchar4)4))
                         + dot(a3, convert_float4(q1_3 & (uchar4)0x0F)) + dot(a7, convert_float4(q1_3 >> (uchar4)4));

        // Row 2
        uchar4 q2_0 = qb2.s0123, q2_1 = qb2.s4567, q2_2 = qb2.s89ab, q2_3 = qb2.scdef;
        float block_acc2 = dot(a0, convert_float4(q2_0 & (uchar4)0x0F)) + dot(a4, convert_float4(q2_0 >> (uchar4)4))
                         + dot(a1, convert_float4(q2_1 & (uchar4)0x0F)) + dot(a5, convert_float4(q2_1 >> (uchar4)4))
                         + dot(a2, convert_float4(q2_2 & (uchar4)0x0F)) + dot(a6, convert_float4(q2_2 >> (uchar4)4))
                         + dot(a3, convert_float4(q2_3 & (uchar4)0x0F)) + dot(a7, convert_float4(q2_3 >> (uchar4)4));

        // Row 3
        uchar4 q3_0 = qb3.s0123, q3_1 = qb3.s4567, q3_2 = qb3.s89ab, q3_3 = qb3.scdef;
        float block_acc3 = dot(a0, convert_float4(q3_0 & (uchar4)0x0F)) + dot(a4, convert_float4(q3_0 >> (uchar4)4))
                         + dot(a1, convert_float4(q3_1 & (uchar4)0x0F)) + dot(a5, convert_float4(q3_1 >> (uchar4)4))
                         + dot(a2, convert_float4(q3_2 & (uchar4)0x0F)) + dot(a6, convert_float4(q3_2 >> (uchar4)4))
                         + dot(a3, convert_float4(q3_3 & (uchar4)0x0F)) + dot(a7, convert_float4(q3_3 >> (uchar4)4));

        float bias = 8.0f * sum_a;
        sum0 += (block_acc0 - bias) * d0;
        sum1 += (block_acc1 - bias) * d1;
        sum2 += (block_acc2 - bias) * d2;
        sum3 += (block_acc3 - bias) * d3;
    }

    l_sum0[warp_id][lane] = sum0;
    l_sum1[warp_id][lane] = sum1;
    l_sum2[warp_id][lane] = sum2;
    l_sum3[warp_id][lane] = sum3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane < 16) {
        l_sum0[warp_id][lane] += l_sum0[warp_id][lane + 16];
        l_sum1[warp_id][lane] += l_sum1[warp_id][lane + 16];
        l_sum2[warp_id][lane] += l_sum2[warp_id][lane + 16];
        l_sum3[warp_id][lane] += l_sum3[warp_id][lane + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < 8) {
        l_sum0[warp_id][lane] += l_sum0[warp_id][lane + 8];
        l_sum1[warp_id][lane] += l_sum1[warp_id][lane + 8];
        l_sum2[warp_id][lane] += l_sum2[warp_id][lane + 8];
        l_sum3[warp_id][lane] += l_sum3[warp_id][lane + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < 4) {
        l_sum0[warp_id][lane] += l_sum0[warp_id][lane + 4];
        l_sum1[warp_id][lane] += l_sum1[warp_id][lane + 4];
        l_sum2[warp_id][lane] += l_sum2[warp_id][lane + 4];
        l_sum3[warp_id][lane] += l_sum3[warp_id][lane + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < 2) {
        l_sum0[warp_id][lane] += l_sum0[warp_id][lane + 2];
        l_sum1[warp_id][lane] += l_sum1[warp_id][lane + 2];
        l_sum2[warp_id][lane] += l_sum2[warp_id][lane + 2];
        l_sum3[warp_id][lane] += l_sum3[warp_id][lane + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane == 0) {
        if (row0 < N) dst[row0] = l_sum0[warp_id][0] + l_sum0[warp_id][1];
        if (row1 < N) dst[row1] = l_sum1[warp_id][0] + l_sum1[warp_id][1];
        if (row2 < N) dst[row2] = l_sum2[warp_id][0] + l_sum2[warp_id][1];
        if (row3 < N) dst[row3] = l_sum3[warp_id][0] + l_sum3[warp_id][1];
    }
}

//------------------------------------------------------------------------------
// FUSED FFN: Multi-Row 8x GEMV (Gate + Up) + SiLU + Mul in 1 Kernel Launch
//------------------------------------------------------------------------------
kernel void gemv_q4_0_ffn_swiglu(
    global const float *a,          // [K = 2048]
    global const uchar *b_gate,     // [N = 6144, K = 2048]
    global const uchar *b_up,       // [N = 6144, K = 2048]
    global float *dst,              // [N = 6144]
    int N,
    int K,
    local float *l_a
) {
    local float l_gate[8][64];
    local float l_up[8][64];

    int base_row = get_group_id(0) * 8;
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    // Dynamically sized shared memory buffer
    for (int i = tid; i < K; i += wg_size) {
        l_a[i] = a[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_blocks = K / 32;
    global const uchar *gate_ptrs[8];
    global const uchar *up_ptrs[8];
    for (int r = 0; r < 8; r++) {
        int row = base_row + r;
        gate_ptrs[r] = (row < N) ? (b_gate + (size_t)row * (size_t)(n_blocks * 18)) : b_gate;
        up_ptrs[r]   = (row < N) ? (b_up   + (size_t)row * (size_t)(n_blocks * 18)) : b_up;
    }

    float sum_gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_up[8]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = tid; blk < n_blocks; blk += wg_size) {
        local const float *a_blk = l_a + blk * 32;

        float d_gate[8], d_up[8];
        uchar16 qg[8], qu[8];

        for (int r = 0; r < 8; r++) {
            global const uchar *bg = gate_ptrs[r] + (size_t)blk * 18;
            d_gate[r] = fp16_to_fp32((ushort)bg[0] | ((ushort)bg[1] << 8));
            qg[r] = vload16(0, bg + 2);

            global const uchar *bu = up_ptrs[r] + (size_t)blk * 18;
            d_up[r] = fp16_to_fp32((ushort)bu[0] | ((ushort)bu[1] << 8));
            qu[r] = vload16(0, bu + 2);
        }

        float4 a0 = vload4(0, a_blk);
        float4 a1 = vload4(1, a_blk);
        float4 a2 = vload4(2, a_blk);
        float4 a3 = vload4(3, a_blk);
        float4 a4 = vload4(4, a_blk);
        float4 a5 = vload4(5, a_blk);
        float4 a6 = vload4(6, a_blk);
        float4 a7 = vload4(7, a_blk);

        float sum_a = (a0.x + a0.y + a0.z + a0.w) +
                      (a1.x + a1.y + a1.z + a1.w) +
                      (a2.x + a2.y + a2.z + a2.w) +
                      (a3.x + a3.y + a3.z + a3.w) +
                      (a4.x + a4.y + a4.z + a4.w) +
                      (a5.x + a5.y + a5.z + a5.w) +
                      (a6.x + a6.y + a6.z + a6.w) +
                      (a7.x + a7.y + a7.z + a7.w);

        float bias = 8.0f * sum_a;

        for (int r = 0; r < 8; r++) {
            uchar16 g16 = qg[r];
            uchar16 u16 = qu[r];

            uchar4 g0 = g16.s0123, g1 = g16.s4567, g2 = g16.s89ab, g3 = g16.scdef;
            uchar4 u0 = u16.s0123, u1 = u16.s4567, u2 = u16.s89ab, u3 = u16.scdef;

            float acc_g = dot(a0, convert_float4(g0 & (uchar4)0x0F)) + dot(a4, convert_float4(g0 >> (uchar4)4))
                        + dot(a1, convert_float4(g1 & (uchar4)0x0F)) + dot(a5, convert_float4(g1 >> (uchar4)4))
                        + dot(a2, convert_float4(g2 & (uchar4)0x0F)) + dot(a6, convert_float4(g2 >> (uchar4)4))
                        + dot(a3, convert_float4(g3 & (uchar4)0x0F)) + dot(a7, convert_float4(g3 >> (uchar4)4));

            float acc_u = dot(a0, convert_float4(u0 & (uchar4)0x0F)) + dot(a4, convert_float4(u0 >> (uchar4)4))
                        + dot(a1, convert_float4(u1 & (uchar4)0x0F)) + dot(a5, convert_float4(u1 >> (uchar4)4))
                        + dot(a2, convert_float4(u2 & (uchar4)0x0F)) + dot(a6, convert_float4(u2 >> (uchar4)4))
                        + dot(a3, convert_float4(u3 & (uchar4)0x0F)) + dot(a7, convert_float4(u3 >> (uchar4)4));

            sum_gate[r] += (acc_g - bias) * d_gate[r];
            sum_up[r]   += (acc_u - bias) * d_up[r];
        }
    }

    for (int r = 0; r < 8; r++) {
        l_gate[r][tid] = sum_gate[r];
        l_up[r][tid]   = sum_up[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32) {
        for (int r = 0; r < 8; r++) {
            l_gate[r][tid] += l_gate[r][tid + 32];
            l_up[r][tid]   += l_up[r][tid + 32];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16) {
        for (int r = 0; r < 8; r++) {
            l_gate[r][tid] += l_gate[r][tid + 16];
            l_up[r][tid]   += l_up[r][tid + 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8) {
        for (int r = 0; r < 8; r++) {
            l_gate[r][tid] += l_gate[r][tid + 8];
            l_up[r][tid]   += l_up[r][tid + 8];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4) {
        for (int r = 0; r < 8; r++) {
            l_gate[r][tid] += l_gate[r][tid + 4];
            l_up[r][tid]   += l_up[r][tid + 4];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2) {
        for (int r = 0; r < 8; r++) {
            l_gate[r][tid] += l_gate[r][tid + 2];
            l_up[r][tid]   += l_up[r][tid + 2];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        for (int r = 0; r < 8; r++) {
            int row = base_row + r;
            if (row < N) {
                float g = l_gate[r][0] + l_gate[r][1];
                float u = l_up[r][0] + l_up[r][1];
                float silu_g = g / (1.0f + exp(-g));
                dst[row] = silu_g * u;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Fused SwiGLU (128-bit SIMD Vectorized): dst[i] = silu(gate[i]) * up[i]
//------------------------------------------------------------------------------
kernel void swiglu_f32(
    global float *out,
    global const float *gate,
    global const float *up,
    int n
) {
    int i = get_global_id(0);
    int n4 = n / 4;
    if (i >= n4) return;

    float4 g = vload4(i, gate);
    float4 u = vload4(i, up);
    float4 silu_g = g / ((float4)(1.0f) + exp(-g));
    vstore4(silu_g * u, i, out);
}

//------------------------------------------------------------------------------
// GPU Parallel ArgMax: Reduces N floats to 1 max token index
// Dispatched with: Global = 256, Local = 256
//------------------------------------------------------------------------------
kernel void argmax_f32(
    global const float *logits,
    global int *out_idx,
    int N
) {
    local float l_max[256];
    local int l_idx[256];

    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    float max_val = -INFINITY;
    int max_idx = 0;

    for (int i = tid; i < N; i += wg_size) {
        float val = logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    l_max[tid] = max_val;
    l_idx[tid] = max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128) {
        if (l_max[tid + 128] > l_max[tid]) {
            l_max[tid] = l_max[tid + 128];
            l_idx[tid] = l_idx[tid + 128];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 64) {
        if (l_max[tid + 64] > l_max[tid]) {
            l_max[tid] = l_max[tid + 64];
            l_idx[tid] = l_idx[tid + 64];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 32) {
        if (l_max[tid + 32] > l_max[tid]) {
            l_max[tid] = l_max[tid + 32];
            l_idx[tid] = l_idx[tid + 32];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16) {
        if (l_max[tid + 16] > l_max[tid]) {
            l_max[tid] = l_max[tid + 16];
            l_idx[tid] = l_idx[tid + 16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8) {
        if (l_max[tid + 8] > l_max[tid]) {
            l_max[tid] = l_max[tid + 8];
            l_idx[tid] = l_idx[tid + 8];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4) {
        if (l_max[tid + 4] > l_max[tid]) {
            l_max[tid] = l_max[tid + 4];
            l_idx[tid] = l_idx[tid + 4];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2) {
        if (l_max[tid + 2] > l_max[tid]) {
            l_max[tid] = l_max[tid + 2];
            l_idx[tid] = l_idx[tid + 2];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        int best_idx = l_idx[0];
        if (l_max[1] > l_max[0]) best_idx = l_idx[1];
        *out_idx = best_idx;
    }
}

//------------------------------------------------------------------------------
// GPU Recurrent Conv1D + SiLU (In-Place GPU State)
//------------------------------------------------------------------------------
kernel void qwen_conv1d_silu(
    global float *conv_state,   // [3 * C]
    global const float *conv_in, // [C]
    global const float *weight,  // [4 * C]
    global float *conv_out,      // [C]
    int C
) {
    int c = get_global_id(0);
    if (c >= C) return;

    float s0 = conv_state[0 * C + c];
    float s1 = conv_state[1 * C + c];
    float s2 = conv_state[2 * C + c];
    float x  = conv_in[c];

    // GGML GGUF layout: contiguous 4 weights per channel
    float w0 = weight[c * 4 + 0];
    float w1 = weight[c * 4 + 1];
    float w2 = weight[c * 4 + 2];
    float w3 = weight[c * 4 + 3];

    float val = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
    float act = val / (1.0f + exp(-val));
    conv_out[c] = act;

    // Shift state in VRAM
    conv_state[0 * C + c] = s1;
    conv_state[1 * C + c] = s2;
    conv_state[2 * C + c] = x;
}

//------------------------------------------------------------------------------
// GPU Recurrent Gated DeltaNet Step (In-Place GPU State S)
// Dispatched with: Global = 16 * 128, Local = 128 (1 workgroup per head)
// Row-parallel execution: each thread i computes row i of head h independently!
//------------------------------------------------------------------------------
kernel void qwen_gated_deltanet_step(
    global float *ssm_state,       // [16, 128, 128]
    global const float *conv_out,   // [C = 6144]: qk_dim=2048 (q, k), linear_inner=2048 (v)
    global const float *alpha_vec,  // [16]
    global const float *beta_vec,   // [16]
    global float *delta_out,       // [linear_inner = 2048]
    int key_dim,                    // 128
    int qk_dim,                     // 2048
    int linear_inner                // 2048
) {
    int h = get_group_id(0);       // Head index in [0, 15]
    int i = get_local_id(0);       // Row index in [0, 127]

    if (h >= 16 || i >= key_dim) return;

    int key_heads = 16;
    int heads_per_group = 16 / key_heads;
    int kh = h / (heads_per_group > 0 ? heads_per_group : 1);

    global const float *q_head = conv_out + kh * key_dim;
    global const float *k_head = conv_out + qk_dim + kh * key_dim;
    global const float *v_head = conv_out + 2 * qk_dim + h * key_dim;

    float a_val = alpha_vec[h];
    float b_val = beta_vec[h];
    float g = 1.0f / (1.0f + exp(a_val));
    float b = 1.0f / (1.0f + exp(-b_val));

    global float *S_h = ssm_state + (size_t)h * (key_dim * key_dim);
    global float *S_row = S_h + (size_t)i * key_dim;

    float vi = v_head[i];
    float pred = 0.0f;

    for (int col = 0; col < key_dim; col++) {
        float s = S_row[col] * g;
        S_row[col] = s;
        pred += s * k_head[col];
    }

    float delta_i = b * (vi - pred);
    float yi = 0.0f;

    for (int col = 0; col < key_dim; col++) {
        float s = S_row[col] + delta_i * k_head[col];
        S_row[col] = s;
        yi += s * q_head[col];
    }

    global float *out_head = delta_out + h * key_dim;
    out_head[i] = yi;
}

//------------------------------------------------------------------------------
// Qwen 3.5 Full Attention Deinterleave Q & Sigmoid Gate
//------------------------------------------------------------------------------
kernel void qwen_deinterleave_q_gate(
    global const float *q_full,
    global float *q,
    global float *gate,
    int num_heads,
    int head_dim
) {
    int idx = get_global_id(0);
    int total = num_heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    int src_base = h * (2 * head_dim);
    q[idx] = q_full[src_base + d];
    gate[idx] = q_full[src_base + head_dim + d];
}

//------------------------------------------------------------------------------
// Qwen 3.5 Per-Head RMSNorm for Q and K
//------------------------------------------------------------------------------
kernel void qwen_per_head_rms_norm(
    global float *dst,
    global const float *src,
    global const float *norm_w,
    int num_heads,
    int head_dim,
    float eps
) {
    local float l_sq[256];
    int h = get_group_id(0);
    int d = get_local_id(0);
    if (h >= num_heads || d >= head_dim) return;

    float val = src[h * head_dim + d];
    l_sq[d] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = head_dim / 2; s > 0; s >>= 1) {
        if (d < s) l_sq[d] += l_sq[d + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_rms = rsqrt(l_sq[0] / (float)head_dim + eps);
    dst[h * head_dim + d] = val * inv_rms * norm_w[d];
}

//------------------------------------------------------------------------------
// Qwen 3.5 Sigmoid Attention Output Gate Multiply
//------------------------------------------------------------------------------
kernel void qwen_attn_gate_mul(
    global float *attn_out,
    global const float *gate,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + exp(-g));
    attn_out[i] *= sig;
}

//------------------------------------------------------------------------------
// GPU Causal Full Attention for 1 Token
// Dispatched with: Global = n_head * head_dim, Local = head_dim
//------------------------------------------------------------------------------
kernel void qwen_full_attention_step(
    global const float *q_buf,        // [n_head * head_dim]
    global const float *k_cache,      // [max_seq * (n_kv_head * head_dim)]
    global const float *v_cache,      // [max_seq * (n_kv_head * head_dim)]
    global float *attn_out,           // [n_head * head_dim]
    int n_head,
    int n_kv_head,
    int head_dim,
    int pos,
    int max_seq
) {
    local float l_scores[512];
    local float l_dot[256];

    int h = get_group_id(0);
    int d = get_local_id(0);

    if (h >= n_head || d >= head_dim) return;

    int q_per_kv = n_head / n_kv_head;
    int h_kv = h / (q_per_kv > 0 ? q_per_kv : 1);
    int kv_stride = n_kv_head * head_dim;

    int S = pos + 1;
    float inv_scale = rsqrt((float)head_dim);

    // Compute scores for head h
    global const float *q_h = q_buf + h * head_dim;

    for (int s = 0; s < S && s < 512; s++) {
        global const float *k_s = k_cache + (size_t)s * kv_stride + h_kv * head_dim;
        l_dot[d] = q_h[d] * k_s[d];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int step = head_dim / 2; step > 0; step >>= 1) {
            if (d < step) l_dot[d] += l_dot[d + step];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (d == 0) {
            l_scores[s] = l_dot[0] * inv_scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Softmax over l_scores[0..S-1]
    if (d == 0) {
        float maxv = l_scores[0];
        for (int s = 1; s < S && s < 512; s++) if (l_scores[s] > maxv) maxv = l_scores[s];
        float sum = 0.0f;
        for (int s = 0; s < S && s < 512; s++) {
            l_scores[s] = exp(l_scores[s] - maxv);
            sum += l_scores[s];
        }
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (int s = 0; s < S && s < 512; s++) l_scores[s] *= inv_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Weighted sum of V
    float acc = 0.0f;
    for (int s = 0; s < S && s < 512; s++) {
        float w = l_scores[s];
        global const float *v_s = v_cache + (size_t)s * kv_stride + h_kv * head_dim;
        acc += w * v_s[d];
    }

    attn_out[h * head_dim + d] = acc;
}

//------------------------------------------------------------------------------
// RoPE (Rotary Position Embedding)
//------------------------------------------------------------------------------
kernel void rope_f32(
    global float *x,
    int n_embd,
    int n_head,
    int pos,
    int n_tokens,
    float freq_base,
    int rope_dim
) {
    int token = get_global_id(0);
    int h_idx = get_global_id(1);
    int hh = get_global_id(2);

    if (token >= n_tokens || h_idx >= n_head) return;

    int head_dim = n_embd / n_head;
    int r_dim = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    if (hh >= r_dim / 2) return;

    global float *row = x + (size_t)token * n_embd + (size_t)h_idx * head_dim;
    float theta = (float)pos * pow(freq_base, -2.0f * (float)hh / (float)r_dim);
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    float v0 = row[hh];
    float v1 = row[hh + r_dim / 2];
    row[hh] = v0 * cos_t - v1 * sin_t;
    row[hh + r_dim / 2] = v0 * sin_t + v1 * cos_t;
}

//------------------------------------------------------------------------------
// Softmax
//------------------------------------------------------------------------------
kernel void softmax_f32(
    global float *x,
    int n,
    int rows
) {
    int row = get_global_id(0);
    if (row >= rows) return;

    global float *r = x + (size_t)row * n;

    float maxv = r[0];
    for (int i = 1; i < n; i++) {
        if (r[i] > maxv) maxv = r[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        r[i] = exp(r[i] - maxv);
        sum += r[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        r[i] *= inv_sum;
    }
}

//------------------------------------------------------------------------------
// SiLU
//------------------------------------------------------------------------------
kernel void silu_f32(
    global float *out,
    global const float *x,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    float xi = x[i];
    out[i] = xi / (1.0f + exp(-xi));
}

//------------------------------------------------------------------------------
// Element-wise Add (128-bit SIMD Vectorized)
//------------------------------------------------------------------------------
kernel void add_f32(
    global float *dst,
    global const float *a,
    global const float *b,
    int n
) {
    int i = get_global_id(0);
    int n4 = n / 4;
    if (i >= n4) return;
    float4 va = vload4(i, a);
    float4 vb = vload4(i, b);
    vstore4(va + vb, i, dst);
}

//------------------------------------------------------------------------------
// Element-wise Multiply (128-bit SIMD Vectorized)
//------------------------------------------------------------------------------
kernel void mul_f32(
    global float *dst,
    global const float *a,
    global const float *b,
    int n
) {
    int i = get_global_id(0);
    int n4 = n / 4;
    if (i >= n4) return;
    float4 va = vload4(i, a);
    float4 vb = vload4(i, b);
    vstore4(va * vb, i, dst);
}

//------------------------------------------------------------------------------
// Copy (128-bit SIMD Vectorized)
//------------------------------------------------------------------------------
kernel void copy_f32(
    global float *dst,
    global const float *src,
    int n
) {
    int i = get_global_id(0);
    int n4 = n / 4;
    if (i >= n4) return;
    float4 v = vload4(i, src);
    vstore4(v, i, dst);
}

//------------------------------------------------------------------------------
// Fill
//------------------------------------------------------------------------------
kernel void fill_f32(
    global float *buf,
    float val,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    buf[i] = val;
}

//------------------------------------------------------------------------------
// FUSED RECURRENT DELTANET OPERATOR:
// Combines: Q/K L2Norm + DeltaNet Step + SSM RMSNorm + Attn Gate SiLU + Elementwise Multiply
// Dispatched with: Global = 16 * 128, Local = 128
//------------------------------------------------------------------------------
kernel void qwen_gated_deltanet_fused(
    global float *ssm_state,         // [16, 128, 128]
    global const float *conv_out,    // [C = 6144]: qk_dim=2048 (q, k), linear_inner=2048 (v)
    global const float *alpha_vec,   // [16]
    global const float *beta_vec,    // [16]
    global const float *ssm_a,       // [16]
    global const float *ssm_dt,      // [16]
    global const float *ssm_norm_w,  // [128]
    global const float *attn_gate,   // [linear_inner = 2048]
    global float *delta_out,         // [linear_inner = 2048]
    int key_dim,                     // 128
    int qk_dim,                      // 2048
    int linear_inner,                // 2048
    float norm_eps                   // 1e-6f
) {
    local float l_sq[128];
    local float l_q[128];
    local float l_k[128];

    int h = get_group_id(0);         // Head index in [0, 15]
    int i = get_local_id(0);         // Row index in [0, 127]

    if (h >= 16 || i >= key_dim) return;

    int key_heads = 16;
    int heads_per_group = 16 / key_heads;
    int kh = h / (heads_per_group > 0 ? heads_per_group : 1);

    global const float *q_head = conv_out + kh * key_dim;
    global const float *k_head = conv_out + qk_dim + kh * key_dim;
    global const float *v_head = conv_out + 2 * qk_dim + h * key_dim;

    // 1. L2 Normalize Q and K per head with local memory reduction
    float qi = q_head[i];
    float ki = k_head[i];

    l_sq[i] = qi * qi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) l_sq[i] += l_sq[i + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm_q = rsqrt(l_sq[0] + 1e-6f);
    // Scale Q by 1 / sqrt(head_dim) = 1 / sqrt(128.0f) = 0.0883883476f
    l_q[i] = (qi * norm_q) * 0.0883883476f;
    barrier(CLK_LOCAL_MEM_FENCE);

    l_sq[i] = ki * ki;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) l_sq[i] += l_sq[i + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm_k = rsqrt(l_sq[0] + 1e-6f);
    l_k[i] = ki * norm_k;
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Compute decay and beta update gate using ssm_a and ssm_dt
    float a_param = (ssm_a != 0) ? ssm_a[h] : -1.0f;
    float dt_val  = (ssm_dt != 0) ? ssm_dt[h] : 0.0f;
    float alpha_biased = alpha_vec[h] + dt_val;
    float sp = (alpha_biased > 20.0f) ? alpha_biased : log(1.0f + exp(alpha_biased));
    float g = exp(sp * a_param);
    float b = 1.0f / (1.0f + exp(-beta_vec[h]));

    // 3. Recurrent DeltaNet step
    global float *S_h = ssm_state + (size_t)h * (key_dim * key_dim);
    global float *S_row = S_h + (size_t)i * key_dim;

    float vi = v_head[i];
    float pred = 0.0f;

    for (int col = 0; col < key_dim; col++) {
        float s = S_row[col] * g;
        S_row[col] = s;
        pred += s * l_k[col];
    }

    float delta_i = b * (vi - pred);
    float yi = 0.0f;

    for (int col = 0; col < key_dim; col++) {
        float s = S_row[col] + delta_i * l_k[col];
        S_row[col] = s;
        yi += s * l_q[col];
    }

    // 4. In-Group SSM RMSNorm over head h
    l_sq[i] = yi * yi;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) {
            l_sq[i] += l_sq[i + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean_sq = l_sq[0] / (float)key_dim;
    float inv_rms = rsqrt(mean_sq + norm_eps);
    float w_norm = (ssm_norm_w != 0) ? ssm_norm_w[i] : 1.0f;
    yi = yi * inv_rms * w_norm;

    // 5. Gating with attn_gate via SiLU
    if (attn_gate != 0) {
        float g_val = attn_gate[h * key_dim + i];
        float g_act = g_val / (1.0f + exp(-g_val));
        yi = yi * g_act;
    }

    global float *out_head = delta_out + h * key_dim;
    out_head[i] = yi;
}

//------------------------------------------------------------------------------
// GPU LOGITS ARGMAX REDUCTION (Greedy decoding: 4 bytes transferred to CPU!)
// Global = 1024, Local = 1024 (Single workgroup grid-stride reduction)
//------------------------------------------------------------------------------
kernel void logits_argmax(
    global const float *logits,
    int N,
    global int *out_token,
    global float *out_max_val
) {
    local float l_max_val[1024];
    local int   l_max_idx[1024];

    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = tid; i < N; i += wg_size) {
        float val = logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    l_max_val[tid] = max_val;
    l_max_idx[tid] = max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (l_max_val[tid + s] > l_max_val[tid]) {
                l_max_val[tid] = l_max_val[tid + s];
                l_max_idx[tid] = l_max_idx[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        *out_token = l_max_idx[0];
        if (out_max_val != 0) {
            *out_max_val = l_max_val[0];
        }
    }
}

//------------------------------------------------------------------------------
// GPU LOGITS TOP CANDIDATE FILTER (Temperature sampling: <= 64 candidates)
//------------------------------------------------------------------------------
kernel void logits_filter_candidates(
    global const float *logits,
    int N,
    float threshold,
    global volatile int *candidate_count,
    global float *out_logits,
    global int *out_ids,
    int max_candidates
) {
    int tid = get_global_id(0);
    int stride = get_global_size(0);

    for (int i = tid; i < N; i += stride) {
        float val = logits[i];
        if (val >= threshold) {
            int slot = atomic_inc(candidate_count);
            if (slot < max_candidates) {
                out_logits[slot] = val;
                out_ids[slot] = i;
            }
        }
    }
}

