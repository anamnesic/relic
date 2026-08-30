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
// RMS Norm (Root Mean Square Normalization)
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

    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += rx[i] * rx[i];
    }
    ss = rsqrt(ss / (float)n + eps);

    for (int i = 0; i < n; i++) {
        rout[i] = rx[i] * ss * weight[i];
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
// Matrix Multiply NT (A: MxK, B: NxK, C: MxN) - B is transposed in memory
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
// GEMV F32 NT (Matrix-Vector multiply: dst[col] = dot(a, b[col, :]))
// Dispatched with GlobalSize = N * WorkGroupSize, LocalSize = WorkGroupSize
//------------------------------------------------------------------------------
kernel void gemv_f32_nt(
    global const float *a,
    global const float *b,
    global float *dst,
    int N,
    int K
) {
    int col = get_group_id(0);
    if (col >= N) return;
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    global const float *b_row = b + (size_t)col * K;

    float sum = 0.0f;
    for (int i = tid; i < K; i += wg_size) {
        sum += a[i] * b_row[i];
    }

    local float l_sum[128];
    l_sum[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (tid < s) l_sum[tid] += l_sum[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        dst[col] = l_sum[0];
    }
}

//------------------------------------------------------------------------------
// FUSED GEMV Q8_0: Direct execution from resident quantized VRAM weights!
// Q8_0 block = 2 bytes (fp16 scale d) + 32 bytes (int8_t qs[32]) = 34 bytes
//------------------------------------------------------------------------------
kernel void gemv_q8_0(
    global const float *a,
    global const uchar *b,
    global float *dst,
    int N,
    int K
) {
    int col = get_group_id(0);
    if (col >= N) return;
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    int n_blocks = K / 32;
    global const uchar *row_ptr = b + (size_t)col * (size_t)(n_blocks * 34);

    float sum = 0.0f;
    for (int blk = tid; blk < n_blocks; blk += wg_size) {
        global const uchar *b_blk = row_ptr + (size_t)blk * 34;
        ushort d_bits = (ushort)b_blk[0] | ((ushort)b_blk[1] << 8);
        float d = fp16_to_fp32(d_bits);
        global const char *qs = (global const char *)(b_blk + 2);
        global const float *a_blk = a + blk * 32;

        for (int i = 0; i < 32; i++) {
            sum += a_blk[i] * ((float)qs[i] * d);
        }
    }

    local float l_sum[128];
    l_sum[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (tid < s) l_sum[tid] += l_sum[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        dst[col] = l_sum[0];
    }
}

//------------------------------------------------------------------------------
// FUSED GEMV Q4_0: Direct execution from resident quantized VRAM weights!
// Q4_0 block = 2 bytes (fp16 scale d) + 16 bytes (nibbles qs[16]) = 18 bytes
//------------------------------------------------------------------------------
kernel void gemv_q4_0(
    global const float *a,
    global const uchar *b,
    global float *dst,
    int N,
    int K
) {
    int col = get_group_id(0);
    if (col >= N) return;
    int tid = get_local_id(0);
    int wg_size = get_local_size(0);

    int n_blocks = K / 32;
    global const uchar *row_ptr = b + (size_t)col * (size_t)(n_blocks * 18);

    float sum = 0.0f;
    for (int blk = tid; blk < n_blocks; blk += wg_size) {
        global const uchar *b_blk = row_ptr + (size_t)blk * 18;
        ushort d_bits = (ushort)b_blk[0] | ((ushort)b_blk[1] << 8);
        float d = fp16_to_fp32(d_bits);
        global const uchar *qs = b_blk + 2;
        global const float *a_blk = a + blk * 32;

        for (int i = 0; i < 16; i++) {
            uchar val = qs[i];
            float v0 = (float)((int)(val & 0x0F) - 8);
            float v1 = (float)((int)(val >> 4) - 8);
            sum += a_blk[i] * (v0 * d) + a_blk[i + 16] * (v1 * d);
        }
    }

    local float l_sum[128];
    l_sum[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wg_size / 2; s > 0; s >>= 1) {
        if (tid < s) l_sum[tid] += l_sum[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        dst[col] = l_sum[0];
    }
}

//------------------------------------------------------------------------------
// Fused SwiGLU: dst[i] = silu(gate[i]) * up[i]
//------------------------------------------------------------------------------
kernel void swiglu_f32(
    global float *out,
    global const float *gate,
    global const float *up,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    float g = gate[i];
    float silu_g = g / (1.0f + exp(-g));
    out[i] = silu_g * up[i];
}

//------------------------------------------------------------------------------
// RoPE (Rotary Position Embedding)
//------------------------------------------------------------------------------
kernel void rope_f32(
    global float *x,
    int n_embd,
    int n_head,
    int pos,
    int n_tokens
) {
    int token = get_global_id(0);
    int h_idx = get_global_id(1);
    int hh = get_global_id(2);

    if (token >= n_tokens) return;
    if (h_idx >= n_head) return;

    int head_dim = n_embd / n_head;
    if (hh >= head_dim / 2) return;

    global float *row = x + (size_t)token * n_embd + (size_t)h_idx * head_dim;
    float theta = (float)pos * pow(10000.0f, -2.0f * (float)hh / (float)head_dim);
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    float v0 = row[hh];
    float v1 = row[hh + head_dim / 2];
    row[hh] = v0 * cos_t - v1 * sin_t;
    row[hh + head_dim / 2] = v0 * sin_t + v1 * cos_t;
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
// SiLU (Sigmoid Linear Unit)
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
// Element-wise Add
//------------------------------------------------------------------------------
kernel void add_f32(
    global float *dst,
    global const float *a,
    global const float *b,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = a[i] + b[i];
}

//------------------------------------------------------------------------------
// Element-wise Multiply
//------------------------------------------------------------------------------
kernel void mul_f32(
    global float *dst,
    global const float *a,
    global const float *b,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = a[i] * b[i];
}

//------------------------------------------------------------------------------
// Copy
//------------------------------------------------------------------------------
kernel void copy_f32(
    global float *dst,
    global const float *src,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i];
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
