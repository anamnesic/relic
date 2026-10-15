// caicos_rt - f32 OpenCL 1.2 kernels
// No FP16, no subgroups, pure OpenCL C 1.2

//------------------------------------------------------------------------------
// RMS Norm
//------------------------------------------------------------------------------
kernel void rms_norm_f32(
    global float *out,
    global const float *x,
    global const float *weight,
    int n,
    float eps
) {
    int row = get_global_id(0);
    x += row * n;
    out += row * n;

    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = rsqrt(ss / (float)n + eps);

    for (int i = 0; i < n; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

//------------------------------------------------------------------------------
// Matrix Multiply (A: MxK, B: KxN, C: MxN) - Row-major A/B
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
    global const float *a_row = a + row * K;
    global const float *b_col = b + col;
    for (int i = 0; i < K; i++) {
        sum += a_row[i] * b_col[i * N];
    }
    dst[row * N + col] = sum;
}

//------------------------------------------------------------------------------
// Matrix Multiply NT (A: MxK, B: NxK, C: MxN) - B is transposed in memory
// More efficient for weight * activation: weights stored as [K, N]
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
    global const float *a_row = a + row * K;
    global const float *b_row = b + col * K;
    for (int i = 0; i < K; i++) {
        sum += a_row[i] * b_row[i];
    }
    dst[row * N + col] = sum;
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

    global float *row = x + token * n_embd + h_idx * head_dim;
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

    global float *r = x + row * n;

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

//------------------------------------------------------------------------------
// Causal softmax (masked)
//------------------------------------------------------------------------------
kernel void causal_softmax_f32(
    global float *x,
    int n,
    int rows
) {
    int row = get_global_id(0);
    if (row >= rows) return;

    global float *r = x + row * n;

    float maxv = -INFINITY;
    for (int i = 0; i <= row && i < n; i++) {
        if (r[i] > maxv) maxv = r[i];
    }
    if (maxv == -INFINITY) maxv = 0.0f;

    float sum = 0.0f;
    for (int i = 0; i <= row && i < n; i++) {
        r[i] = exp(r[i] - maxv);
        sum += r[i];
    }
    for (int i = row + 1; i < n; i++) {
        r[i] = 0.0f;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i <= row && i < n; i++) {
        r[i] *= inv_sum;
    }
}
