#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace relic::cpu {

inline void matmul_nt_cpu(float *dst, const float *a, const float *b,
                          int64_t M, int64_t N, int64_t K)
{
    for (int64_t m = 0; m < M; m++)
    {
        for (int64_t n = 0; n < N; n++)
        {
            float sum = 0.0f;
            const float *a_row = a + m * K;
            const float *b_row = b + n * K;
            for (int64_t k = 0; k < K; k++)
            {
                sum += a_row[k] * b_row[k];
            }
            dst[m * N + n] = sum;
        }
    }
}

inline void rms_norm_cpu(float *dst, const float *x, const float *weight,
                         int64_t n, int64_t rows, float eps = 1e-6f)
{
    for (int64_t r = 0; r < rows; r++)
    {
        float sum = 0.0f;
        const float *xr = x + r * n;
        for (int64_t i = 0; i < n; i++)
            sum += xr[i] * xr[i];
        float inv_rms = 1.0f / sqrtf(sum / (float)n + eps);
        float *dr = dst + r * n;
        for (int64_t i = 0; i < n; i++)
            dr[i] = xr[i] * inv_rms * weight[i];
    }
}

inline void silu_cpu(float *dst, const float *x, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        float xi = x[i];
        dst[i] = xi / (1.0f + expf(-xi));
    }
}

inline void add_cpu(float *dst, const float *a, const float *b, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        dst[i] = a[i] + b[i];
}

inline void rope_cpu(float *x, int64_t n_embd, int64_t n_head, int pos, int n_tokens,
                     float base = 10000.0f, int64_t rope_dim = 0)
{
    int64_t head_dim = n_embd / n_head;
    int64_t r_dim = (rope_dim > 0 && rope_dim <= head_dim) ? rope_dim : head_dim;
    for (int t = 0; t < n_tokens; t++)
    {
        for (int64_t h = 0; h < n_head; h++)
        {
            float *row = x + t * n_embd + h * head_dim;
            for (int64_t hh = 0; hh < r_dim / 2; hh++)
            {
                float theta = (float)pos * powf(base, -2.0f * (float)hh / (float)r_dim);
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                float v0 = row[hh];
                float v1 = row[hh + r_dim / 2];
                row[hh] = v0 * cos_t - v1 * sin_t;
                row[hh + r_dim / 2] = v0 * sin_t + v1 * cos_t;
            }
        }
    }
}

} // namespace relic::cpu
