#pragma once

#include "architecture.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace relic::quant
{

    // Half-precision (IEEE 754-2008 binary16) bit conversions
    inline uint16_t float_to_half_bits(float f)
    {
        uint32_t u;
        memcpy(&u, &f, 4);
        uint16_t sign = (u >> 16) & 0x8000;
        int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = u & 0x007FFFFF;
        if (exp <= 0)
        {
            mant = (mant | 0x00800000) >> (1 - exp);
            return (uint16_t)(sign | (mant >> 13));
        }
        if (exp >= 31)
            return (uint16_t)(sign | 0x7C00);
        return (uint16_t)(sign | (exp << 10) | (mant >> 13));
    }

    inline float half_bits_to_float(uint16_t h)
    {
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = ((h >> 10) & 0x1F) - 15 + 127;
        uint32_t mant = h & 0x03FF;
        if (exp <= 0)
        {
            mant = (mant | 0x0400) >> (1 - exp);
            exp = 0;
        }
        if (exp >= 255)
        {
            uint32_t u = sign | 0x7F800000 | (mant << 13);
            float f;
            memcpy(&f, &u, 4);
            return f;
        }
        uint32_t u = sign | (exp << 23) | (mant << 13);
        float f;
        memcpy(&f, &u, 4);
        return f;
    }

    inline void dequantize_q4_0_row(const uint8_t *src, float *out, int64_t n)
    {
        int64_t nb = n / 32;
        for (int64_t b = 0; b < nb; b++)
        {
            const uint8_t *block = src + b * 18;
            uint16_t d_raw = *(const uint16_t *)block;
            float d = half_bits_to_float(d_raw);
            const uint8_t *qs = block + 2;

            for (int64_t i = 0; i < 16; i++)
            {
                uint8_t byte_val = qs[i];
                int8_t v0 = (int8_t)(byte_val & 0x0F) - 8;
                int8_t v1 = (int8_t)(byte_val >> 4) - 8;
                out[b * 32 + i] = (float)v0 * d;
                out[b * 32 + i + 16] = (float)v1 * d;
            }
        }
    }

    inline void dequantize_q8_0_row(const uint8_t *src, float *out, int64_t n)
    {
        int64_t nb = n / 32;
        for (int64_t b = 0; b < nb; b++)
        {
            const uint8_t *block = src + b * 34;
            uint16_t d_raw = *(const uint16_t *)block;
            float d = half_bits_to_float(d_raw);
            const int8_t *qs = (const int8_t *)(block + 2);

            for (int64_t i = 0; i < 32; i++)
            {
                out[b * 32 + i] = (float)qs[i] * d;
            }
        }
    }

    inline void dequantize_f16_row(const uint8_t *block, float *out, int64_t n)
    {
        const uint16_t *f16 = (const uint16_t *)block;
        for (int64_t i = 0; i < n; i++)
        {
            out[i] = half_bits_to_float(f16[i]);
        }
    }

    inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
    {
        if (j < 4)
        {
            *d = (uint8_t)(q[j] & 63);
            *m = (uint8_t)(q[j + 4] & 63);
        }
        else
        {
            *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    inline void dequantize_rows(GgmlType type, const uint8_t *data, int64_t row_size,
                                int64_t start_row, int64_t num_rows, float *out)
    {
        if (num_rows <= 0 || row_size <= 0)
            return;

        switch (type)
        {
        case GgmlType::F32:
        {
            const float *src = (const float *)data;
            memcpy(out, src + start_row * row_size, (size_t)(num_rows * row_size * sizeof(float)));
            break;
        }
        case GgmlType::F16:
        {
            const uint8_t *src = data + start_row * row_size * 2;
            dequantize_f16_row(src, out, num_rows * row_size);
            break;
        }
        case GgmlType::Q4_0:
        {
            size_t bytes_per_row = (size_t)((row_size + 31) / 32) * 18;
            for (int64_t r = 0; r < num_rows; r++)
            {
                const uint8_t *src_row = data + (start_row + r) * bytes_per_row;
                dequantize_q4_0_row(src_row, out + r * row_size, row_size);
            }
            break;
        }
        case GgmlType::Q8_0:
        {
            size_t bytes_per_row = (size_t)((row_size + 31) / 32) * 34;
            for (int64_t r = 0; r < num_rows; r++)
            {
                const uint8_t *src_row = data + (start_row + r) * bytes_per_row;
                dequantize_q8_0_row(src_row, out + r * row_size, row_size);
            }
            break;
        }
        case GgmlType::Q4_1:
        {
            size_t bytes_per_row = (size_t)((row_size + 31) / 32) * 20;
            for (int64_t r = 0; r < num_rows; r++)
            {
                const uint8_t *src_row = data + (start_row + r) * bytes_per_row;
                int64_t n_blocks = row_size / 32;
                float *out_row = out + r * row_size;
                for (int64_t b = 0; b < n_blocks; b++)
                {
                    const uint8_t *block = src_row + b * 20;
                    float d = half_bits_to_float(*(const uint16_t *)block);
                    float m = half_bits_to_float(*(const uint16_t *)(block + 2));
                    const uint8_t *qs = block + 4;
                    for (int64_t i = 0; i < 16; i++)
                    {
                        int x0 = qs[i] & 0x0F;
                        int x1 = qs[i] >> 4;
                        out_row[b * 32 + i] = (float)x0 * d + m;
                        out_row[b * 32 + i + 16] = (float)x1 * d + m;
                    }
                }
            }
            break;
        }
        case GgmlType::Q4_K:
        {
            size_t bytes_per_row = (size_t)((row_size + 255) / 256) * 144;
            for (int64_t r = 0; r < num_rows; r++)
            {
                const uint8_t *src_row = data + (start_row + r) * bytes_per_row;
                int64_t n_super_blocks = row_size / 256;
                float *out_row = out + r * row_size;
                for (int64_t sb = 0; sb < n_super_blocks; sb++)
                {
                    const uint8_t *b_sb = src_row + sb * 144;
                    float d = half_bits_to_float(*(const uint16_t *)b_sb);
                    float min = half_bits_to_float(*(const uint16_t *)(b_sb + 2));
                    const uint8_t *scales = b_sb + 4;
                    const uint8_t *q = b_sb + 16;
                    float *y = out_row + sb * 256;
                    int is = 0;
                    uint8_t sc = 0, mn = 0;
                    for (int j = 0; j < 256; j += 64)
                    {
                        get_scale_min_k4(is + 0, scales, &sc, &mn);
                        float d1 = d * (float)sc, m1 = min * (float)mn;
                        get_scale_min_k4(is + 1, scales, &sc, &mn);
                        float d2 = d * (float)sc, m2 = min * (float)mn;
                        for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0x0F) - m1;
                        for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
                        q += 32;
                        is += 2;
                    }
                }
            }
            break;
        }
        case GgmlType::Q5_K:
        {
            size_t bytes_per_row = (size_t)((row_size + 255) / 256) * 176;
            for (int64_t r = 0; r < num_rows; r++)
            {
                const uint8_t *src_row = data + (start_row + r) * bytes_per_row;
                int64_t n_super_blocks = row_size / 256;
                float *out_row = out + r * row_size;
                for (int64_t sb = 0; sb < n_super_blocks; sb++)
                {
                    const uint8_t *b_sb = src_row + sb * 176;
                    float d = half_bits_to_float(*(const uint16_t *)b_sb);
                    float min = half_bits_to_float(*(const uint16_t *)(b_sb + 2));
                    const uint8_t *scales = b_sb + 4;
                    const uint8_t *qh = b_sb + 16;
                    const uint8_t *ql = b_sb + 48;
                    float *y = out_row + sb * 256;
                    int is = 0;
                    uint8_t u1 = 1, u2 = 2;
                    uint8_t sc = 0, mn = 0;
                    for (int j = 0; j < 256; j += 64)
                    {
                        get_scale_min_k4(is + 0, scales, &sc, &mn);
                        float d1 = d * (float)sc, m1 = min * (float)mn;
                        get_scale_min_k4(is + 1, scales, &sc, &mn);
                        float d2 = d * (float)sc, m2 = min * (float)mn;
                        for (int l = 0; l < 32; l++)
                            *y++ = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
                        for (int l = 0; l < 32; l++)
                            *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                        ql += 32;
                        is += 2;
                        u1 = (uint8_t)(u1 << 2);
                        u2 = (uint8_t)(u2 << 2);
                    }
                }
            }
            break;
        }
        case GgmlType::Q6_K:
        {
            size_t bytes_per_row = (size_t)((row_size + 255) / 256) * 210;
            for (int64_t r = 0; r < num_rows; r++)
            {
                const uint8_t *src_row = data + (start_row + r) * bytes_per_row;
                int64_t n_super_blocks = row_size / 256;
                float *out_row = out + r * row_size;
                for (int64_t sb = 0; sb < n_super_blocks; sb++)
                {
                    const uint8_t *b_sb = src_row + sb * 210;
                    float d = half_bits_to_float(*(const uint16_t *)(b_sb + 208));
                    const uint8_t *ql = b_sb;
                    const uint8_t *qh = b_sb + 128;
                    const int8_t *sc = (const int8_t *)(b_sb + 192);
                    float *y = out_row + sb * 256;
                    for (int n = 0; n < 256; n += 128)
                    {
                        for (int l = 0; l < 32; l++)
                        {
                            int is = l / 16;
                            int q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                            int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                        }
                        y += 128;
                        ql += 64;
                        qh += 32;
                        sc += 8;
                    }
                }
            }
            break;
        }
        default:
            fprintf(stderr, "Unsupported type for row dequantization: %d\n", (int)type);
            memset(out, 0, (size_t)(num_rows * row_size * sizeof(float)));
            break;
        }
    }

} // namespace relic::quant
