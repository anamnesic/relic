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
                    uint16_t d_raw = *(const uint16_t *)b_sb;
                    uint16_t dmin_raw = *(const uint16_t *)(b_sb + 2);
                    float d = half_bits_to_float(d_raw);
                    float dmin = half_bits_to_float(dmin_raw);
                    const uint8_t *qs = b_sb + 16;
                    for (int64_t i = 0; i < 128; i++)
                    {
                        uint8_t q = qs[i];
                        int q0 = q & 0x0F;
                        int q1 = q >> 4;
                        out_row[sb * 256 + i] = (float)q0 * d - dmin;
                        out_row[sb * 256 + i + 128] = (float)q1 * d - dmin;
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
                    const uint8_t *ql = b_sb;
                    const uint8_t *qh = b_sb + 128;
                    const int8_t *scales = (const int8_t *)(b_sb + 192);
                    uint16_t d_raw = *(const uint16_t *)(b_sb + 208);
                    float d = half_bits_to_float(d_raw);

                    for (int64_t i = 0; i < 128; i++)
                    {
                        int ql0 = ql[i] & 0x0F;
                        int ql1 = ql[i] >> 4;
                        uint8_t qh_byte = qh[i / 2];
                        int qh0 = (i % 2 == 0) ? (qh_byte & 0x03) : ((qh_byte >> 2) & 0x03);
                        int qh1 = (i % 2 == 0) ? ((qh_byte >> 4) & 0x03) : ((qh_byte >> 6) & 0x03);

                        int q0 = (ql0 | (qh0 << 4)) - 32;
                        int q1 = (ql1 | (qh1 << 4)) - 32;

                        float sc0 = (float)scales[i / 8];
                        float sc1 = (float)scales[(i + 128) / 8];

                        out_row[sb * 256 + i] = d * sc0 * (float)q0;
                        out_row[sb * 256 + i + 128] = d * sc1 * (float)q1;
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
