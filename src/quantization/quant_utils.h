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
        default:
            fprintf(stderr, "Unsupported type for row dequantization: %d\n", (int)type);
            memset(out, 0, (size_t)(num_rows * row_size * sizeof(float)));
            break;
        }
    }

} // namespace relic::quant
