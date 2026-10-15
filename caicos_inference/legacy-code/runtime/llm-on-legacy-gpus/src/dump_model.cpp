#include "gguf_reader.h"
#include "model.h"
#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    GgufReader reader;
    if (!reader.load(argv[1])) { fprintf(stderr, "Failed to load GGUF\n"); return 1; }

    fprintf(stdout, "=== Metadata ===\n");
    for (auto &[k, v] : reader.metadata_str)
        fprintf(stdout, "  str %s = %s\n", k.c_str(), v.c_str());
    for (auto &[k, v] : reader.metadata_int)
        fprintf(stdout, "  int %s = %lld\n", k.c_str(), (long long)v);
    for (auto &[k, v] : reader.metadata_float)
        fprintf(stdout, "  flo %s = %f\n", k.c_str(), v);
    for (auto &[k, v] : reader.metadata_uint32)
        fprintf(stdout, "  u32 %s = %u\n", k.c_str(), v);

    fprintf(stdout, "\n=== Tensors (%zu) ===\n", reader.tensors.size());
    int i = 0;
    for (auto &[name, info] : reader.tensors) {
        int64_t ne = tensor_nelements(info);
        int tsz = ggml_type_size(info.type);
        int blck = ggml_blck_size(info.type);
        fprintf(stdout, "  %-45s type=%-3d dims=[", name.c_str(), (int)info.type);
        for (size_t d = 0; d < info.dims.size(); d++) {
            if (d) fprintf(stdout, ",");
            fprintf(stdout, "%lld", (long long)info.dims[d]);
        }
        fprintf(stdout, "] ne=%lld tsz=%d blck=%d\n", (long long)ne, tsz, blck);
        if (++i > 20) { fprintf(stdout, "  ... (%zu total)\n", reader.tensors.size()); break; }
    }

    // Verify first weight tensor by dequantizing first few values
    fprintf(stdout, "\n=== Weight verification ===\n");
    auto first = reader.tensors.begin();
    if (first != reader.tensors.end()) {
        const void *src = reader.tensor_data(first->first);
        int64_t ne = tensor_nelements(first->second);
        fprintf(stdout, "  Tensor: %s, ne=%lld, type=%d\n",
                first->first.c_str(), (long long)ne, (int)first->second.type);
        if (first->second.type == GgmlType::F16 && ne > 0) {
            const uint16_t *f16 = (const uint16_t *)src;
            fprintf(stdout, "  First 8 F16 values (hex):");
            for (int j = 0; j < 8 && j < ne; j++) {
                fprintf(stdout, " %04X", f16[j]);
            }
            fprintf(stdout, "\n");
            // Dequantize first 8
            float f32[8];
            for (int j = 0; j < 8 && j < ne; j++) {
                f32[j] = half_bits_to_float(f16[j]);
            }
            fprintf(stdout, "  Dequantized F32 values:");
            for (int j = 0; j < 8 && j < ne; j++) {
                fprintf(stdout, " %f", f32[j]);
            }
            fprintf(stdout, "\n");
        }
    }

    return 0;
}
