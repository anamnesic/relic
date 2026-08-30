#include "gguf_reader.h"
#include "model.h"
#include "architecture.h"
#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    GgufReader reader;
    if (!reader.load(argv[1])) { fprintf(stderr, "Failed to load GGUF\n"); return 1; }

    fprintf(stdout, "=== Architecture Config Test ===\n");
    std::string arch_name = reader.get_metadata<std::string>("general.architecture", "");
    fprintf(stdout, "general.architecture: %s\n", arch_name.c_str());

    const ModelArchitecture *adapter = find_architecture(arch_name);
    ArchitectureSpec spec;
    std::string err;
    if (!adapter) {
        fprintf(stdout, "No adapter for %s\n", arch_name.c_str());
    } else if (!adapter->configure(reader, spec, err)) {
        fprintf(stdout, "Adapter configure failed: %s\n", err.c_str());
    } else {
        fprintf(stdout, "Adapter configure OK: kind=%d, vocab=%lld, embd=%lld, layers=%lld, heads=%lld, kv_heads=%lld, inner=%lld\n",
                (int)spec.kind, (long long)spec.n_vocab, (long long)spec.n_embd,
                (long long)spec.n_layer, (long long)spec.n_head, (long long)spec.n_head_kv,
                (long long)spec.linear_inner_size);
    }

    fprintf(stdout, "\n=== All Tensors (%zu) ===\n", reader.tensors.size());
    for (auto &[name, info] : reader.tensors) {
        if (name.rfind("v.", 0) == 0) continue; // skip vision
        fprintf(stdout, "  %-40s type=%-2d dims=[", name.c_str(), (int)info.type);
        for (size_t d = 0; d < info.dims.size(); d++) {
            if (d) fprintf(stdout, ",");
            fprintf(stdout, "%lld", (long long)info.dims[d]);
        }
        fprintf(stdout, "]\n");
    }
    return 0;
}
