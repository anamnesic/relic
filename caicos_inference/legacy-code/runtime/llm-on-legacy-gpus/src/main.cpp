#include "gguf_reader.h"
#include "model.h"
#include "opencl_backend.h"
#include "tokenizer.h"
#include "inference.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

void print_usage(const char *prog) {
    fprintf(stdout, "Caicos RT - Minimal Inference Runtime for OpenCL 1.2\n");
    fprintf(stdout, "Usage: %s [options] -m <model.gguf>\n", prog);
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -m <file>         Model file (GGUF format)\n");
    fprintf(stdout, "  -p <prompt>       Input prompt\n");
    fprintf(stdout, "  -n <int>          Number of tokens to generate (default: 256)\n");
    fprintf(stdout, "  -t <float>        Temperature (default: 0.8)\n");
    fprintf(stdout, "  -k <int>          Top-k sampling (default: 40)\n");
    fprintf(stdout, "  --list-devices    List OpenCL devices and exit\n");
    fprintf(stdout, "  --platform <int>  OpenCL platform index (default: 0)\n");
    fprintf(stdout, "  --device <int>    OpenCL device index (default: 0)\n");
    fprintf(stdout, "  --cpu             Force CPU-only mode\n");
    fprintf(stdout, "  --max-seq-len <int> Maximum sequence length (default: 2048)\n");
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "Once upon a time";
    int n_tokens = 256;
    float temperature = 0.8f;
    int top_k = 40;
    int platform_idx = 0;
    int device_idx = 0;
    bool list_devices = false;
    bool cpu_only = false;
    int max_seq_len = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--list-devices") == 0) list_devices = true;
        else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc) platform_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device_idx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu") == 0) cpu_only = true;
        else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) max_seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // List devices mode
    if (list_devices) {
        OpenClBackend cl;
        cl_uint n_platforms;
        clGetPlatformIDs(0, nullptr, &n_platforms);
        if (n_platforms == 0) {
            fprintf(stdout, "No OpenCL platforms found\n");
            return 0;
        }
        std::vector<cl_platform_id> platforms(n_platforms);
        clGetPlatformIDs(n_platforms, platforms.data(), nullptr);

        for (cl_uint p = 0; p < n_platforms; p++) {
            char name[1024];
            clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
            fprintf(stdout, "Platform %u: %s\n", p, name);

            cl_uint nd;
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
            if (nd > 0) {
                std::vector<cl_device_id> devs(nd);
                clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
                for (cl_uint d = 0; d < nd; d++) {
                    char dname[1024], version[128];
                    cl_device_type dtype;
                    size_t mem;
                    clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
                    clGetDeviceInfo(devs[d], CL_DEVICE_VERSION, sizeof(version), version, nullptr);
                    clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
                    clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
                    fprintf(stdout, "  Device %d: %s [%s] %zu MB %s\n",
                            d, dname, version, mem / (1024 * 1024),
                            dtype == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");
                }
            }
        }
        return 0;
    }

    if (model_path.empty()) {
        fprintf(stderr, "No model file specified. Use -m <model.gguf>\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    fprintf(stdout, "Loading model...\n");
    LlamaModel model;
    if (!model.load(model_path.c_str())) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Load tokenizer
    GgufReader reader;
    if (!reader.load(model_path.c_str())) {
        fprintf(stderr, "Failed to read GGUF metadata for tokenizer\n");
        return 1;
    }

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(reader)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Initialize OpenCL
    OpenClBackend cl;
    bool cl_ok = false;
    if (!cpu_only) {
        cl_ok = cl.init(platform_idx, device_idx);
        if (!cl_ok) {
            fprintf(stdout, "OpenCL init failed, falling back to CPU\n");
        } else {
            fprintf(stdout, "OpenCL initialized: %s\n", cl.dev.name.c_str());
        }
    }

    // Initialize inference engine
    InferenceEngine engine;
    if (!engine.init(&model, &tokenizer, cl_ok ? &cl : nullptr, max_seq_len)) {
        fprintf(stderr, "Failed to initialize inference engine\n");
        return 1;
    }

    fprintf(stdout, "\n=== Caicos RT Inference ===\n");
    fprintf(stdout, "Prompt: %s\n", prompt.c_str());
    fprintf(stdout, "Generating %d tokens...\n\n", n_tokens);

    std::string result = engine.generate(prompt, n_tokens, temperature, top_k);

    fprintf(stdout, "\n=== Done ===\n");

    engine.free_buffers();
    fflush(stdout);
    return 0;
}
