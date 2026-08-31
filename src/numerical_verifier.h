#pragma once
#include <string>
#include <vector>
#include "opencl_backend.h"
#include "backends/intel_uhd_backend.h"

struct NumericalMetric
{
    std::string layer_name;
    double max_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double cosine_similarity = 1.0;
    bool passed = true;
};

class NumericalVerifier
{
public:
    static double compute_max_abs_error(const float *a, const float *b, size_t n);
    static double compute_mean_abs_error(const float *a, const float *b, size_t n);
    static double compute_cosine_similarity(const float *a, const float *b, size_t n);

    // Runs comprehensive operator verification against CPU reference
    static std::vector<NumericalMetric> verify_layers(OpenClBackend *cl_backend, IntelUhdBackend *intel_backend);

    // Runs end-to-end model-level layer-by-layer verification
    static std::vector<NumericalMetric> verify_model_layer_by_layer(const LlamaModel &model, OpenClBackend *cl_backend);
};
