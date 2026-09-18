#pragma once

#include "cli_options.h"
#include "../model.h"
#include "../tokenizer.h"
#include "../opencl_backend.h"
#include "../inference.h"
#include "../planner/hardware_profile.h"

class BenchmarkCommand
{
public:
    static int execute_ablation(const CliOptions &opt, NeuralModel &model, Tokenizer &tokenizer,
                                OpenClBackend *cl, bool cl_ok, const HardwareProfile &prof);

    static int execute_cliff_analysis(const CliOptions &opt, NeuralModel &model, Tokenizer &tokenizer,
                                      OpenClBackend *cl, bool cl_ok, const HardwareProfile &prof);

    static int execute_sweep(const CliOptions &opt, NeuralModel &model, Tokenizer &tokenizer,
                             OpenClBackend *cl, bool cl_ok, const HardwareProfile &prof);

    static int execute_standard_bench(const CliOptions &opt, InferenceEngine &engine);
};
