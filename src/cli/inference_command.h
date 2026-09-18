#pragma once

#include "cli_options.h"
#include "../inference.h"

class InferenceCommand
{
public:
    static int execute_interactive(const CliOptions &opt, InferenceEngine &engine);
    static int execute_generate(const CliOptions &opt, InferenceEngine &engine);
};
