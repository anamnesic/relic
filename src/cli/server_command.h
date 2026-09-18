#pragma once

#include "cli_options.h"
#include "../inference.h"

class ServerCommand
{
public:
    static int execute_server(const CliOptions &opt, InferenceEngine &engine);
    static int execute_client(const CliOptions &opt);
};
