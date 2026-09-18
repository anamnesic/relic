#include "inference_command.h"
#include "../server.h"
#include <cstdio>

int InferenceCommand::execute_interactive(const CliOptions &opt, InferenceEngine &engine)
{
    run_relic_interactive(engine, opt.n_tokens, opt.temperature, opt.top_k);
    return 0;
}

int InferenceCommand::execute_generate(const CliOptions &opt, InferenceEngine &engine)
{
    fprintf(stdout, "\n=== Relic Inference ===\n");
    fprintf(stdout, "Prompt: %s\n", opt.prompt.c_str());
    fprintf(stdout, "Generating %d tokens...\n\n", opt.n_tokens);

    std::string result = engine.generate(opt.prompt, opt.n_tokens, opt.temperature, opt.top_k);

    fprintf(stdout, "\n=== Done ===\n");
    return 0;
}
