#include "server_command.h"
#include "../server.h"

int ServerCommand::execute_server(const CliOptions &opt, InferenceEngine &engine)
{
    run_relic_server(engine, opt.server_port, opt.n_tokens, opt.temperature, opt.top_k);
    return 0;
}

int ServerCommand::execute_client(const CliOptions &opt)
{
    return run_relic_client(opt.client_port, opt.prompt, opt.n_tokens, opt.temperature, opt.top_k);
}
