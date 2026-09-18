#pragma once

#include "inference.h"
#include <string>

// Interactive REPL session that keeps the model resident in GPU VRAM
void run_relic_interactive(InferenceEngine &engine, int default_max_tokens, float default_temp, int default_top_k);

// Persistent HTTP / TCP Server daemon that keeps the model resident in GPU VRAM
bool run_relic_server(InferenceEngine &engine, int port, int default_max_tokens, float default_temp, int default_top_k);

// Client to query a running persistent Relic server
int run_relic_client(int port, const std::string &prompt, int n_tokens, float temp, int top_k);
