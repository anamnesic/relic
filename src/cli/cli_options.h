#pragma once

#include <string>
#include <vector>

struct CliOptions
{
    std::string model_path;
    std::string prompt = "Once upon a time";
    std::string bench_json_path;
    std::string bench_csv_path;
    int n_tokens = 256;
    float temperature = 0.8f;
    int top_k = 40;
    int platform_idx = -1;
    int device_idx = 0;
    bool list_devices = false;
    bool run_profile = false;
    bool run_bench = false;
    bool cpu_only = false;
    bool speculative = false;
    int speculative_ngram = 3;
    int speculative_draft_max = 3;
    int max_seq_len = 2048;
    int vram_budget_mb = 0;
    bool run_sweep = false;
    bool run_ablation = false;
    bool run_cliff_analysis = false;
    int bench_runs = 5;
    int bench_warmup = 2;
    bool interactive = false;
    bool server_mode = false;
    int server_port = 8080;
    bool client_mode = false;
    int client_port = 8080;
    bool show_help = false;
};

void print_usage(const char *prog);
CliOptions parse_cli_options(int argc, char **argv);
