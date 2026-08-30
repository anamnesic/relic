#include "inference.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

bool InferenceEngine::init(LlamaModel *m, Tokenizer *tok, OpenClBackend *backend,
                            int64_t max_seq_limit) {
    if (!m) return false;
    model = m;
    tokenizer = tok;
    cl = backend;
    max_seq_len = max_seq_limit;
    n_past = 0;

    decoder = create_decoder(model->architecture, cl);
    if (!decoder) {
        fprintf(stderr, "No decoder adapter available for architecture: %s\n",
                model->architecture.name.c_str());
        return false;
    }

    if (!decoder->init(model->architecture, max_seq_len)) {
        fprintf(stderr, "Failed to initialize decoder for architecture: %s\n",
                model->architecture.name.c_str());
        return false;
    }

    return true;
}

void InferenceEngine::free_buffers() {
    if (decoder) decoder->reset();
    n_past = 0;
}

int InferenceEngine::forward(int token_id, float *logits) {
    if (!decoder || !model) return -1;
    int res = decoder->forward(*model, token_id, n_past, logits);
    if (res == 0) {
        n_past++;
    }
    return res;
}

std::string InferenceEngine::generate(const std::string &prompt, int max_tokens,
                                      float temperature, int top_k) {
    if (!tokenizer || !model || !decoder) return "";

    std::vector<int> input_tokens = tokenizer->encode(prompt);
    if (input_tokens.empty()) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        return "";
    }

    std::vector<float> logits((size_t)model->n_vocab, 0.0f);
    std::string output;
    std::mt19937 rng(42);

    auto t_start = std::chrono::high_resolution_clock::now();

    fprintf(stdout, "Prompt tokens: %zu\n", input_tokens.size());
    fflush(stdout);

    for (int tok : input_tokens) {
        if (forward(tok, logits.data()) != 0) {
            fprintf(stderr, "Forward pass failed during prompt processing\n");
            fflush(stderr);
            return output;
        }
        fprintf(stdout, ".");
        fflush(stdout);
    }

    auto t_prompt_done = std::chrono::high_resolution_clock::now();
    double prompt_sec = std::chrono::duration<double>(t_prompt_done - t_start).count();
    double prompt_speed = (double)input_tokens.size() / (prompt_sec > 0 ? prompt_sec : 1e-6);

    fprintf(stdout, "\n[Benchmark] Prompt: %zu tokens in %.2f ms (%.2f tok/s)\n",
            input_tokens.size(), prompt_sec * 1000.0, prompt_speed);
    fprintf(stdout, "Output: ");
    fflush(stdout);

    int generated_count = 0;
    int last_token = 0;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_tokens; i++) {
        if (temperature < 0.01f) {
            last_token = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        } else {
            for (auto &l : logits) l /= temperature;
            if (top_k > 0 && top_k < (int)logits.size()) {
                std::vector<std::pair<float, int>> scored;
                for (int j = 0; j < (int)logits.size(); j++)
                    scored.emplace_back(logits[j], j);
                std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                                  [](auto &a, auto &b) { return a.first > b.first; });
                float threshold = scored[top_k - 1].first;
                for (int j = 0; j < (int)logits.size(); j++)
                    if (logits[j] < threshold) logits[j] = -INFINITY;
            }
            float maxv = logits[0];
            for (auto &l : logits) if (l > maxv) maxv = l;
            float sum = 0.0f;
            for (auto &l : logits) { l = expf(l - maxv); sum += l; }
            float inv = 1.0f / sum;
            for (auto &l : logits) l *= inv;
            float r = (float)rng() / (float)rng.max();
            float cum = 0.0f;
            last_token = (int)logits.size() - 1;
            for (int j = 0; j < (int)logits.size(); j++) {
                cum += logits[j];
                if (r < cum) { last_token = j; break; }
            }
        }

        if (last_token == tokenizer->eos_id) break;
        std::string piece = tokenizer->decode({last_token});
        output += piece;
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);
        generated_count++;

        if (forward(last_token, logits.data()) != 0) {
            fprintf(stderr, "\nForward pass failed during token generation\n");
            break;
        }
    }

    auto t_gen_done = std::chrono::high_resolution_clock::now();
    double gen_sec = std::chrono::duration<double>(t_gen_done - t_gen_start).count();
    double gen_speed = (double)generated_count / (gen_sec > 0 ? gen_sec : 1e-6);
    double total_sec = std::chrono::duration<double>(t_gen_done - t_start).count();

    fprintf(stdout, "\n\n[Benchmark] Generation: %d tokens in %.2f ms (%.2f tok/s)\n",
            generated_count, gen_sec * 1000.0, gen_speed);
    fprintf(stdout, "[Benchmark] Total runtime: %.2f ms\n", total_sec * 1000.0);

    return output;
}
