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

std::vector<int> InferenceEngine::find_prompt_lookup_draft(const std::vector<int> &tokens, int ngram_len, int max_draft) {
    std::vector<int> draft;
    if ((int)tokens.size() < ngram_len + 1) return draft;

    int n = (int)tokens.size();
    const int *cur_ngram = &tokens[n - ngram_len];

    // Search backwards in the context history for a matching n-gram
    for (int i = n - ngram_len - 1; i >= ngram_len; i--) {
        bool match = true;
        for (int k = 0; k < ngram_len; k++) {
            if (tokens[i - ngram_len + k] != cur_ngram[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            for (int d = 0; d < max_draft && (i + d) < (n - ngram_len); d++) {
                draft.push_back(tokens[i + d]);
            }
            break;
        }
    }
    return draft;
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
    std::vector<int> all_tokens = input_tokens;
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
    if (enable_speculative) {
        fprintf(stdout, "[Speculation] Prompt-Lookup Speculative Decoding Active (N-gram=%d, DraftMax=%d)\n",
                speculative_ngram, speculative_max_draft);
    }
    fprintf(stdout, "Output: ");
    fflush(stdout);

    int generated_count = 0;
    int accepted_spec_tokens = 0;
    int last_token = 0;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    auto sample_token = [&](std::vector<float> &l_vec) -> int {
        if (temperature < 0.01f) {
            return (int)(std::max_element(l_vec.begin(), l_vec.end()) - l_vec.begin());
        }
        for (auto &l : l_vec) l /= temperature;
        if (top_k > 0 && top_k < (int)l_vec.size()) {
            std::vector<std::pair<float, int>> scored;
            for (int j = 0; j < (int)l_vec.size(); j++)
                scored.emplace_back(l_vec[j], j);
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                              [](auto &a, auto &b) { return a.first > b.first; });
            float threshold = scored[top_k - 1].first;
            for (int j = 0; j < (int)l_vec.size(); j++)
                if (l_vec[j] < threshold) l_vec[j] = -INFINITY;
        }
        float maxv = l_vec[0];
        for (auto &l : l_vec) if (l > maxv) maxv = l;
        float sum = 0.0f;
        for (auto &l : l_vec) { l = expf(l - maxv); sum += l; }
        float inv = 1.0f / sum;
        for (auto &l : l_vec) l *= inv;
        float r = (float)rng() / (float)rng.max();
        float cum = 0.0f;
        int choice = (int)l_vec.size() - 1;
        for (int j = 0; j < (int)l_vec.size(); j++) {
            cum += l_vec[j];
            if (r < cum) { choice = j; break; }
        }
        return choice;
    };

    while (generated_count < max_tokens) {
        last_token = sample_token(logits);

        if (last_token == tokenizer->eos_id) break;
        std::string piece = tokenizer->decode({last_token});
        output += piece;
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);
        all_tokens.push_back(last_token);
        generated_count++;

        if (forward(last_token, logits.data()) != 0) {
            fprintf(stderr, "\nForward pass failed during token generation\n");
            break;
        }

        // Speculative decoding verification loop
        if (enable_speculative && generated_count < max_tokens) {
            std::vector<int> drafts = find_prompt_lookup_draft(all_tokens, speculative_ngram, speculative_max_draft);
            for (int draft_tok : drafts) {
                if (generated_count >= max_tokens) break;
                int target_pred = sample_token(logits);
                if (target_pred == draft_tok) {
                    // Speculative draft accepted!
                    accepted_spec_tokens++;
                    if (draft_tok == tokenizer->eos_id) break;
                    std::string draft_piece = tokenizer->decode({draft_tok});
                    output += draft_piece;
                    fprintf(stdout, "%s", draft_piece.c_str());
                    fflush(stdout);
                    all_tokens.push_back(draft_tok);
                    generated_count++;
                    if (forward(draft_tok, logits.data()) != 0) break;
                } else {
                    break;
                }
            }
        }
    }

    auto t_gen_done = std::chrono::high_resolution_clock::now();
    double gen_sec = std::chrono::duration<double>(t_gen_done - t_gen_start).count();
    double gen_speed = (double)generated_count / (gen_sec > 0 ? gen_sec : 1e-6);
    double total_sec = std::chrono::duration<double>(t_gen_done - t_start).count();

    fprintf(stdout, "\n\n[Benchmark] Generation: %d tokens in %.2f ms (%.2f tok/s)",
            generated_count, gen_sec * 1000.0, gen_speed);
    if (enable_speculative) {
        fprintf(stdout, " [Accepted Speculative: %d tokens]", accepted_spec_tokens);
    }
    fprintf(stdout, "\n[Benchmark] Total runtime: %.2f ms\n", total_sec * 1000.0);

    return output;
}
