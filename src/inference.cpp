#include "inference.h"
#include "sampler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

bool InferenceEngine::init(LlamaModel *m, Tokenizer *tok, OpenClBackend *backend,
                           int64_t max_seq_limit, const ExecutionPlan *exec_plan)
{
    if (!m)
        return false;
    model = m;
    tokenizer = tok;
    cl = backend;
    plan = exec_plan;
    max_seq_len = max_seq_limit;
    n_past = 0;

    decoder = create_decoder(model->architecture, cl);
    if (!decoder)
    {
        fprintf(stderr, "No decoder adapter available for architecture: %s\n",
                model->architecture.name.c_str());
        return false;
    }

    if (!decoder->init(model->architecture, max_seq_len, plan))
    {
        fprintf(stderr, "Failed to initialize decoder for architecture: %s\n",
                model->architecture.name.c_str());
        return false;
    }

    decoder->warm_up(*model);

    return true;
}

void InferenceEngine::free_buffers()
{
    if (decoder)
        decoder->reset();
    n_past = 0;
}

int InferenceEngine::forward(int token_id, float *logits)
{
    if (!decoder || !model)
        return -1;
    int res = decoder->forward(*model, token_id, n_past, logits);
    if (res == 0)
    {
        n_past++;
    }
    return res;
}

std::vector<int> InferenceEngine::find_prompt_lookup_draft(const std::vector<int> &tokens, int ngram_len, int max_draft)
{
    std::vector<int> draft;
    if ((int)tokens.size() < ngram_len + 1)
        return draft;

    int n = (int)tokens.size();
    const int *cur_ngram = &tokens[n - ngram_len];

    // Search backwards in the context history for a matching n-gram
    for (int i = n - ngram_len - 1; i >= ngram_len; i--)
    {
        bool match = true;
        for (int k = 0; k < ngram_len; k++)
        {
            if (tokens[i - ngram_len + k] != cur_ngram[k])
            {
                match = false;
                break;
            }
        }
        if (match)
        {
            for (int d = 0; d < max_draft && (i + d) < (n - ngram_len); d++)
            {
                draft.push_back(tokens[i + d]);
            }
            break;
        }
    }
    return draft;
}

std::string InferenceEngine::generate(const std::string &prompt, int max_tokens,
                                      float temperature, int top_k)
{
    if (!tokenizer || !model || !decoder)
        return "";

    std::vector<int> input_tokens = tokenizer->encode(prompt);
    if (input_tokens.empty())
    {
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

    for (size_t i = 0; i < input_tokens.size(); i++)
    {
        float *l_ptr = (i + 1 == input_tokens.size()) ? logits.data() : nullptr;
        if (forward(input_tokens[i], l_ptr) != 0)
        {
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
    if (enable_speculative)
    {
        fprintf(stdout, "[Speculation] Prompt-Lookup Speculative Decoding Active (N-gram=%d, DraftMax=%d)\n",
                speculative_ngram, speculative_max_draft);
    }
    fprintf(stdout, "Output: ");
    fflush(stdout);

    int generated_count = 0;
    int accepted_spec_tokens = 0;
    int last_token = 0;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    while (generated_count < max_tokens)
    {
        last_token = Sampler::sample(logits.data(), (size_t)model->n_vocab, temperature, top_k, 0.9f);

        if (last_token == tokenizer->eos_id)
            break;
        std::string piece = tokenizer->decode({last_token});
        output += piece;
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);
        all_tokens.push_back(last_token);
        generated_count++;

        if (forward(last_token, logits.data()) != 0)
        {
            fprintf(stderr, "\nForward pass failed during token generation\n");
            break;
        }

        // Batched speculative verification pipeline with state checkpoint & rollback
        if (enable_speculative && generated_count < max_tokens)
        {
            std::vector<int> drafts = find_prompt_lookup_draft(all_tokens, speculative_ngram, speculative_max_draft);
            if (!drafts.empty())
            {
                // Step 1: Pre-draft logits (produced by forward(last_token)) verify draft 0
                int target_d0 = Sampler::sample(logits.data(), (size_t)model->n_vocab, temperature, top_k, 0.9f);
                if (target_d0 == drafts[0])
                {
                    size_t num_draft = drafts.size();
                    std::vector<float> batched_logits(num_draft * (size_t)model->n_vocab);

                    // Checkpoint recurrent state & KV cache before speculative exploration
                    decoder->save_state_checkpoint();

                    if (decoder->forward_batch(*model, drafts, n_past, batched_logits.data()) == 0)
                    {
                        int accepted_count = 1; // drafts[0] verified by pre-draft logits
                        for (size_t d = 0; d + 1 < num_draft; d++)
                        {
                            float *pred_logits = batched_logits.data() + d * (size_t)model->n_vocab;
                            int target_pred = Sampler::sample(pred_logits, (size_t)model->n_vocab, temperature, top_k, 0.9f);
                            if (target_pred == drafts[d + 1])
                            {
                                accepted_count++;
                            }
                            else
                            {
                                break;
                            }
                        }

                        // If fewer than all drafts accepted, restore checkpoint and commit ONLY accepted tokens
                        if (accepted_count < (int)num_draft)
                        {
                            decoder->restore_state_checkpoint();
                            for (int i = 0; i < accepted_count; i++)
                            {
                                decoder->forward(*model, drafts[i], n_past + (int64_t)i, nullptr);
                            }
                        }

                        for (int i = 0; i < accepted_count; i++)
                        {
                            if (generated_count >= max_tokens)
                                break;
                            accepted_spec_tokens++;
                            n_past++;
                            std::string draft_piece = tokenizer->decode({drafts[i]});
                            output += draft_piece;
                            fprintf(stdout, "%s", draft_piece.c_str());
                            fflush(stdout);
                            all_tokens.push_back(drafts[i]);
                            generated_count++;
                            float *last_acc_logits = batched_logits.data() + (size_t)i * (size_t)model->n_vocab;
                            memcpy(logits.data(), last_acc_logits, (size_t)model->n_vocab * sizeof(float));
                        }
                    }
                    else
                    {
                        decoder->restore_state_checkpoint();
                    }
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
    if (enable_speculative)
    {
        fprintf(stdout, " [Accepted Speculative: %d tokens]", accepted_spec_tokens);
    }
    fprintf(stdout, "\n[Benchmark] Total runtime: %.2f ms\n", total_sec * 1000.0);

    return output;
}
