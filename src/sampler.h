#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <queue>

class Sampler {
public:
    static int sample(const float *logits, size_t vocab_size, float temperature = 0.0f, int top_k = 40, float top_p = 0.9f) {
        if (!logits || vocab_size == 0) return 0;

        // 1. Greedy sampling (Temperature <= 0)
        if (temperature <= 0.001f) {
            int max_idx = 0;
            float max_val = logits[0];
            for (size_t i = 1; i < vocab_size; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    max_idx = (int)i;
                }
            }
            return max_idx;
        }

        // 2. Temperature scaling + Min-Heap Top-K selection
        struct TokenLogit {
            int token_id;
            float logit;
            bool operator>(const TokenLogit &other) const {
                return logit > other.logit;
            }
        };

        std::priority_queue<TokenLogit, std::vector<TokenLogit>, std::greater<TokenLogit>> min_heap;
        int k = (top_k > 0) ? std::min<int>(top_k, (int)vocab_size) : (int)vocab_size;

        for (size_t i = 0; i < vocab_size; i++) {
            float scaled_logit = logits[i] / temperature;
            if ((int)min_heap.size() < k) {
                min_heap.push({(int)i, scaled_logit});
            } else if (scaled_logit > min_heap.top().logit) {
                min_heap.pop();
                min_heap.push({(int)i, scaled_logit});
            }
        }

        std::vector<TokenLogit> candidates;
        candidates.reserve(min_heap.size());
        while (!min_heap.empty()) {
            candidates.push_back(min_heap.top());
            min_heap.pop();
        }
        std::reverse(candidates.begin(), candidates.end()); // Descending order

        // 3. Softmax computation over candidates
        float max_l = candidates[0].logit;
        float sum_exp = 0.0f;
        std::vector<float> probs(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) {
            probs[i] = std::exp(candidates[i].logit - max_l);
            sum_exp += probs[i];
        }
        for (size_t i = 0; i < candidates.size(); i++) {
            probs[i] /= sum_exp;
        }

        // 4. Top-P (Nucleus) Truncation
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum_prob = 0.0f;
            size_t cutoff = candidates.size();
            for (size_t i = 0; i < candidates.size(); i++) {
                cum_prob += probs[i];
                if (cum_prob >= top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            candidates.resize(cutoff);
            probs.resize(cutoff);
            // Renormalize
            float sub_sum = 0.0f;
            for (float p : probs) sub_sum += p;
            if (sub_sum > 0.0f) {
                for (float &p : probs) p /= sub_sum;
            }
        }

        // 5. Categorical Sampling
        static thread_local std::mt19937 gen(1337);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(gen);
        float acc = 0.0f;
        for (size_t i = 0; i < candidates.size(); i++) {
            acc += probs[i];
            if (r <= acc) {
                return candidates[i].token_id;
            }
        }

        return candidates[0].token_id;
    }
};
