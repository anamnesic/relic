#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct Tokenizer {
    struct Token {
        std::string text;
        float score = 0.0f;
    };

    std::vector<Token> vocab;
    std::unordered_map<std::string, int> token_to_id;
    int bos_id = 1;
    int eos_id = 2;
    int pad_id = 0;
    bool add_bos = false;

    // BPE merge data
    struct Merge {
        int left_id;
        int right_id;
        int new_id;
    };
    std::vector<Merge> merges;
    bool is_bpe = false;

    bool load_from_gguf(class GgufReader &reader);
    std::vector<int> encode(const std::string &text, int max_len = 512) const;
    std::string decode(const std::vector<int> &tokens) const;

    int vocab_size() const { return (int)vocab.size(); }

private:
    std::vector<int> encode_bpe(const std::string &text, int max_len) const;
    std::vector<int> encode_fallback(const std::string &text, int max_len) const;
};
