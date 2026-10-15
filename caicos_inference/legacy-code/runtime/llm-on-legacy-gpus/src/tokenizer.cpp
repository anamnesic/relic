#include "tokenizer.h"
#include "gguf_reader.h"
#include <cstdio>
#include <algorithm>
#include <sstream>

bool Tokenizer::load_from_gguf(GgufReader &reader) {
    std::string model_type = reader.get_metadata<std::string>("tokenizer.ggml.model", "");
    is_bpe = (model_type == "gpt2" || model_type == "bpe");

    fprintf(stdout, "Tokenizer model: %s\n", model_type.c_str());

    // Read vocabulary
    int64_t n_vocab = reader.get_metadata<uint32_t>("tokenizer.ggml.vocab_size",
                       reader.get_metadata<uint32_t>("llama.vocab_size", 32000));

    vocab.clear();
    vocab.reserve((size_t)n_vocab);

    for (int64_t i = 0; i < n_vocab; i++) {
        std::string key = "tokenizer.ggml.tokens_" + std::to_string(i);
        std::string token_str = reader.get_metadata<std::string>(key, "");
        if (token_str.empty()) {
            // Try alternate key format
            char buf[64];
            snprintf(buf, sizeof(buf), "tokenizer.ggml.tokens[%lld]", (long long)i);
            token_str = reader.get_metadata<std::string>(buf, "");
        }

        Token t;
        t.text = token_str;
        t.score = 0.0f;

        std::string score_key = "tokenizer.ggml.scores_" + std::to_string(i);
        t.score = reader.get_metadata<float>(score_key, 0.0f);

        vocab.push_back(t);
        if (!token_str.empty()) {
            token_to_id[token_str] = (int)i;
        }
    }

    // Read special token IDs
    bos_id = (int)reader.get_metadata<uint32_t>("tokenizer.ggml.bos_token_id", 1);
    eos_id = (int)reader.get_metadata<uint32_t>("tokenizer.ggml.eos_token_id", 2);
    pad_id = (int)reader.get_metadata<uint32_t>("tokenizer.ggml.pad_token_id", 0);

    fprintf(stdout, "  vocab_size=%zu bos=%d eos=%d\n", vocab.size(), bos_id, eos_id);

    // For BPE, read merges
    if (is_bpe) {
        // GGUF stores merges as an array of strings under
        // "tokenizer.ggml.merges" key, which the reader expands to
        // individual "tokenizer.ggml.merges_0", "_1", etc.
        // Count how many merge entries exist
        int64_t n_merges = 0;
        for (int64_t i = 0; ; i++) {
            std::string key = "tokenizer.ggml.merges_" + std::to_string(i);
            if (reader.metadata_str.find(key) != reader.metadata_str.end()) {
                n_merges = i + 1;
            } else {
                break;
            }
        }

        if (n_merges > 0) {
            merges.reserve((size_t)n_merges);
            for (int64_t i = 0; i < n_merges; i++) {
                std::string key = "tokenizer.ggml.merges_" + std::to_string(i);
                std::string merge_str = reader.get_metadata<std::string>(key, "");
                if (merge_str.empty()) continue;

                auto space = merge_str.find(' ');
                if (space != std::string::npos) {
                    std::string left = merge_str.substr(0, space);
                    std::string right = merge_str.substr(space + 1);
                    auto lit = token_to_id.find(left);
                    auto rit = token_to_id.find(right);
                    if (lit != token_to_id.end() && rit != token_to_id.end()) {
                        Merge m;
                        m.left_id = lit->second;
                        m.right_id = rit->second;
                        auto merged_it = token_to_id.find(left + right);
                        if (merged_it != token_to_id.end()) {
                            m.new_id = merged_it->second;
                            merges.push_back(m);
                        }
                    }
                }
            }
        }
        fprintf(stdout, "  merges=%zu\n", merges.size());
    }

    return true;
}

static std::vector<std::string> split_utf8_bytes(const std::string &text) {
    std::vector<std::string> result;
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        int len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        result.push_back(text.substr(i, len));
        i += len;
    }
    return result;
}

// GPT-2 byte-to-Unicode mapping table
// Bytes in printable ranges map to themselves; others map to Unicode 256+
static std::vector<uint32_t> build_gpt2_byte_table() {
    std::vector<uint32_t> table(256);
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            table[b] = (uint32_t)b;
        } else {
            table[b] = 256 + n;
            n++;
        }
    }
    return table;
}

// GPT-2 pre-tokenization: split text into "words" with preceding spaces attached
static std::vector<std::string> gpt2_pretokenize(const std::string &text) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        // Collect spaces (including newlines, tabs)
        size_t start = i;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                                    text[i] == '\n' || text[i] == '\r')) {
            i++;
        }
        std::string spaces = text.substr(start, i - start);

        // Collect non-space content
        start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t' &&
                                    text[i] != '\n' && text[i] != '\r') {
            i++;
        }
        std::string content = text.substr(start, i - start);

        if (!content.empty()) {
            // Attach preceding spaces to the content
            words.push_back(spaces + content);
        } else if (!spaces.empty()) {
            // Only spaces at end of text
            // Skip trailing spaces (GPT-2 ignores them)
        }
    }
    return words;
}

std::vector<int> Tokenizer::encode_bpe(const std::string &text, int max_len) const {
    static const std::vector<uint32_t> byte_table = build_gpt2_byte_table();

    // Pre-tokenize into words with attached spaces
    std::vector<std::string> word_chunks = gpt2_pretokenize(text);

    std::vector<int> ids;
    for (const auto &word : word_chunks) {
        // Convert each byte to its GPT-2 Unicode representation and look up in vocab
        std::vector<int> word_ids;
        for (unsigned char b : word) {
            uint32_t cp = byte_table[b];
            // Convert Unicode code point to UTF-8 string
            std::string utf8_char;
            if (cp < 0x80) {
                utf8_char += (char)cp;
            } else if (cp < 0x800) {
                utf8_char += (char)(0xC0 | (cp >> 6));
                utf8_char += (char)(0x80 | (cp & 0x3F));
            } else {
                utf8_char += (char)(0xE0 | (cp >> 12));
                utf8_char += (char)(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += (char)(0x80 | (cp & 0x3F));
            }
            auto it = token_to_id.find(utf8_char);
            if (it != token_to_id.end()) {
                word_ids.push_back(it->second);
            }
        }

        // Apply BPE merges within this word only
        bool changed = true;
        while (changed && (int)ids.size() + (int)word_ids.size() < max_len) {
            changed = false;
            int best_rank = 1000000;
            int best_pos = -1;

            for (size_t i = 0; i + 1 < word_ids.size(); i++) {
                for (size_t m = 0; m < merges.size(); m++) {
                    if (merges[m].left_id == word_ids[i] && merges[m].right_id == word_ids[i + 1]) {
                        if ((int)m < best_rank) {
                            best_rank = (int)m;
                            best_pos = (int)i;
                        }
                        break;
                    }
                }
            }

            if (best_pos >= 0) {
                word_ids[best_pos] = merges[best_rank].new_id;
                word_ids.erase(word_ids.begin() + best_pos + 1);
                changed = true;
            }
        }

        // Append this word's token IDs
        for (int id : word_ids) ids.push_back(id);
        if ((int)ids.size() >= max_len) break;
    }

    if ((int)ids.size() > max_len) {
        ids.resize(max_len);
    }

    return ids;
}

std::vector<int> Tokenizer::encode_fallback(const std::string &text, int max_len) const {
    std::vector<int> result;

    // Simple: split by words and look up in vocab
    std::string current;
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        int ch_len = 1;
        if ((c & 0x80) == 0) ch_len = 1;
        else if ((c & 0xE0) == 0xC0) ch_len = 2;
        else if ((c & 0xF0) == 0xE0) ch_len = 3;
        else if ((c & 0xF8) == 0xF0) ch_len = 4;

        std::string ch = text.substr(i, (size_t)ch_len);

        // Try to find the current token + new char
        std::string candidate = current + ch;
        auto it = token_to_id.find(candidate);
        if (it != token_to_id.end()) {
            current = candidate;
        } else {
            if (!current.empty()) {
                auto cit = token_to_id.find(current);
                if (cit != token_to_id.end()) {
                    result.push_back(cit->second);
                } else {
                    // Fallback: add byte tokens
                    for (unsigned char bc : current) {
                        std::string bs(1, (char)bc);
                        auto bit = token_to_id.find(bs);
                        if (bit != token_to_id.end())
                            result.push_back(bit->second);
                    }
                }
            }
            // Start new token
            auto nit = token_to_id.find(ch);
            if (nit != token_to_id.end()) {
                current = ch;
            } else {
                // Unknown character - try byte-level
                for (unsigned char bc : ch) {
                    std::string bs(1, (char)bc);
                    auto bit = token_to_id.find(bs);
                    if (bit != token_to_id.end())
                        result.push_back(bit->second);
                }
                current.clear();
            }
        }

        i += (size_t)ch_len;
        if ((int)result.size() >= max_len - 1) break;
    }

    // Flush last token
    if (!current.empty()) {
        auto cit = token_to_id.find(current);
        if (cit != token_to_id.end()) {
            result.push_back(cit->second);
        } else {
            for (unsigned char bc : current) {
                std::string bs(1, (char)bc);
                auto bit = token_to_id.find(bs);
                if (bit != token_to_id.end())
                    result.push_back(bit->second);
            }
        }
    }

    return result;
}

std::vector<int> Tokenizer::encode(const std::string &text, int max_len) const {
    if (is_bpe && !merges.empty()) {
        return encode_bpe(text, max_len);
    }
    return encode_fallback(text, max_len);
}

std::string Tokenizer::decode(const std::vector<int> &tokens) const {
    std::string result;
    for (int id : tokens) {
        if (id >= 0 && id < (int)vocab.size()) {
            result += vocab[id].text;
        }
    }
    return result;
}
