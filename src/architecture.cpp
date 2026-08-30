#include "architecture.h"

namespace {

class LlamaArchitecture final : public ModelArchitecture {
public:
    bool configure(const GgufReader &reader, ArchitectureSpec &spec, std::string &error) const override {
        spec.kind = ArchitectureKind::Llama;
        spec.name = "llama";
        spec.n_vocab = reader.get_metadata<int64_t>("llama.vocab_size",
            reader.get_metadata<int64_t>("tokenizer.ggml.vocab_size", 32000));
        spec.n_embd = reader.get_metadata<int64_t>("llama.embedding_length",
            reader.get_metadata<int64_t>("llama.d_model", reader.get_metadata<int64_t>("llama.n_embd", 4096)));
        spec.n_head = reader.get_metadata<int64_t>("llama.attention.head_count",
            reader.get_metadata<int64_t>("llama.n_head", 32));
        spec.n_head_kv = reader.get_metadata<int64_t>("llama.attention.head_count_kv",
            reader.get_metadata<int64_t>("llama.n_head_kv", spec.n_head));
        spec.n_layer = reader.get_metadata<int64_t>("llama.block_count",
            reader.get_metadata<int64_t>("llama.n_layer", 32));
        spec.n_ff = reader.get_metadata<int64_t>("llama.feed_forward_length",
            reader.get_metadata<int64_t>("llama.n_ff", 4 * spec.n_embd));
        spec.norm_eps = reader.get_metadata<float>("llama.attention.layer_norm_rms_epsilon",
            reader.get_metadata<float>("llama.norm_eps", 1e-5f));
        spec.rope_freq_base = reader.get_metadata<float>("llama.rope.freq_base", 10000.0f);
        if (spec.n_embd <= 0 || spec.n_head <= 0 || spec.n_embd % spec.n_head != 0) {
            error = "invalid llama dimensions";
            return false;
        }
        return true;
    }
};

class Qwen35Architecture final : public ModelArchitecture {
public:
    bool configure(const GgufReader &reader, ArchitectureSpec &spec, std::string &error) const override {
        spec.kind = ArchitectureKind::Qwen35;
        spec.name = "qwen35";
        const auto embedding = reader.tensors.find("token_embd.weight");
        spec.n_vocab = embedding != reader.tensors.end() && embedding->second.dims.size() >= 2
            ? embedding->second.dims[1]
            : reader.get_metadata<int64_t>("qwen35.vocab_size", 0);
        spec.n_embd = reader.get_metadata<int64_t>("qwen35.embedding_length", 0);
        spec.n_layer = reader.get_metadata<int64_t>("qwen35.block_count", 0);
        spec.n_head = reader.get_metadata<int64_t>("qwen35.attention.head_count", 0);
        const auto key = reader.tensors.find("blk.3.attn_k.weight");
        const int64_t head_dim = reader.get_metadata<int64_t>("qwen35.attention.key_length", 0);
        spec.n_head_kv = key != reader.tensors.end() && key->second.dims.size() >= 2 && head_dim > 0
            ? key->second.dims[1] / head_dim
            : reader.get_metadata<int64_t>("qwen35.attention.head_count_kv", 0);
        spec.n_ff = reader.get_metadata<int64_t>("qwen35.feed_forward_length", 0);
        spec.norm_eps = reader.get_metadata<float>("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
        spec.rope_freq_base = reader.get_metadata<float>("qwen35.rope.freq_base", 10000000.0f);
        spec.full_attention_interval = reader.get_metadata<int64_t>("qwen35.full_attention_interval", 4);
        spec.linear_conv_kernel = reader.get_metadata<int64_t>("qwen35.ssm.conv_kernel", 0);
        spec.linear_inner_size = reader.get_metadata<int64_t>("qwen35.ssm.inner_size", 0);
        spec.linear_key_head_dim = reader.get_metadata<int64_t>("qwen35.ssm.state_size", 0);
        spec.linear_value_head_dim = spec.linear_key_head_dim;
        spec.linear_key_heads = reader.get_metadata<int64_t>("qwen35.ssm.group_count", 0);
        spec.linear_value_heads = reader.get_metadata<int64_t>("qwen35.ssm.time_step_rank", 0);
        if (spec.n_vocab <= 0 || spec.n_embd <= 0 || spec.n_layer <= 0 || spec.n_head <= 0 ||
             spec.n_head_kv <= 0 || spec.n_ff <= 0 || spec.linear_conv_kernel <= 0 || spec.linear_inner_size <= 0 ||
            spec.linear_key_head_dim <= 0 || spec.linear_key_heads <= 0 || spec.linear_value_heads <= 0) {
            error = "incomplete qwen35 GGUF metadata";
            return false;
        }
        if (spec.n_embd % spec.n_head != 0 || spec.n_head % spec.n_head_kv != 0 ||
            spec.linear_inner_size % spec.linear_value_heads != 0) {
            error = "invalid qwen35 attention dimensions";
            return false;
        }
        return true;
    }
};

const LlamaArchitecture llama_architecture;
const Qwen35Architecture qwen35_architecture;

} // namespace

const ModelArchitecture *find_architecture(const std::string &name) {
    if (name == "llama") return &llama_architecture;
    if (name == "qwen35") return &qwen35_architecture;
    return nullptr;
}
