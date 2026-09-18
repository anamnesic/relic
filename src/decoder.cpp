#include "decoder.h"
#include "adapters/llama_decoder_adapter.h"
#include "adapters/qwen35_decoder_adapter.h"

std::unique_ptr<ArchitectureDecoder> create_decoder(const ArchitectureSpec &spec, OpenClBackend *backend)
{
    if (spec.kind == ArchitectureKind::Llama)
    {
        return std::make_unique<LlamaDecoderAdapter>(backend);
    }
    if (spec.kind == ArchitectureKind::Qwen35)
    {
        return std::make_unique<Qwen35DecoderAdapter>(backend);
    }
    return nullptr;
}
