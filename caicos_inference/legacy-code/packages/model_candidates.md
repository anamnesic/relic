# Modelos candidatos para a Caicos XT

Baseado na VRAM reportada de 1 GB e na necessidade de deixar folga para runtime, contexto e cache, estes são os candidatos mais práticos que encontrei.

## Faixa mais segura

- [smollm2:135m](https://ollama.com/library/smollm2:135m) - 135M parâmetros, cerca de 271 MB no Ollama. É o candidato mais folgado da lista e deve ser o primeiro teste para GPU-only.
- [gemma3:270m](https://ollama.com/library/gemma3:270m) - 268M parâmetros, cerca de 292 MB no Ollama. Também é leve e já vem com contexto de 32K.
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) - 494M parâmetros, cerca de 398 MB no Ollama. Boa opção para texto e código leve.

## Faixa ainda plausível

- [smollm2:360m](https://ollama.com/library/smollm2:360m) - 362M parâmetros, cerca de 726 MB no Ollama. Ainda pode caber, mas já deixa bem menos espaço para cache e overhead.
- [tinyllama:1.1b](https://ollama.com/library/tinyllama) - 1.1B parâmetros, cerca de 638 MB no Ollama. É pequeno em peso bruto, mas o contexto e o overhead podem apertar mais do que os números sugerem.

## Limite que eu evitaria como alvo principal

- [gemma3:1b](https://ollama.com/library/gemma3:1b) - 1.0B parâmetros, cerca de 815 MB no Ollama. Já carregou em CPU-only aqui; na Caicos XT ele fica perto demais do teto para ser uma aposta segura.

## Critério prático

Para essa placa, modelos abaixo de 400 MB continuam sendo os unicos candidatos plausiveis em memoria bruta. Mesmo assim, a rota `llama.cpp` OpenCL esta bloqueada antes do tamanho do modelo por falta de recursos do device: OpenCL C 1.2 e ausencia de `cl_khr_fp16`.

## Status nesta maquina

Testes locais com `gemma3:270m`, `qwen2.5:0.5b`, `smollm2:135m`, `smollm2:360m`, `tinyllama` e `gemma3:1b` mostraram 100% CPU no `ollama ps`. A verificacao direta com `clinfo` e `llama.cpp` indica que, neste hardware, a Caicos XT nao atende os requisitos do backend OpenCL atual para GPU-only.
