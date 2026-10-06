# AGENTS.md — Relic

## Regra obrigatória: pesquisar antes de corrigir

Ao encontrar **qualquer erro ou bug**, antes de alterar o código:

1. **Pesquisar na web** (web search / webfetch) por uma solução já existente:
   - issue/PR correspondente no llama.cpp, GGML, Khronos OpenCL ou no próprio relic;
   - discussão/stack overflow sobre o mesmo sintoma;
   - a especificação oficial (GGUF, GGML quants, OpenCL 1.2 spec) quando se tratar de formato ou comportamento.
2. Comparar a solução encontrada com o código atual e **citar a fonte** no commit/comentário.
3. Só então aplicar a correção — nunca "chutar" uma correção só por parecer plausível.

Isso vale para erros de build, erros de runtime, NaN/garbage em inferência, divergência numérica,
crashes e erros de compilação. Se a pesquisa não achar nada, documentar isso antes de implementar.

## Comandos

```bash
# build
cmake --build build -j 4

# inferência (prompt + nº de tokens + temperatura; -t 0 = greedy)
./build/relic -m <model.gguf> -p "The capital of France is" -n 16 -t 0

# ferramentas
./build/relic_dump_model <model.gguf>
./build/relic_test
```

### Variáveis de depuração

| Variável | Efeito |
| :-- | :-- |
| `RELIC_DEBUG_TOKENS` | imprime os IDs de tokens do prompt |
| `RELIC_DEBUG_STATE` | imprime embedding/estado da camada por passo |
| `RELIC_DEBUG_PLACEMENT` | imprime onde cada tensor foi residente (VRAM/CPU) |
| `RELIC_DEBUG_DISPATCH` | imprime cada dispatch do `dispatch_gemv` |

## Referência de validação

- Ground truth: `llama.cpp` compilado em `/tmp/opencode/llama.cpp` (`build/bin/llama-cli`, `-p ... -n 16 --temp 0 --no-jinja -st --simple-io`).
- Dump tensores por camada: `build/bin/llama-eval-callback -m <model> -p "The" -ngl 0` (imprime nome, `sum` e primeiros/últimos valores de cada tensor — use para comparar estados camada a camada).
- Metadados GGUF: `python3 /tmp/opencode/ggufmeta.py <model.gguf> [substituto-de-chave]`.

## Convenções

- `kernels/kernels.cl` é fonte: após qualquer edição rode o gerador para regenerar `src/embedded_kernels.h`.
- Nunca alterar: `~/.nvm`, `~/.local/share/opencode`, `~/.config/opencode`, `~/.cache/opencode`, `~/.rustup`, `.gemini`.
- Espaço: `/tmp/opencode` para trabalho pesado; `/home` é pequeno (~1 GB livre) — nada de `node_modules`/caches lá.
