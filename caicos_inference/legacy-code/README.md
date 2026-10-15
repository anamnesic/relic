# legacy-code

Harness local para coding com modelos Ollama, mas sempre centrado na AMD Caicos XT desta maquina, que nao tem suporte moderno.

## O que foi montado

Foi criado um CLI Python simples em `src/caicos_harness` para servir de base do fluxo local:

- `chat`: envia prompts para um modelo local do Ollama;
- `models`: lista os modelos disponíveis no Ollama;
- `doctor`: checa host, modelo padrão e disponibilidade;
- `read`: imprime um arquivo do workspace para alimentar o fluxo de coding.

O pacote usa `OLLAMA_HOST` se estiver definido e cai para `http://localhost:11434`.
Ele também tenta forçar offload para GPU via `options.num_gpu` e falha se o modelo continuar aparecendo como `CPU` no `ollama ps`.

## Como usar

1. Instale Python 3.11 ou mais novo.
2. Inicie o Ollama localmente.
3. No diretório `legacy-code`, rode `pip install -e .`.
4. Use `caicos-harness doctor` para validar o ambiente.

## Runtime base (submodulo)

O runtime `llm-on-legacy-gpus` fica em `runtime/llm-on-legacy-gpus` como submodulo Git.

Para clonar tudo corretamente:

```bash
git submodule update --init --recursive
```

Para atualizar o runtime depois:

```bash
git submodule update --remote --merge
```

## Finalidade

Este diretório documenta o agente customizado usado como base de um fluxo de coding local, no estilo Codex/Claude Code, apoiado por modelos locais via Ollama e guiado pelo limite real da Caicos XT.

## O que o harness cobre

- leitura e edicao orientadas ao codigo do proprio workspace;
- uso de modelos locais via Ollama para tarefas de coding;
- manutencao do contexto da GPU Caicos XT como alvo principal;
- atualizacao sincronizada da documentacao em `README.md` e `packages/`.

## Ferramentas previstas (implementacao)

- Ollama API: `/api/chat`, `/api/tags`, `ollama ps` (status GPU), `options.num_gpu/main_gpu`;
- CLI local: `caicos-harness chat|models|doctor|read` e futuras `runtime|build|bench`;
- Leitura de arquivos: trechos por linha, glob de pastas, lista de diretorios;
- Busca no workspace: termo, regex e arvore;
- Git: `status`, `diff`, `log`, `branch`, `submodule update --init --recursive`;
- Build do runtime: `cmake`, `cmake --build`, scripts em PowerShell;
- OpenCL diagnostics: listagem de plataformas e devices (via `--list-devices` do runtime);
- Benchmarks: tokens/s, tempo por camada, memoria, tamanho de KV cache;
- Tokenizacao/IO: encode/decode via tokenizer do runtime (ou GGUF reader);
- Testes: smoke tests e comparacao de logits/outputs;
- Config local: `legacy-code.toml` e variaveis de ambiente;
- Logs: prompts/respostas em JSONL para auditoria.

## Agente ativo

O agente customizado que o VS Code pode descobrir fica em [caicos-gpu.agent.md](caicos-gpu.agent.md).

## Regra do harness

Tudo aqui deve continuar orientado ao uso pratico do workspace, sem perder o limite real da placa: 1 GB de VRAM, driver legado e caminho prático centrado em OpenCL e modelos muito quantizados.
