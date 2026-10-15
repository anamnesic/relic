# caicos_inference

Projeto experimental para juntar referências de inferência local em GPUs AMD antigas, com foco em OpenCL, forks de `llama.cpp` e alternativas leves para modelos quantizados.

## Estrutura

- `packages/README.md`: índice dos repositórios encontrados e uma nota rápida sobre o papel de cada um.
- `packages/repositories.json`: inventário estruturado para evoluir depois para scripts, filtros ou automação.
- `legacy-code/README.md`: visão geral do harness local de coding com Ollama.
- `legacy-code/caicos-gpu.agent.md`: agente customizado que guia o fluxo local de coding.

## Objetivo

Mapear projetos que já tentaram contornar a falta de suporte oficial em hardware AMD legado, para servir como base de teste e comparação no seu ambiente.

## Specs coletadas

- [packages/gpu_specs.md](packages/gpu_specs.md): ficha tecnica consolidada da GPU alvo, com nomes comerciais, PCI ID, arquitetura, memoria, clocks, driver e implicacoes praticas.

## GPU alvo desta maquina

- Nome no Windows: AMD Radeon HD 7000 series
- PCI ID: `VEN_1002&DEV_6778`
- Chip exato: Caicos XT
- Nomes comerciais comuns: Radeon HD 7470/8470, R5 235/310 (OEM)
- Driver atual: 15.201.1151.1008
- VRAM reportada: 1 GB

## Direcao tecnica

Como o chip e legado e nao tem caminho moderno de ROCm, o projeto prioriza:

- tentativas com OpenCL;
- forks e patches de `llama.cpp` com foco em hardware limitado;
- modelos muito quantizados e pipelines de baixo consumo de VRAM.

## Status da opcao 1

A tentativa de subir `llama.cpp` fora do Ollama foi checada. O host consegue compilar o backend OpenCL e o `llama-cli`, mas a execucao GPU-only na Caicos XT fica bloqueada por recursos ausentes no device: OpenCL C 1.2, sem `cl_khr_fp16`, sem SVM e com alocacao maxima de 512 MB.

O caminho reproduzivel esta em [llama.cpp/docs/backend/CAICOS.md](llama.cpp/docs/backend/CAICOS.md). O teste direto com `llama-cli --list-devices` rejeita a GPU com `OpenCL C 2.0 or above is required; device reports 1.2`. A conclusao pratica e que nao basta CMake: seria necessario portar/criar um backend OpenCL 1.2 sem FP16 e sem subgroups para essa placa.
