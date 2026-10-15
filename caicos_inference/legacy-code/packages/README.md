# packages

Catálogo dos repositórios que valem acompanhar neste projeto. A ideia aqui é manter uma lista curta e prática, priorizando tentativas reais de inferência local em hardware limitado, forks de `llama.cpp` e variantes com OpenCL ou caminhos parecidos.

## Foco de hardware

- GPU alvo: AMD Caicos XT (`PCI 1002:6778`)
- Classe: AMD legada (Northern Islands)
- Limite principal: 1 GB de VRAM
- Direcao: OpenCL e modelos quantizados pequenos

## Ficha tecnica

- [gpu_specs.md](gpu_specs.md): dados consolidados da Caicos XT e fontes publicas usadas na pesquisa.
- [model_candidates.md](model_candidates.md): shortlist de modelos pequenos que cabem com mais folga na Caicos XT.

## Repositórios

- [a8nova/adreno-llms](https://github.com/a8nova/adreno-llms) — LMs pequenas ajustadas para GPUs Adreno 6xx em Android, com implementação em C++/OpenCL. É um bom ponto de referência para ver como manter tudo leve em hardware fraco.
- [EffortlessMetrics/BitNet-rs](https://github.com/EffortlessMetrics/BitNet-rs) — runtime Rust para modelos BitNet 1-bit, compatível com GGUF e `llama.cpp`. Serve como trilha de inferência ultraleve.
- [pt13762104/llama.cpp](https://github.com/pt13762104/llama.cpp) — fork de `llama.cpp` com patches para dispositivos TU11x. Não é o alvo exato do Caicos, mas entra como exemplo de fork focado em GPU legado.
- [362132718/llamacpp-gfx906-furnace](https://github.com/362132718/llamacpp-gfx906-furnace) — fork de `llama.cpp` para AMD gfx906 com kernels customizados, TurboQuant e ajustes de HIP. Útil para comparar o tipo de adaptação que forks AMD costumam exigir.
- [eliranwong/AMD_iGPU_AI_Setup](https://github.com/eliranwong/AMD_iGPU_AI_Setup) — ambiente e testes para AMD iGPU AI com `llama.cpp`, ROCm e Vulkan. É mais moderno que a Caicos, mas ajuda a entender o fluxo de setup e benchmark.
- [stoflom/build-llama-cpp](https://github.com/stoflom/build-llama-cpp) — anotações práticas para compilar e rodar `llama.cpp` em hardware AMD no Fedora. Bom para referências de build e execução.

## Critério usado

Mantive nesta primeira rodada apenas repositórios que apareceram na busca e que têm ligação direta com inferência local, `llama.cpp`, OpenCL ou adaptação para AMD. Se quiser, depois posso separar em três grupos: "OpenCL puro", "forks de llama.cpp" e "setup/benchmarks AMD".

## Rota fora do Ollama

A tentativa mais natural para a Caicos XT seria `llama.cpp` com aceleracao local. O host compila `ggml-opencl`, mas o device Caicos reporta OpenCL C 1.2, nao possui `cl_khr_fp16` e limita alocacoes individuais a 512 MB. Como o backend OpenCL atual depende de FP16/subgroups e e orientado a Adreno/Intel, a opcao `llama.cpp` GPU-only foi verificada e bloqueada para este hardware sem um novo caminho de kernels OpenCL 1.2.

## Proximo foco

Se a meta for realmente GPU local nesta placa, o caminho restante precisa sair de `llama.cpp` e ir para outro runtime OpenCL legado ou para uma trilha CPU-first com modelos muito pequenos.
