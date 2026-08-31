# RELIC Research Foundation: 2026 Theses & Applied Roadmap

> **Inferência local em hardware de consumidor, restrição extrema de VRAM, múltiplos backends heterogêneos e minimização da movimentação de dados.**

Este documento consolida o estado-da-arte da literatura de pesquisa de **2026** aplicável ao **RELIC** (parte do ecossistema [Anamnesic](https://github.com/anamnesic)), estabelecendo os fundamentos teóricos, as decisões arquiteturais e o roteiro prático de implementação do runtime.

---

## 🧭 Visão Geral e Filosofia do RELIC

Diferente de frameworks tradicionais (CUDA-only, focados em data centers e GPUs modernas com dezenas de gigabytes de VRAM), o RELIC ataca a **fronteira dos recursos restritos**:
1. **Memória limitada** (1 GB a 4 GB de VRAM dedicadas ou iGPUs com memória compartilhada).
2. **Hardware heterogêneo não-utilizado** (coexistência de CPU AVX2, Intel UHD Graphics e GPU dedicada NVIDIA/AMD).
3. **Gargalo de movimentação de dados**: na inferência autoregressiva (decode), a largura de banda de memória (*memory bandwidth*) e o overhead de transferência PCIe são ordens de grandeza mais limitantes do que o pico de FLOPS aritméticos.

---

## 📊 Matriz de Prioridade dos Trabalhos de 2026

| Prioridade | Trabalho de 2026 | Relevância para o RELIC | Componente / O que implementar no RELIC |
| :--- | :--- | :---: | :--- |
| 🥇 **1** | [**ATSInfer — Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices**](https://arxiv.org/abs/2607.10183) (Jul/2026) | ⭐⭐⭐⭐⭐ | **Tensor Planner granular (CPU/iGPU/dGPU)** baseado na métrica `benefit_per_byte`. |
| 🥈 **2** | [**Llamas on the Web — Memory-Efficient, Performance-Portable LLM Inference with WebGPU**](https://www.microsoft.com/en-us/research/publication/llamas-on-the-web-memory-efficient-performance-portable-and-multi-precision-llm-inference-with-webgpu/) (Mai/2026) | ⭐⭐⭐⭐⭐ | **DeviceProfile + KernelRegistry + Autotuner** e **Static Memory Planning** portável. |
| 🥉 **3** | [**AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell**](https://arxiv.org/abs/2607.16831) (Jul/2026) | ⭐⭐⭐⭐⭐ | **Fusão estrutural de kernels** para arquitetura recorrente / híbrida (Qwen 3.5 / DeltaNet). |
| **4** | [**Fast NF4 Dequantization Kernels for Large Language Model Inference**](https://arxiv.org/abs/2604.02556) (Abr/2026) | ⭐⭐⭐⭐ | **Hierarquia de memória local/compartilhada (`__local`)** e reuso de scales/metadados. |
| **5** | [**Cache-Resident LLM Inference in GB-Scale Last-Level Caches**](https://doi.org/10.48550/arXiv.2606.25353) (Jun/2026) | ⭐⭐⭐⭐ | **Princípio arquitetural de Residência**: Data movement como custo de primeira classe. |
| **6** | [**SAW-INT4: System-Aware 4-Bit KV-Cache Quantization**](https://arxiv.org/abs/2604.19157) (Abr/2026) | ⭐⭐⭐⭐ | **Quantização e rotação fundidas inline no kernel** de KV cache sob pressão de VRAM. |
| **7** | [**FluxBin: Flexible LUT-based Ultra-low-bit LLM Inference**](https://arxiv.deeppaper.ai/papers/2608.15602v1) (Ago/2026) | ⭐⭐⭐ | Backend experimental ultra-low-bit (Q2/Q3 com LUT) para GPUs antigas (1GB VRAM). |
| **8** | [**Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing**](https://arxiv.org/abs/2607.07953) (Jul/2026) | ⭐⭐⭐ | Formalização matemática do estado recorrente DeltaNet/Gated DeltaNet para guiar fusões. |

---

## 🔬 Análise Técnica Detalhada & Aplicação no RELIC

---

### 1. ATSInfer — Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference

* **Autores / Publicação:** arXiv:2607.10183 (Julho de 2026)
* **Problema:** Desktops e notebooks de consumidor onde o modelo não cabe integralmente na VRAM da GPU.
* **Tese Principal:** A partição tradicional por camada inteira (*layer offloading*) é subótima. Decisões de alocação devem ser tomadas no nível de **tensor individual**, priorizando tensores com maior ganho de latência por byte consumido de VRAM (`benefit_per_byte`). O sistema adapta essas decisões dinamicamente durante a execução conforme o desempenho de CPU, GPU e transferências oscila.
* **Resultados:** Até **1,94× no prefill** e **3,29× no decode** em comparação ao `llama.cpp` sob o mesmo orçamento estrito de VRAM.

#### Aplicação Direta no RELIC:

```text
RELIC Tensor Planner
                      ┌──────────────────────┐
                      │    Tensor Spec       │
                      │ ──────────────────── │
                      │ • Tamanho (bytes)    │
                      │ • Custo de execução  │
                      │   (CPU, iGPU, dGPU)  │
                      │ • Custo transferência│
                      │ • Frequência de uso  │
                      └──────────┬───────────┘
                                 │
                                 ▼
                 benefit_per_byte = ΔLatency / VRAM_Bytes
                                 │
                                 ▼
                     Dynamic Placement Sorter
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     Tier 0 (dGPU)        Tier 1 (iGPU)        Tier 2 (CPU)
   Hot Attention/MLP    Overflow Sub-layers    Embeddings / Cold
```

* **Implementação:** Integrar o cálculo de `benefit_per_byte` no `AdaptivePlanner` (`src/planner/adaptive_planner.cpp`), decidindo individualmente a residência de pesos como `wq`, `wk`, `wv`, `w_gate`, `w_up`, `w_down` em vez de migrar blocos inteiros de camadas.

---

### 2. Llamas on the Web — Performance-Portable LLM Inference with WebGPU

* **Autores / Publicação:** Microsoft Research (Maio de 2026)
* **Tese Principal:** *Performance Portability* em hardware heterogêneo (testado em 16 dispositivos de 8 fabricantes distintos) é alcançada com três pilares:
  1. **Static Memory Planning** (reaproveitamento determinístico de buffers de ativação sem alocações dinâmicas em tempo de execução).
  2. **Biblioteca de Kernels Especializados** (templates otimizados para vetores, subgrupos, memória local e multi-row).
  3. **Seleção guiada por Device Profile** (o runtime afere as características do hardware e autotuna a escolha do melhor kernel).
* **Resultados:** 29–33% menor pegada de memória e 45–69% maior taxa de decode frente a outros frameworks WebGPU, competindo diretamente com backends nativos específicos de fabricante.

#### Aplicação Direta no RELIC:

Substituir condicionais estáticas de plataforma (`if (nvidia) ... else ...`) por um **Pipeline de Perfil & Registro de Kernels**:

```text
       DeviceProfile
       ├── vendor & device_id
       ├── local_mem_size
       ├── max_workgroup_size
       ├── vector_width (SIMD / Warp)
       ├── subgroup_support / warp_size
       ├── fp16 / int8 capability
       └── bandwidth_estimate (GB/s)
                   │
                   ▼
             KernelRegistry
       ┌───────────────────────────────┐
       │ GEMV Q4 Variants:             │
       │  • generic_scalar             │
       │  • local_mem_tiled            │
       │  • multi_row_8x_coalesced     │
       │  • subgroup_shuffle_warp32    │
       │  • vectorized_opencl12        │
       └───────────────┬───────────────┘
                       │
                       ▼
            Micro-Benchmark Autotuner
       (Escolhe a variante ótima no probe)
```

---

### 3. AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell

* **Autores / Publicação:** MLSys 2026 FlashInfer Contest / arXiv:2607.16831 (Julho de 2026)
* **Tese Principal:** Aceleração de operadores lineares recorrentes (Gated DeltaNet) atinge um platô quando se otimiza apenas o corpo interno do kernel. O salto de desempenho (**1,58× de speedup**) vem da **reestruturação do workload** e **mega-fusão de operadores adjacentes**.
* **Pipeline de Fusão de Operações:**
  $$\text{Update Recurrent State} \longrightarrow \text{RMSNorm} \longrightarrow \text{Gating (Swish/Sigmoid)} \longrightarrow \text{Output Projection}$$

#### Aplicação Direta no RELIC:

O RELIC já validou esse conceito na transição do FFN não-fundido para o **Fused FFN (Gate + Up + SwiGLU)** com salto de performance. Para modelos recorrentes e híbridos (como Qwen 3.5 com DeltaNet / Gated DeltaNet):
* Fundir a atualização do estado temporal recorrente com a normalização e o gating em um único despacho OpenCL, eliminando escritas e leituras intermediárias na memória global.

---

### 4. Fast NF4 Dequantization Kernels for Large Language Model Inference

* **Autores / Publicação:** arXiv:2604.02556 (Abril de 2026)
* **Tese Principal:** A desquantização em tempo de execução pode anular a economia de memória se mal implementada. A solução ótima é a exploração agressiva de memória compartilhada/local (`__local`) para reter metadados pequenos (tabelas de quantização, scales, zero-points) e reutilizá-los entre threads do mesmo workgroup.
* **Resultados:** 2,0–2,2× de aceleração direta no kernel e até **1,54× end-to-end**, utilizando apenas 64 bytes de shared memory por bloco.

#### Aplicação Direta no RELIC:

```c
// Metodologia de Hierarquia de Memória em OpenCL:
__global  // Carregar blocos compactados (pesos quantizados)
   ↓
__local   // Reter scales, bias e LUTs no cache local do Workgroup
   ↓
__private // Acumulação nos registradores da thread (SIMD/Warp)
```

* No OpenCL do RELIC, explorar sistematicamente os modificadores de memória `__global`, `__local` e `__private`, medindo onde scales, zero-points e LUTs de quantização devem residir.

---

### 5. Cache-Resident LLM Inference in GB-Scale Last-Level Caches

* **Autores / Publicação:** DOI: 10.48550/arXiv.2606.25353 (Junho de 2026)
* **Tese Principal:** **A inferência moderna é estritamente limitada pelo custo de movimentação de dados (*data movement bound*), não por FLOPs.** Organizar a execução em torno do conceito de **Residência** (working set residente, redução de sincronizações de pipeline e minimização de transferências de barramento) é a prioridade #1 de projeto.

#### Aplicação Direta no RELIC:

Transformar esse princípio em **regra arquitetural mandatória**:

> *"Data movement is a first-class cost in RELIC's execution planner."*

* A arquitetura de `GpuTensorStore` e buffers de residência de tensores (`VramPoolAllocator`) devem garantir que tensores frequentes jamais sofram descarte/recarregamento redundante via PCIe.

```text
ERRADO:
operator -> load -> operator -> load -> operator -> load

MELHOR (RELIC):
placement -> resident working set -> execute many times -> minimal data movement
```

---

### 6. SAW-INT4: System-Aware 4-Bit KV-Cache Quantization

* **Autores / Publicação:** arXiv:2604.19157 (Abril de 2026)
* **Tese Principal:** A quantização de KV-cache só é viável se for integrada de forma síncrona e fundida (*fused rotation + quantization*) no fluxo de execução, atingindo overhead de latência próximo de zero em relação ao INT4 padrão.

#### Aplicação Direta no RELIC:

```text
Não fazer:
quantize -> store -> load -> dequantize -> compute

Fazer (RELIC Fused Path):
load packed -> decode/dequantize inline -> compute
(executado no mesmo kernel de atenção)
```

* No gerenciador `AdaptiveKVManager` (`src/kv/kv_manager.cpp`), implementar a compressão dinâmica sob pressão de memória ($FP16 \to Q8\_0 \to Q4\_0$) com desquantização inline no kernel de atenção, eliminando passagens dedicadas de conversão.

---

### 7. FluxBin: Flexible LUT-based Ultra-low-bit LLM Inference

* **Autores / Publicação:** arXiv:2608.15602 (Agosto de 2026)
* **Tese Principal:** Inferência em representações ultra-compactas (Q2/Q3 e sub-2-bit) através de tabelas de consulta (*LUT-based*) e fusão direta com a escala do bloco, alcançando até **5,92× de speedup** sem exigir instruções especializadas de hardware.

#### Aplicação Direta no RELIC:

```text
RELIC Q2/Q3 Ultra-Low-Bit Pipeline:
packed weights -> LUT lookup -> scale fusion -> direct accumulation
```

* Criar backend experimental de ultra-baixa precisão para placas com **1 GB a 2 GB de VRAM** (ex: AMD Radeon HD 6450 / Caicos, Intel HD Graphics legadas), viabilizando modelos de 3B a 7B em memórias diminutas.

---

### 8. Linear Attention Architectures: Mechanisms, Trade-offs & Cross-Layer Routing

* **Autores / Publicação:** arXiv:2607.07953 (Julho de 2026)
* **Tese Principal:** Unificação matemática e análise de trade-offs entre arquiteturas recorrentes lineares (DeltaNet, Gated DeltaNet, Kimi Delta Attention e Gated DeltaNet-2).
* **Utilidade no RELIC:** Serve como a base algorítmica para implementar a representação de estado recorrente de modelos como Qwen 3.5 com complexidade de memória constante $O(1)$ em inferência autoregressiva.

---

## 🏗️ Síntese Arquitetural do RELIC (2026)

A integração dessas teses resulta no seguinte fluxo holístico de execução:

```text
                            RELIC CLI / API
                                   │
                                   ▼
                        Hardware Micro-Profiler
                   (VRAM, DMA Bandwidth, SIMD/Warp)
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
             CPU               Intel UHD            GTX 1650
          (AVX2/512)       (Shared Mem iGPU)       (GDDR5 dGPU)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                         Hardware Cost Database
                      (tensor / kernel / transfer)
                                   │
                                   ▼
                           Execution Planner
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        Memory Planner       Tensor Planner       Kernel Planner
       (Static Buffers)    (benefit_per_byte)    (Autotuned Registry)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                             Execution Plan
                        (Prefill vs Decode Plans)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              PREFILL PLAN                   DECODE PLAN
           (Compute-oriented)            (Bandwidth-oriented)
           Batched Gemm / Chunked        Fused GEMV / Resident State
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                          Async Heterogeneous
                                Runtime
```

---

## 🎯 Roteiro Prático de Implementação (Next Milestones)

Em vez de focar apenas em micro-otimizações pontuais de GEMV, o desenvolvimento do RELIC seguirá as diretrizes estruturais de 2026:

1. **Milestone 1 — Hardware Micro-Profiler & Cost Database:**
   - Detecção automatizada de largura de banda PCIe host-to-device e bandwidth de memória local por dispositivo (`relic --profile`).
2. **Milestone 2 — Static Memory Planner:**
   - Alocação estática determinística de buffers de ativação (`BackendBuffer`), eliminando fragmentação e chamadas de driver em tempo de execução.
3. **Milestone 3 — Tensor-Level Placement (`benefit_per_byte`):**
   - Agendamento granular de tensores por matriz individual entre VRAM dedicada, iGPU compartilhada e Host RAM.
4. **Milestone 4 — Desacoplamento de Planos de Prefill e Decode:**
   - Estratégias especializadas para a fase *compute-heavy* (prefill) e a fase *bandwidth-heavy* (decode).
5. **Milestone 5 — Kernel Registry com Autotuning:**
   - Biblioteca de variações de kernels com micro-benchmarking em tempo de inicialização.
6. **Milestone 6 — Runtime Assíncrono Heterogêneo:**
   - Agendamento concorrente e sem contenção entre CPU, Intel UHD e GPU dedicada.

---

## 📚 Referências Bibliográficas

1. [Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices (ATSInfer)](https://arxiv.org/abs/2607.10183) — arXiv:2607.10183, Jul 2026.
2. [Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU](https://www.microsoft.com/en-us/research/publication/llamas-on-the-web-memory-efficient-performance-portable-and-multi-precision-llm-inference-with-webgpu/) — Microsoft Research, Mai 2026.
3. [Technical Report: AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell](https://arxiv.org/abs/2607.16831) — arXiv:2607.16831, Jul 2026.
4. [Fast NF4 Dequantization Kernels for Large Language Model Inference](https://arxiv.org/abs/2604.02556) — arXiv:2604.02556, Abr 2026.
5. [Cache-Resident LLM Inference in GB-Scale Last-Level Caches](https://doi.org/10.48550/arXiv.2606.25353) — arXiv:2606.25353, Jun 2026.
6. [SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving](https://arxiv.org/abs/2604.19157) — arXiv:2604.19157, Abr 2026.
7. [FluxBin: Flexible LUT-based Ultra-low-bit LLM Inference by Algorithm-Kernel Synergy](https://arxiv.deeppaper.ai/papers/2608.15602v1) — arXiv:2608.15602, Ago 2026.
8. [Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing](https://arxiv.org/abs/2607.07953) — arXiv:2607.07953, Jul 2026.