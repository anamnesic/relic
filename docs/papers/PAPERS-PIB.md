# Papers Acadêmicos — PIB para Anamnesic

> Compilado em: 15 Jul 2026
> Foco: Context Management, Multi-Agent Systems, MCP, AI Memory

---

## Resumo Executivo

O Anamnesic se encaixa perfeitamente em uma linha de pesquisa acadêmica ativa e crescente: **gestão de contexto para agentes de IA em tarefas de longo horizonte**. Os papers abaixo validam o problema que o Anamnesic resolve e oferecem fundamentação teórica para suas decisões arquiteturais.

**Áreas de pesquisa relevantes:**
1. Context Management para SWE Agents
2. Memory Hierarchies para LLMs
3. Model Context Protocol (MCP)
4. Multi-Agent Systems para Software Engineering
5. Codified Context para Projetos

---

## 1. Context Management para SWE Agents

### Paper: "Context as a Tool (CAT)"
- **URL:** https://aclanthology.org/2026.findings-acl.1032.pdf
- **Venue:** ACL 2026 Findings
- **Autores:** Não especificado
- **Resumo:** Propõe CAT, um paradigma que trata gestão de contexto como tool chamável pelo agente. Estrutura workspace com:
  - Task semantics (estável)
  - Long-term memory (condensada)
  - Short-term working memory (alta fidelidade)
- **Relevância para Anamnesic:**
  - Valida a abordagem de "context as a first-class citizen"
  - O Anamnesic já implementa isso via `ContextService` + `DecisionService`
  - Possível extensão: tornar context management uma tool MCP

**Dados-chave:**
- SWE-Compressor alcançou 57.6% no SWE-Bench Verified
- Supera agents ReAct e baselines de compressão estática
- Mantém reasoning estável sob orçamento de contexto limitado

### Paper: "SWE-MeM: Learning Adaptive Memory Management"
- **URL:** https://arxiv.org/html/2606.28434
- **Venue:** arXiv 2026
- **Resumo:** Framework de treino para gestão de memória proativa em agents de software engineering.
- **Relevância:**
  - Demonstra que gestão de memória proativa melhora performance
  - SWE-MeM alcançou 60.2% no SWE-Bench com modelo de 30B
  - O Anamnesic poderia implementar "memory-aware GRPO"

---

## 2. Memory Hierarchies para LLMs

### Paper: "Neural Paging: Learning Context Management Policies"
- **URL:** https://arxiv.org/pdf/2603.02228
- **Venue:** arXiv 2026
- **Autores:** Liang Ruo Chen, Qi Liu
- **Resumo:** Framework que desacopla reasoning de memory management, inspirado em sistemas operacionais.
- **Conceitos-chave:**
  - Context Paging Problem (CPP)
  - Page Controller como "neural MMU"
  - Hierarchical Neural Turing Machine (H-NTM)
- **Relevância:**
  - O Anamnesic já separa `core` (lógica) de `mcp-server` (transporte)
  - Possível extensão: implementar "demand paging" para contexto
  - Reduz complexidade de O(N²) para O(N·K²)

**Dados-chave:**
- Under bounded context window K, reduz complexidade assintótica
- Competitive-ratio degradação é graceful sob policy-dependent access

### Paper: "The Missing Memory Hierarchy: Demand Paging for LLM Context Windows"
- **URL:** https://doi.org/10.48550/arxiv.2603.09023
- **Venue:** arXiv 2026
- **Autores:** Tony Mason
- **Resumo:** Pichay — sistema de demand paging para context windows de LLMs.
- **Hierarquia proposta:**
  - L1: Generation window (context window atual)
  - L2: Working set (demand-paged, pinned)
  - L3: Session history (compressão cooperativa)
  - L4: Cross-session persistent memory
- **Relevância:**
  - O Anamnesic implementa L4 (SQLite local) mas não L2/L3
  - Possível extensão: implementar L2 (fault-driven pinning)
  - 21.8% dos tokens são "structural waste" em produção

**Dados-chave:**
- Fault rate: 0.0254% offline
- Redução de contexto: até 93% (5,038KB → 339KB)
- Em produção por 681 turns

### Paper: "Contextual Memory Virtualisation (CMV)"
- **URL:** https://arxiv.org/pdf/2602.22402
- **Venue:** arXiv 2026
- **Autores:** Cosmo Santoni
- **Resumo:** Trata contexto acumulado como estado version-controlled, inspirado em memória virtual de SO.
- **Conceitos:**
  - DAG para histórico de sessões
  - Snapshot, branch, trim primitives
  - Trimming estruturalmente lossless (20-86% redução)
- **Relevância:**
  - O Anamnesic já tem `SnapshotService` e `RollbackService`
  - Possível extensão: versionamento de contexto via DAG
  - 39% de redução em sessões com tool use misto

---

## 3. Model Context Protocol (MCP)

### Paper: "MCP: Landscape, Security Threats, and Future Research Directions"
- **URL:** https://dl.acm.org/doi/10.1145/3796519
- **Venue:** ACM TOSEM 2026
- **Resumo:** Primeira análise abrangente do ecossistema MCP.
- **Ciclo de vida do MCP server:**
  1. Creation
  2. Deployment
  3. Operation
  4. Maintenance
- **16 atividades-chave** mapeadas
- **Relevância:**
  - Valida a escolha do MCP como protocolo padrão
  - Identifica gaps de segurança que o Anamnesic pode endereçar
  - Ecossistema em crescimento: 5,000+ servers community

**Adoção por empresas:**
- Anthropic, OpenAI, Baidu, Replit, Microsoft, Cloudflare, Stripe

### Paper: "Breaking the Protocol: Security Analysis of MCP"
- **URL:** https://arxiv.org/pdf/2601.17549v1
- **Venue:** arXiv 2026
- **Resumo:** Análise de segurança do MCP v1.0.
- **3 vulnerabilidades de protocolo:**
  1. Ausência de capability attestation
  2. Bidirectional sampling sem autenticação
  3. Trust propagation implícito em multi-server
- **Proposta:** AttestMCP (extensão backward-compatible)
- **Relevância:**
  - O Anamnesic pode implementar capability attestation
  - O `SecurityAnalyzer` já detecta algumas dessas issues
  - Possível contribuição: implementar AttestMCP

**Dados-chave:**
- MCP amplifica attack success rate em 23-41%
- AttestMCP reduz ASR de 52.8% para 12.4%
- Overhead: 8.3ms por mensagem

### Paper: "Enhancing MCP with Context-Aware Server Collaboration"
- **URL:** https://arxiv.org/pdf/2601.11595v2
- **Venue:** arXiv 2026
- **Resumo:** CA-MCP — extensão com Shared Context Store (SCS).
- **Problema:** MCP tradicional é stateless, sem contexto global
- **Solução:** SCS acessível por todos os servers
- **Relevância:**
  - O Anamnesic implementa isso via `PersistentEventBus`
  - Possível extensão: tornar SCS explícito como tool
  - Reduz chamadas LLM em complex tasks

**Dados-chave:**
- Redução de chamadas LLM
- Melhoria em TravelPlanner e REALM-Bench
- Coordenação autônoma entre servers

---

## 4. Multi-Agent Systems para Software Engineering

### Paper: "LLM-Based Multi-Agent Systems for SE: Literature Review"
- **URL:** https://dl.acm.org/doi/10.1145/3712003
- **Venue:** ACM TOSEM 2026
- **Resumo:** Revisão sistemática de 41 papers sobre LMA systems em SE.
- **Fases do SDLC cobertas:**
  - Requirements Engineering
  - Code Generation
  - QA
  - Maintenance
  - End-to-end Development
- **Research Agenda:**
  - Phase 1: Enhancing Individual Agent Capabilities
  - Phase 2: Optimizing Agent Synergy
- **Relevância:**
  - O Anamnesic implementa 11 agent roles (PM, Architect, Backend, etc.)
  - Possível extensão: auto-scaling de agentes
  - Gap: Privacy e confidentiality em LMA systems

**Papers-chave citados:**
- AutoGen, ChatDev, MetaGPT
- Software Engineering 2.0

### Paper: "COMPASS: Context-Organized Multi-Agent Planning"
- **URL:** https://aclanthology.org/2026.acl-long.152
- **Venue:** ACL 2026 Long Papers
- **Autores:** Guangya Wan et al. (Google Cloud AI + UVA)
- **Resumo:** Framework hierárquico com 3 componentes:
  1. Main Agent (reasoning + tool use)
  2. Meta-Thinker (monitoramento + intervenções estratégicas)
  3. Context Manager (maintém briefs concisos)
- **Relevância:**
  - O Anamnesic já separa Concerns (core, mcp, cli)
  - Possível extensão: adicionar Meta-Thinker layer
  - Context Manager = `ContextService` + `ExportService`

**Dados-chave:**
- +20% de accuracy vs single/multi-agent baselines
- GAIA: 88.3% Pass@3
- BrowseComp-ZH: 65.9% Pass@3

---

## 5. Codified Context para Projetos

### Paper: "Codified Context: Infrastructure for AI Agents"
- **URL:** https://arxiv.org/pdf/2602.20478
- **Venue:** arXiv 2026
- **Autores:** Aristidis Vasilopoulos
- **Resumo:** Infraestrutura de 3 tiers para contexto codificado:
  1. Hot-memory constitution (convenções)
  2. Domain-expert agents (19 agentes especializados)
  3. Cold-memory knowledge base (34 documentos)
- **Relevância:**
  - O Anamnesic implementa isso via `.cursorrules`, `CLAUDE.md`
  - Possível extensão: tiered context loading
  - 108,000 linhas de C# construídas com essa abordagem

**Dados-chave:**
- 283 sessões de desenvolvimento
- 2,801 prompts humanos
- 1,197 invocações de agentes
- 16,522 turns de agentes

### Paper: "Spec Kit Agents: Context-Grounded Agentic Workflows"
- **URL:** https://arxiv.org/pdf/2604.05278
- **Venue:** arXiv 2026
- **Resumo:** Pipeline multi-agente com context-grounding hooks.
- **Fases:** Specify → Plan → Tasks → Implement
- **Relevância:**
  - O Anamnesic já tem pipeline de 11 fases
  - Possível extensão: adicionar validation hooks
  - 58.2% Pass@1 no SWE-bench Lite

---

## 6. Gaps de Pesquisa que o Anamnesic Pode Endereçar

### Gap 1: Cross-Session Context Persistence
- **Problema:** Maioria dos sistemas perde contexto entre sessões
- **Solução Anamnesic:** SQLite local + export para múltiplos formatos
- **Oportunidade:** Publicar paper sobre "Persistent Context for AI Coding Agents"

### Gap 2: Multi-Format Context Export
- **Problema:** Cada ferramenta IA tem seu formato
- **Solução Anamnesic:** Export unificado para 6 formatos
- **Oportunidade:** Publicar paper sobre "Universal Context Interchange Protocol"

### Gap 3: Context Versioning
- **Problema:** Não há versionamento de decisões arquiteturais
- **Solução Anamnesic:** `DecisionService` + `AuditEvent`
- **Oportunidade:** Publicar paper sobre "Version-Controlled Context for Software Projects"

### Gap 4: Context Security
- **Problema:** MCP tem vulnerabilidades de protocolo
- **Solução Anamnesic:** `SecurityAnalyzer` + guardrails
- **Oportunidade:** Publicar paper sobre "Secure Context Propagation in Multi-Agent Systems"

---

## 7. Sugestões de Publicações (PIB)

### Opção 1: Paper de Segurança
- **Título:** "Secure Context Synchronization for AI Coding Agents: A Protocol-Level Analysis"
- **Baseado em:** Breaking the Protocol (MCP security)
- **Contribuição:** Análise de segurança do Anamnesic + proposta de AttestMCP
- **Venue:** ACM CCS ou NDSS

### Opção 2: Paper de Arquitetura
- **Título:** "Universal Context Interchange: A Multi-Format Export Protocol for AI Tools"
- **Baseado em:** CAT, Cognitive Scaffold
- **Contribuição:** Protocolo unificado para export de contexto
- **Venue:** ACL ou EMNLP

### Opção 3: Paper de Sistemas
- **Título:** "Demand Paging for AI Context Windows: A Memory Hierarchy Approach"
- **Baseado em:** Neural Paging, Pichay
- **Contribuição:** Implementação de memory hierarchy no Anamnesic
- **Venue:** OSDI ou SOSP

### Opção 4: Paper de Engenharia
- **Título:** "Building a 108K-Line System with Codified Context: A Case Study"
- **Baseado em:** Codified Context
- **Contribuição:** Case study do Anamnesic com métricas reais
- **Venue:** ICSE ou FSE

---

## 8. Referências Bibliográficas

### Context Management
1. CAT (ACL 2026) - https://aclanthology.org/2026.findings-acl.1032.pdf
2. Neural Paging (arXiv 2026) - https://arxiv.org/pdf/2603.02228
3. Pichay (arXiv 2026) - https://doi.org/10.48550/arxiv.2603.09023
4. CMV (arXiv 2026) - https://arxiv.org/pdf/2602.22402
5. SWE-MeM (arXiv 2026) - https://arxiv.org/html/2606.28434
6. Cognitive Scaffold (ACL 2026) - https://aclanthology.org/2026.acl-long.1170

### MCP
7. MCP Landscape (ACM TOSEM 2026) - https://dl.acm.org/doi/10.1145/3796519
8. Breaking the Protocol (arXiv 2026) - https://arxiv.org/pdf/2601.17549v1
9. Enterprise-Grade Security (IEEE 2026) - https://doi.org/10.1109/icaic67076.2026.11395723
10. CA-MCP (arXiv 2026) - https://arxiv.org/pdf/2601.11595v2

### Multi-Agent Systems
11. LMA Systems for SE (ACM TOSEM 2026) - https://dl.acm.org/doi/10.1145/3712003
12. COMPASS (ACL 2026) - https://aclanthology.org/2026.acl-long.152

### Codified Context
13. Codified Context (arXiv 2026) - https://arxiv.org/pdf/2602.20478
14. Spec Kit Agents (arXiv 2026) - https://arxiv.org/pdf/2604.05278

### Git Context Controller
15. GCC (arXiv 2025) - https://arxiv.org/pdf/2508.00031

---

## 9. Próximos Passos Acadêmicos

### Curto Prazo (1-3 meses)
- [ ] Revisar CAT e implementar "context as a tool" no Anamnesic
- [ ] Estudar Pichay e implementar L2 (fault-driven pinning)
- [ ] Analisar AttestMCP e propor implementação

### Médio Prazo (3-6 meses)
- [ ] Escrever draft do paper de segurança (Opção 1)
- [ ] Implementar versionamento de contexto via DAG (CMV)
- [ ] Criar benchmark para context export

### Longo Prazo (6-12 meses)
- [ ] Publicar paper em conferência top-tier
- [ ] Contribuir para spec do MCP v2.0
- [ ] Criar dataset de context synchronization

---

> "O Anamnesic não é apenas um produto — é uma contribuição para a pesquisa em gestão de contexto para IA. Os papers acima validam o problema e oferecem fundamentação para suas decisões arquiteturais."
