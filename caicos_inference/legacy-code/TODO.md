# TODO - legacy-code base

## Runtime base (llm-on-legacy-gpus)
- [ ] Add config file (legacy-code.toml) with runtime path, model gguf, platform/device, max seq len
- [ ] Add CLI command `runtime` (or `infer`) to call runtime binary with prompt, tokens, temperature, top-k
- [ ] Add command `runtime-build` to build CMake in runtime/llm-on-legacy-gpus
- [ ] Add `runtime-doctor` to check OpenCL device list and runtime binary
- [ ] Add helper to sync submodule to a pinned commit (documented workflow)

## Ollama harness improvements
- [ ] Add streaming option to `chat` (optional `--stream`)
- [ ] Add session log (save prompt/response JSONL)
- [ ] Add `read` support for directories/globs and line ranges
- [ ] Add `models --details` to show context length and quant

## Scripts and automation
- [ ] Add PowerShell scripts to build the runtime and run a smoke prompt
- [ ] Add a script to export env vars (CAICOS_MODEL, CAICOS_NUM_GPU, CAICOS_MAIN_GPU, CAICOS_KEEP_ALIVE)

## Tests
- [ ] Add unit tests for `ollama.py` error handling with mocked HTTP
- [ ] Add CLI smoke test for `doctor` and `chat` (skipped if Ollama is not running)
- [ ] Add runtime smoke test (small gguf) and compare output to a known token

## Docs
- [ ] Document environment variables and defaults in README
- [ ] Add quickstart to clone with submodule and run a runtime sample
- [ ] Add a supported GPU profile doc referencing Caicos limits

## Production readiness
- [ ] Add config file (legacy-code.toml) with schema validation and defaults
- [ ] Add structured logging (JSONL) with request IDs and error codes
- [ ] Add retries with backoff for Ollama API failures
- [ ] Add configurable timeouts per command
- [ ] Add `--json` output mode for CLI commands
- [ ] Add unit tests + smoke tests in CI
- [ ] Add lint/format (ruff or black) and pre-commit
- [ ] Add release flow (tags + changelog + version bump)
- [ ] Add secrets redaction in logs
- [ ] Add performance metrics (latency, tokens/s)

## 2026/27 external requirements (alignment)
- [ ] EU AI Act alignment: transparency to users when interacting with AI, and labeling of AI-generated content where applicable (rules go live Aug 2026)
- [ ] EU AI Act high-risk readiness (if used in high-risk domains): risk management, data quality, logging/traceability, technical docs, human oversight, robustness/cybersecurity
- [ ] GPAI transparency/copyright obligations (if applicable) and documentation of model provenance
- [ ] NIST AI RMF mapping: document risks by Govern/Map/Measure/Manage and keep a risk register
- [ ] NIST GenAI profile (AI-600-1): add evals for harmful output, misuse, and reliability
- [ ] OWASP LLM Top 10 2025 mitigations: prompt injection defenses, sensitive data controls, supply-chain integrity, model/data poisoning checks
- [ ] OWASP LLM Top 10: output validation/sanitization, agency limits/sandbox, system prompt leakage protections
- [ ] OWASP LLM Top 10: RAG/vector store hardening, misinformation checks, and unbounded consumption limits
