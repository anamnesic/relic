from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request


DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def _base_url() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL).rstrip("/")


def _gpu_options() -> dict[str, int | float]:
    num_gpu = int(os.environ.get("CAICOS_NUM_GPU", "999"))
    main_gpu = int(os.environ.get("CAICOS_MAIN_GPU", "0"))
    return {
        "temperature": 0.2,
        "num_gpu": num_gpu,
        "main_gpu": main_gpu,
    }


def _parse_ps_output(output: str, model: str) -> str | None:
    for line in output.splitlines():
        if model not in line:
            continue
        return line
    return None


def gpu_status(model: str) -> str | None:
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OllamaError("Nao foi possivel consultar o estado do Ollama com 'ollama ps'.") from exc

    return _parse_ps_output(completed.stdout, model)


def require_gpu_only(model: str) -> None:
    line = gpu_status(model)
    if line is None:
        raise OllamaError(
            "O modelo nao apareceu em 'ollama ps'. Carregue-o primeiro e tente de novo."
        )
    if "CPU" in line.upper():
        raise OllamaError(
            f"O modelo {model} esta rodando sem GPU-only: {line}. O harness exige GPU-only."
        )


def chat(model: str, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                **_gpu_options(),
                "temperature": temperature,
            },
            "keep_alive": os.environ.get("CAICOS_KEEP_ALIVE", "5m"),
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{_base_url()}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            "Nao foi possivel conectar ao Ollama. Verifique se o servidor local esta rodando."
        ) from exc

    message = data.get("message", {})
    content = message.get("content")
    if not content:
        raise OllamaError("Ollama respondeu sem conteudo util.")
    return content


def list_models() -> list[str]:
    request = urllib.request.Request(f"{_base_url()}/api/tags", method="GET")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            "Nao foi possivel consultar os modelos do Ollama."
        ) from exc

    return [item.get("name", "") for item in data.get("models", []) if item.get("name")]


def load_model_gpu_only(model: str) -> None:
    payload = json.dumps(
        {
            "model": model,
            "messages": [],
            "stream": False,
            "options": _gpu_options(),
            "keep_alive": os.environ.get("CAICOS_KEEP_ALIVE", "5m"),
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{_base_url()}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError("Nao foi possivel carregar o modelo no Ollama.") from exc

    require_gpu_only(model)