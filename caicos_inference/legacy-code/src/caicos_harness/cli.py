from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .ollama import OllamaError, chat, list_models, load_model_gpu_only, require_gpu_only


DEFAULT_MODEL = os.environ.get("CAICOS_MODEL", "llama3.1:8b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caicos-harness",
        description="Local coding harness for the unsupported AMD Caicos XT target.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Send a coding prompt to Ollama")
    chat_parser.add_argument("prompt", help="Prompt to send to the local model")
    chat_parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")

    models_parser = subparsers.add_parser("models", help="List Ollama models")
    models_parser.add_argument("--show-empty", action="store_true", help="Keep empty entries")

    doctor_parser = subparsers.add_parser("doctor", help="Check the local setup")
    doctor_parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")

    path_parser = subparsers.add_parser(
        "read", help="Print a file to feed a coding prompt"
    )
    path_parser.add_argument("path", help="Path to a file inside the workspace")

    return parser


def _system_prompt() -> str:
    return (
        "You are Caicos Coding Harness. Keep answers concrete, local-first, and grounded in "
        "the AMD Caicos XT limits when GPU or inference details matter. Prefer small, testable "
        "changes and concise reasoning."
    )


def run_chat(prompt: str, model: str) -> int:
    load_model_gpu_only(model)
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": prompt},
    ]
    print(chat(model, messages))
    return 0


def run_models(show_empty: bool) -> int:
    models = list_models()
    for model in models:
        if model or show_empty:
            print(model)
    return 0


def run_doctor(model: str) -> int:
    models = list_models()
    print(f"Ollama host: {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"Default model: {model}")
    print(f"Models found: {len(models)}")
    try:
        load_model_gpu_only(model)
        print("Status: gpu-only ready")
    except OllamaError as exc:
        print(f"Status: {exc}")
        return 2
    return 0


def run_read(path: str) -> int:
    content = Path(path).read_text(encoding="utf-8")
    print(content)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "chat":
            return run_chat(args.prompt, args.model)
        if args.command == "models":
            return run_models(args.show_empty)
        if args.command == "doctor":
            return run_doctor(args.model)
        if args.command == "read":
            return run_read(args.path)
    except OllamaError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Arquivo nao encontrado: {exc.filename}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())