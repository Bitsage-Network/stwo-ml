#!/usr/bin/env python3
"""Chat engine for Obelysk live demo.

Handles the interactive chat loop, calling llama.cpp server,
writing conversation.json, and launching background proving.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

PORT = 8192
BASE = f"http://localhost:{PORT}"

# Background proving state
_prove_threads: list[threading.Thread] = []
_prove_results: list[dict] = []
_prove_lock = threading.Lock()


def chat(user_input: str, messages: list) -> tuple[str, int]:
    """Send a message and get the response."""
    messages.append({"role": "user", "content": user_input})
    payload = json.dumps({
        "model": "qwen2-0.5b",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
    }).encode()

    t0 = time.time()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    gen_time = int((time.time() - t0) * 1000)
    reply = data["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply, gen_time


def tokenize(text: str) -> list[int]:
    """Tokenize text via llama.cpp server."""
    payload = json.dumps({"content": text}).encode()
    req = urllib.request.Request(
        f"{BASE}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return data.get("tokens", [0])
    except Exception:
        return [0]


def start_background_prove(turn_index: int, model_dir: str, log_dir: str):
    """Launch background proving for a turn (async).

    This is a placeholder for the async architecture. In production,
    each turn would be proved independently in a background thread
    immediately after capture, so by the time the user finishes
    chatting, most proofs are already done.
    """
    # For now we just record that proving should happen.
    # The actual proving happens in the audit step after capture.
    with _prove_lock:
        _prove_results.append({
            "turn": turn_index,
            "status": "queued",
            "start_time": time.time(),
        })


def main():
    if len(sys.argv) < 2:
        print("Usage: chat_engine.py <output_conversation.json>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]
    messages = []
    turns = []
    turn_index = 0

    GREEN = "\033[0;32m"
    CYAN = "\033[0;36m"
    WHITE = "\033[1;37m"
    DIM = "\033[0;90m"
    YELLOW = "\033[1;33m"
    RESET = "\033[0m"

    print(f"{WHITE}Chat with Qwen2-0.5B. Type {GREEN}prove{WHITE} when done.{RESET}")
    print(f"{DIM}─────────────────────────────────────────────────{RESET}")
    print()

    while True:
        try:
            user_input = input(f"{GREEN}You: {RESET}")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("prove", "quit", "done", ""):
            if user_input.strip().lower() in ("prove", "quit", "done"):
                break
            continue

        try:
            reply, gen_time = chat(user_input, messages)
        except Exception as e:
            print(f"{CYAN}Qwen: {RESET}[error: {e}]")
            print()
            continue

        print(f"{CYAN}Qwen: {RESET}{reply}")
        print()

        # Tokenize
        tokens = tokenize(user_input)
        resp_tokens = tokenize(reply)
        last_token = tokens[-1] if tokens else 0

        turns.append({
            "turn_index": turn_index,
            "content": user_input,
            "full_context_tokens": tokens,
            "last_token_id": last_token,
            "response": {
                "content": reply,
                "tokens": resp_tokens,
                "generation_time_ms": gen_time,
            },
        })
        turn_index += 1

    # Write conversation.json
    conv = {
        "conversation_id": f"demo-{int(time.time())}",
        "topic": "live demo",
        "turns": turns,
    }

    with open(output_path, "w") as f:
        json.dump(conv, f, indent=2)

    print()
    print(f"{DIM}Saved {len(turns)} turns to {output_path}{RESET}")


if __name__ == "__main__":
    main()
