"""A local LLM through Ollama (https://ollama.com), for the opt-in abstractive summaries.

Ollama runs the model on this machine (on the GPU when there is room) and serves an HTTP API that we
call with `requests`, so no extra Python package is needed. Decoding is deterministic (temperature 0,
fixed seed) and every reply is cached on disk (cache/llm/) under a hash of the model's digest, the
messages and the options: evaluation reruns cost nothing and give the same text.

  ollama pull qwen3:4b-instruct        # once (2.5 GB); then the Ollama app must be running
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .paths import cache_dir
from .utils.cache import load_json, save_json, text_hash

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    """The LLM could not answer (server not running, model not pulled, request failed)."""


@dataclass
class Stats:
    calls: int = 0
    cached: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0

    def to_json(self) -> dict:
        return {"calls": self.calls, "cached": self.cached, "seconds": round(self.seconds, 2),
                "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens}


@dataclass
class Ollama:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b-instruct"
    temperature: float = 0.0
    seed: int = 0
    num_ctx: int = 8192
    timeout: float = 600.0
    keep_alive: str = "5m"
    use_cache: bool = True
    stats: Stats = field(default_factory=Stats)
    _digest: str | None = None

    @classmethod
    def from_config(cls, c) -> "Ollama":
        return cls(c.host, c.model, c.temperature, c.seed, c.num_ctx, c.timeout, c.keep_alive)

    def _url(self, path: str) -> str:
        return self.host.rstrip("/") + path

    def status(self, timeout: float = 3.0) -> dict:
        """Is the server up and the model pulled? Never raises."""
        import requests
        out = {"provider": "ollama", "host": self.host, "model": self.model, "available": False,
               "installed": False, "models": [], "error": None}
        try:
            r = requests.get(self._url("/api/tags"), timeout=timeout)
            r.raise_for_status()
            models = r.json().get("models", [])
        except Exception as exc:
            out["error"] = f"Ollama is not running at {self.host} ({type(exc).__name__}). Start the Ollama app."
            return out
        out["available"] = True
        out["models"] = sorted(m.get("name", "") for m in models)
        for m in models:
            if m.get("name") in (self.model, f"{self.model}:latest"):
                out["installed"] = True
                self._digest = m.get("digest") or m.get("name")
        if not out["installed"]:
            out["error"] = f"The model {self.model} is not pulled: run `ollama pull {self.model}`."
        return out

    def check(self) -> None:
        """Raise LLMError unless the model can answer."""
        st = self.status()
        if not st["installed"]:
            raise LLMError(st["error"] or "the LLM is unavailable")

    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        """The assistant reply to `messages` ([{role, content}]), from the cache when possible."""
        import requests
        if self._digest is None:
            self.check()
        options = {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                   "num_predict": max_tokens}
        path = cache_dir("llm") / f"{text_hash(self._digest, self.model, messages, options)}.json"
        if self.use_cache and path.exists():
            self.stats.cached += 1
            return load_json(path)["reply"]
        t0 = time.perf_counter()
        try:
            r = requests.post(self._url("/api/chat"), timeout=self.timeout,
                              json={"model": self.model, "messages": messages, "stream": False,
                                    "options": options, "keep_alive": self.keep_alive})
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed ({type(exc).__name__}: {exc})") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise LLMError(f"Ollama answered {r.status_code}: {str(detail)[:300]}")
        data = r.json()
        reply = _THINK.sub("", data.get("message", {}).get("content", "")).strip()
        dt = time.perf_counter() - t0
        self.stats.calls += 1
        self.stats.seconds += dt
        self.stats.prompt_tokens += int(data.get("prompt_eval_count") or 0)
        self.stats.output_tokens += int(data.get("eval_count") or 0)
        log.info("llm %s: %d+%d tokens in %.1f s", self.model, data.get("prompt_eval_count") or 0,
                 data.get("eval_count") or 0, dt)
        if self.use_cache:
            save_json(path, {"model": self.model, "digest": self._digest, "messages": messages, "options": options,
                             "reply": reply, "seconds": round(dt, 2)})
        return reply

    def unload(self) -> None:
        """Ask Ollama to free the model's (GPU) memory now instead of after keep_alive."""
        import requests
        try:
            requests.post(self._url("/api/generate"), json={"model": self.model, "keep_alive": 0}, timeout=10)
        except Exception as exc:                                   # nothing to free if the server is gone
            log.debug("ollama unload: %s", exc)
