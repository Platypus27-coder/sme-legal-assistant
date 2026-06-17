"""
VietPhapLy RAG — Gemma-2-9B-it generator.

Model: unsloth/gemma-2-9b-it-bnb-4bit
  - Release: June 2024 ✅ (trước 01/03/2026)
  - Params: 9B < 14B ✅
  - License: Apache 2.0 ✅
  - VRAM (4-bit): ~6GB ✅ fit Colab T4

Fallback chain:
  gemma-2-9b-it → gemma-2-2b-it → Qwen2.5-3B
"""

from __future__ import annotations

import gc
from typing import Any, Sequence

from vpl.answer.prompts import build_messages
from vpl.settings import GENERATION


class LegalGenerator:
    """Batched LLM generator với OOM auto-recovery."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls) -> "LegalGenerator":
        cfg = GENERATION

        # 1. Try Ollama local (fastest, no VRAM)
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0) as r:
                if r.status == 200:
                    print("✅ Ollama detected — using local API")
                    return cls(model="ollama", tokenizer=None)
        except Exception:
            pass

        # 2. Try Unsloth (optimized 4-bit)
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.model_name,
                max_seq_length=cfg.max_seq_length,
                dtype=None,
                load_in_4bit=cfg.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
            print(f"✅ Loaded via Unsloth: {cfg.model_name}")
            return cls(model=model, tokenizer=tokenizer)
        except Exception as exc:
            print(f"⚠ Unsloth failed ({exc}), trying fallbacks...")

        # 3. Standard transformers fallback
        import platform
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = cfg.model_name
        # Map unsloth 4-bit names to standard HF names
        model_map = {
            "unsloth/gemma-2-9b-it-bnb-4bit": "google/gemma-2-9b-it",
            "unsloth/gemma-2-2b-it-bnb-4bit": "google/gemma-2-2b-it",
        }
        model_name = model_map.get(model_name, model_name)

        # Try fallback models on Apple Silicon
        if platform.system() == "Darwin" and "9b" in model_name.lower():
            model_name = "google/gemma-2-2b-it"

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else (
            torch.float16 if device == "mps" else torch.float32
        )
        print(f"Loading {model_name} on {device} ({dtype})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device in ("mps", "cpu"):
            model = model.to(device)
        print(f"✅ Loaded via transformers: {model_name}")
        return cls(model=model, tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cpu"

    def generate(
        self,
        questions: Sequence[str],
        contexts: Sequence[Sequence[Any]],
    ) -> list[str]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must be the same length")

        # Ollama path
        if self.model == "ollama":
            return self._generate_ollama(questions, contexts)

        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(q, ctx, GENERATION.max_context_chars),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q, ctx in zip(questions, contexts)
        ]
        answers: list[str] = []
        bs = GENERATION.batch_size
        for i in range(0, len(prompts), bs):
            answers.extend(self._batch_with_backoff(prompts[i : i + bs]))
        return answers

    def generate_raw(self, messages: list[dict], max_new_tokens: int = 150) -> list[str]:
        """Single-message generation (dùng cho HyDE)."""
        if self.model == "ollama":
            return self._ollama_single(messages, max_new_tokens)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._batch_with_backoff([prompt], override_max_tokens=max_new_tokens)

    def _batch_with_backoff(
        self, prompts: Sequence[str], override_max_tokens: int | None = None
    ) -> list[str]:
        try:
            return self._do_generate(prompts, override_max_tokens)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if len(prompts) <= 1 or not any(t in msg for t in ("out of memory", "cuda", "cublas")):
                raise
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(prompts) // 2
            return (
                self._batch_with_backoff(prompts[:mid], override_max_tokens)
                + self._batch_with_backoff(prompts[mid:], override_max_tokens)
            )

    def _do_generate(
        self, prompts: Sequence[str], override_max_tokens: int | None = None
    ) -> list[str]:
        import torch

        cfg = GENERATION
        max_new = override_max_tokens or cfg.max_new_tokens
        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_length - max_new,
        ).to(self._device())
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "repetition_penalty": cfg.repetition_penalty,
            "use_cache": True,
        }
        if cfg.temperature <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)

        width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        generated = outputs[:, width:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        del inputs, outputs, generated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [a.strip() for a in decoded]

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _ollama_single(self, messages: list[dict], max_new_tokens: int) -> list[str]:
        import json
        import time
        import urllib.request

        cfg = GENERATION
        payload = {
            "model": "gemma2:9b",
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": 0.0},
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60.0) as r:
                    return [json.loads(r.read())["message"]["content"].strip()]
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    print(f"❌ Ollama failed: {exc}")
                    return [""]
        return [""]

    def _generate_ollama(
        self,
        questions: Sequence[str],
        contexts: Sequence[Sequence[Any]],
    ) -> list[str]:
        import json
        import time
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor

        cfg = GENERATION

        def _call(args: tuple) -> str:
            q, ctx = args
            msgs = build_messages(q, ctx, cfg.max_context_chars)
            payload = {
                "model": "gemma2:9b",
                "messages": msgs,
                "stream": False,
                "options": {
                    "num_predict": cfg.max_new_tokens,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                },
            }
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=120.0) as r:
                        return json.loads(r.read())["message"]["content"].strip()
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
                    else:
                        return f"Lỗi: {exc}"
            return ""

        with ThreadPoolExecutor(max_workers=max(1, cfg.batch_size)) as pool:
            return list(pool.map(_call, zip(questions, contexts)))
