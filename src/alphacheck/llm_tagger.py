"""A4 AI Lab: Optional Local LLM Event Tagger (Ollama).

Tags headlines into discrete financial event categories:
['earnings', 'regulatory', 'macro', 'contract', 'other']
at temperature 0.0 with deterministic local caching.
Gracefully falls back to heuristic tagging if Ollama is not installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import requests


EVENT_TAGS = ["earnings", "regulatory", "macro", "contract", "other"]

EVENT_PROMPT = """You are a financial event classifier. Given this headline, respond with ONLY ONE tag from:
[earnings, regulatory, macro, contract, other]
Headline: "{headline}"
Tag:"""


def get_cache_key(headline: str, model: str) -> str:
    return hashlib.md5(f"{model}:{headline.strip()}".encode("utf-8")).hexdigest()


class LocalLLMEventTagger:
    """Classifies headlines using local Ollama model (temperature 0) with persistent disk cache."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434/api/generate",
        cache_path: str = "data/processed/llm_event_cache.json",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        os.makedirs(self.cache_path.parent, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def is_ollama_available(self) -> bool:
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=1.0)
            return res.status_code == 200
        except Exception:
            return False

    def tag_headline(self, headline: str) -> str:
        key = get_cache_key(headline, self.model_name)
        if key in self.cache:
            return self.cache[key]

        # If Ollama is available, call it with temperature 0
        if self.is_ollama_available():
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": EVENT_PROMPT.format(headline=headline),
                    "stream": False,
                    "options": {"temperature": 0.0},
                }
                resp = requests.post(self.ollama_url, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    text = resp.json().get("response", "").strip().lower()
                    for tag in EVENT_TAGS:
                        if tag in text:
                            self.cache[key] = tag
                            self._save_cache()
                            return tag
            except Exception:
                pass

        # Deterministic regex/keyword fallback if Ollama is not installed
        h_lower = headline.lower()
        if any(w in h_lower for w in ["profit", "loss", "revenue", "ebitda", "q1", "q2", "q3", "q4", "earnings", "results"]):
            tag = "earnings"
        elif any(w in h_lower for w in ["sebi", "rbi", "penalty", "notice", "compliance", "court", "probe"]):
            tag = "regulatory"
        elif any(w in h_lower for w in ["inflation", "gdp", "repo", "interest", "budget", "crude", "rupee"]):
            tag = "macro"
        elif any(w in h_lower for w in ["contract", "order", "deal", "partnership", "wins", "acquisition"]):
            tag = "contract"
        else:
            tag = "other"

        self.cache[key] = tag
        self._save_cache()
        return tag

    def tag_headlines_batch(self, headlines: List[str]) -> List[str]:
        return [self.tag_headline(h) for h in headlines]

