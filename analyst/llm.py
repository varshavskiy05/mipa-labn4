"""Тонка обгортка над LLM (Claude): структурований JSON-вихід + логування.

Підтримуються два способи доступу до Claude (обирається автоматично за ключем):
  * ANTHROPIC_API_KEY  -> офіційний Anthropic SDK, structured outputs через
                          output_config.format + server-side fallback;
  * OPENROUTER_API_KEY -> Claude через OpenRouter (OpenAI-сумісний API),
                          structured outputs через response_format=json_schema.
Ключ можна покласти у змінну середовища або у файл .env у корені проєкту.

Кожен виклик:
  * отримує system prompt, user prompt і JSON Schema відповіді;
  * повертає dict, валідний за схемою — без парсингу «тексту з числами»;
  * логується у logs/llm_calls.jsonl (промпт, відповідь, токени).

Якщо ключа немає або offline=True — `available` == False, і функції асистента
повертають лише детерміновані (pandas) результати.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "llm_calls.jsonl"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Мінімальний завантажувач .env (KEY=VALUE). Не перезаписує вже задані змінні."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        elif line.startswith("sk-or-"):  # «голий» ключ OpenRouter
            os.environ.setdefault("OPENROUTER_API_KEY", line)
        elif line.startswith("sk-ant-"):
            os.environ.setdefault("ANTHROPIC_API_KEY", line)


load_dotenv()
DEFAULT_MODEL = os.environ.get("ANALYST_MODEL", "claude-opus-5-5")


DEGENERATE_RETRY_NOTE = (
    "\n\nIMPORTANT: your previous answer contained placeholder values such as '...' or empty strings. "
    "Fill every field with real, specific content based on the data above."
)


# порожній рядок заглушкою НЕ вважаємо: напр. code="" — легітимна відповідь при feasible=false
_PLACEHOLDER = re.compile(r"[.…\-—_*]+|(?i:n/?a|none|null|todo|tbd)")


def _string_leaves(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _string_leaves(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _string_leaves(v)]
    return []


def is_degenerate(data: Any, threshold: float = 0.4) -> bool:
    """True, якщо значна частка текстових полів — заглушки ('...', '…', '', 'N/A').

    Structured outputs гарантують лише відповідність схемі, а не змістовність:
    слабкі моделі іноді повертають {"what_happened": "...", ...}.
    """
    leaves = _string_leaves(data)
    if not leaves:
        return False
    junk = sum(1 for s in leaves if _PLACEHOLDER.fullmatch(s.strip()))
    return junk / len(leaves) >= threshold


class LLMUnavailable(RuntimeError):
    """LLM недоступна (немає ключа / офлайн-режим)."""


class LLMRefusal(RuntimeError):
    """Модель відмовилась відповідати."""


def _detect_provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return None


def _openrouter_model(model: str) -> str:
    """claude-opus-5-5 -> anthropic/claude-opus-5.5 (ID у каталозі OpenRouter)."""
    if "/" in model:
        return model
    return "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    offline: bool = False
    log_path: Path = LOG_PATH
    provider: str | None = field(default=None, init=False)
    total_input_tokens: int = field(default=0, init=False)
    total_output_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._client = None
        if self.offline:
            return
        self.provider = _detect_provider()
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif self.provider == "openrouter":
            from openai import OpenAI

            self._client = OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"])
            self.model = _openrouter_model(self.model)

    @property
    def available(self) -> bool:
        return self._client is not None

    def call_json(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        effort: str = "medium",
        max_tokens: int | None = None,
        nonempty: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Викликає модель і повертає dict, що відповідає `schema`.

        `nonempty` — ключі верхнього рівня, які не можуть бути порожніми списками
        (напр. quality_findings при знайдених проблемах якості): схема це не гарантує.
        """
        max_tokens = max_tokens or int(os.environ.get("ANALYST_MAX_TOKENS", 16000))
        if not self.available:
            raise LLMUnavailable("LLM недоступна: задайте ANTHROPIC_API_KEY або OPENROUTER_API_KEY (.env).")
        prompt = user
        for attempt in range(1, 3):
            t0 = time.time()
            if self.provider == "anthropic":
                text, model, tin, tout = self._call_anthropic(task, system, prompt, schema, effort, max_tokens)
            else:
                text, model, tin, tout = self._call_openrouter(task, system, prompt, schema, effort, max_tokens)
            elapsed = time.time() - t0
            data = json.loads(text)
            self.total_input_tokens += tin
            self.total_output_tokens += tout
            degenerate = is_degenerate(data) or any(not data.get(k) for k in nonempty)
            self._log(task, system, prompt, data, model, tin, tout, elapsed, degenerate=degenerate)
            if not degenerate:
                return data
            # JSON валідний за схемою, але поля заповнені заглушками ("...") — просимо ще раз
            prompt = user + DEGENERATE_RETRY_NOTE
        print(f"⚠️  {task}: модель двічі повернула порожні/заглушкові поля.", file=sys.stderr)
        return data

    # ---------------------------------------------------------------- бекенди

    def _call_anthropic(self, task, system, user, schema, effort, max_tokens):
        # server-side fallback: якщо класифікатор безпеки відхилить запит,
        # API сам повторить його на резервній моделі в межах того ж виклику.
        r = self._client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if r.stop_reason == "refusal":
            raise LLMRefusal(f"Модель відмовилась відповідати ({task}).")
        if r.stop_reason == "max_tokens":
            raise RuntimeError(f"Відповідь обрізана по max_tokens ({task}).")
        text = next(b.text for b in r.content if b.type == "text")
        return text, r.model, r.usage.input_tokens, r.usage.output_tokens

    def _call_openrouter(self, task, system, user, schema, effort, max_tokens):
        r = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": task, "strict": True, "schema": schema}},
            extra_body={"reasoning": {"effort": effort}},
        )
        choice = r.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(f"Відповідь обрізана по max_tokens ({task}).")
        if not choice.message.content:
            raise LLMRefusal(f"Порожня відповідь моделі ({task}, finish_reason={choice.finish_reason}).")
        u = r.usage
        return choice.message.content, r.model, u.prompt_tokens if u else 0, u.completion_tokens if u else 0

    def _log(self, task, system, user, data, model, tin, tout, elapsed, degenerate=False) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "provider": self.provider,
            "model": model,
            "elapsed_sec": round(elapsed, 2),
            "input_tokens": tin,
            "output_tokens": tout,
            "degenerate": degenerate,
            "system": system,
            "user": user,
            "response": data,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
