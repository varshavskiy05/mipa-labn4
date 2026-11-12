"""Власна функція №6: програмна перевірка чисел у тексті LLM (anti-hallucination).

Ідея: LLM ніколи не рахує — вона лише інтерпретує числа, які порахував pandas.
Але модель може «округлити по-своєму», переплутати сегменти або вигадати число.
Тому кожен згенерований текст проганяється через `verify_numbers`:

1. з тексту витягуються всі числа (з одиницями: %, п.п., x/разів, тис/k, млн);
2. з набору фактів (dict/list із pandas) збирається множина «дозволених» значень,
   включно з похідними: частка -> відсоток, а також (опційно) різниці,
   відношення та відносні зміни між парами фактів;
3. число вважається підтвердженим, якщо збігається з дозволеним значенням
   з точністю до округлення, яке видно в тексті (+ 1.5% відносної похибки).

Непідтверджені числа не обов'язково хибні, але їх слід перевірити вручну —
звіт позначає їх ⚠️.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Шаблони, які НЕ є метриками: ISO-дати, версії, роки, тижні.
_IGNORE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",  # дати
    r"\b\d{4}-\d{2}\b",                                    # рік-місяць
    r"\b\d+\.\d+\.\d+\b",                                  # версії 5.4.0
    r"\b(?:19|20)\d{2}\b(?![.,]\d)",                       # роки
    r"\b[A-Z]{1,3}\d+\b",                                  # ID на кшталт F12, W42
    r"\b\d{1,2}:\d{2}\b",                                  # час
    # категорії-діапазони («35-44», «55+») — назви сегментів, а не метрики
    r"(?<![\d.,])\d{1,3}\s?[-–]\s?\d{1,3}(?![\d.,%])",
    r"\b\d{1,3}\+",
    # тривалості періодів («останні 14 днів», «2 тижні») — параметри аналізу, а не метрики
    r"\b\d+\s*(?:дн(?:і|ів|я)|день|доби|тижн\w*|місяц\w*|days?|weeks?|months?)\b",
]

_NUM_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-−+])?(?P<num>\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"\s*(?P<unit>%|п\.\s?п\.?|pp|в\.п\.|x\b|×|рази|разів|раза|k\b|K\b|тис\.?|млн|m\b|M\b)?",
)


@dataclass
class Claim:
    raw: str
    value: float
    unit: str
    decimals: int
    context: str


@dataclass
class VerificationResult:
    verified: list[Claim] = field(default_factory=list)
    unverified: list[Claim] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.verified) + len(self.unverified)

    @property
    def ok(self) -> bool:
        return not self.unverified

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.total,
            "verified": len(self.verified),
            "unverified": [{"number": c.raw, "context": c.context} for c in self.unverified],
        }

    def badge(self) -> str:
        if self.total == 0:
            return "ℹ️ чисел у тексті немає"
        if self.ok:
            return f"✅ усі {self.total} чисел підтверджені даними"
        return f"⚠️ {len(self.unverified)} з {self.total} чисел не знайдено у даних — перевірте вручну"


def _normalize_unit(u: str | None) -> str:
    if not u:
        return ""
    u = u.lower().replace(" ", "").replace(" ", "")
    if u == "%":
        return "%"
    if u.startswith("п.п") or u in {"pp", "в.п."}:
        return "pp"
    if u in {"x", "×", "рази", "разів", "раза"}:
        return "x"
    if u in {"k", "тис", "тис."}:
        return "k"
    if u in {"млн", "m"}:
        return "m"
    return ""


def extract_claims(text: str) -> list[Claim]:
    masked = text
    for pat in _IGNORE_PATTERNS:
        masked = re.sub(pat, lambda m: " " * len(m.group(0)), masked)
    claims = []
    for m in _NUM_RE.finditer(masked):
        num = m.group("num").replace(" ", "").replace(" ", "").replace(" ", "")
        if re.fullmatch(r"\d{1,3}(,\d{3})+", num):
            num = num.replace(",", "")
        num = num.replace(",", ".")
        value = float(num)
        if m.group("sign") in {"-", "−"}:
            value = -value
        unit = _normalize_unit(m.group("unit"))
        decimals = len(num.split(".")[1]) if "." in num else 0
        if unit == "k":
            value *= 1_000
        elif unit == "m":
            value *= 1_000_000
        # дрібні цілі без одиниць — нумерація, «3 сегменти», «топ-5»: не перевіряємо
        if unit == "" and decimals == 0 and abs(value) <= 10:
            continue
        s, e = max(0, m.start() - 50), min(len(text), m.end() + 30)
        claims.append(Claim(m.group(0).strip(), value, unit, decimals, text[s:e].replace("\n", " ")))
    return claims


def flatten_numbers(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_numbers(v))
    elif isinstance(obj, str):
        for c in extract_claims(obj):
            out.append(c.value)
    else:  # numpy-скаляри
        try:
            f = float(obj)
            if math.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def _candidates(values: list[float], derived: bool) -> dict[str, np.ndarray]:
    base = set(values)
    direct = base | {v * 100 for v in base} | {v / 100 for v in base} | {abs(v) for v in base}
    cands = {"direct": direct, "ratio": set(), "pp": set(), "rel": set(), "diff": set()}
    if derived and len(base) <= 600:
        for a, b in itertools.permutations(base, 2):
            if b != 0:
                cands["ratio"].add(a / b)
                cands["rel"].add((a - b) / abs(b) * 100)
            cands["pp"].add((a - b) * 100)
            cands["diff"].add(a - b)
        cands["pp"] |= cands["diff"]
    return {k: np.fromiter(v, dtype=float) for k, v in cands.items()}


def _matches(claim: Claim, pool: np.ndarray) -> bool:
    if pool.size == 0:
        return False
    half_unit = 0.5 * 10 ** (-claim.decimals) if claim.unit not in {"k", "m"} else abs(claim.value) * 0.05
    tol = np.maximum(half_unit * 1.0001, np.abs(pool) * 0.015)
    return bool((np.abs(abs(claim.value) - np.abs(pool)) <= tol).any())


def verify_numbers(text: str, facts: Any, derived: bool = True) -> VerificationResult:
    """Перевіряє, що кожне число у `text` випливає з `facts`."""
    values = flatten_numbers(facts)
    cands = _candidates(values, derived)
    res = VerificationResult()
    for claim in extract_claims(text):
        if claim.unit == "x":
            pools = [cands["ratio"], cands["direct"]]
        elif claim.unit == "pp":
            pools = [cands["pp"], cands["direct"]]
        elif claim.unit == "%":
            pools = [cands["direct"], cands["rel"], cands["pp"]]
        else:
            # абсолютна різниця двох фактів («+465 рядків») допустима і без одиниць
            pools = [cands["direct"], cands["diff"]]
        if any(_matches(claim, p) for p in pools):
            res.verified.append(claim)
        else:
            res.unverified.append(claim)
    return res
