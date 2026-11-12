"""Функція 5. Executive summary для стейкхолдера.

На вхід — набір уже порахованих метрик (dict / JSON-файл) або DataFrame,
з якого `compute_kpis` порахує стандартні KPI за два періоди.
LLM пише короткий бізнес-текст; всі числа в ньому перевіряються.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import prompts
from .data import DataError, infer_schema
from .llm import LLMClient
from .verify import verify_numbers


def _period_kpis(d: pd.DataFrame, conv: str | None, rev: str | None, user: str | None) -> dict[str, float]:
    k: dict[str, float] = {"rows": len(d)}
    if user:
        k["unique_users"] = int(d[user].nunique())
    if conv:
        k["conversions"] = int(d[conv].sum())
        k["conversion_rate"] = round(float(d[conv].mean()), 4)
    if rev:
        k["revenue"] = round(float(d[rev].sum()), 2)
        if conv and d[conv].sum():
            k["aov"] = round(float(d[rev].sum() / d[conv].sum()), 2)
        if user and d[user].nunique():
            k["revenue_per_user"] = round(float(d[rev].sum() / d[user].nunique()), 2)
    if "added_to_cart" in d.columns:
        k["add_to_cart_rate"] = round(float(d["added_to_cart"].mean()), 4)
    return k


def compute_kpis(df: pd.DataFrame, current: tuple[str, str], baseline: tuple[str, str] | None = None) -> dict[str, Any]:
    schema = infer_schema(df)
    if not schema.date_col:
        raise DataError("Для KPI за періоди потрібна колонка з датою.")
    conv = next((c for c in schema.binary_metrics if "convert" in c.lower() or "purchase" in c.lower()), None)
    rev = next((c for c in schema.value_metrics if any(k in c.lower() for k in ("revenue", "amount", "gmv"))), None)
    user = next((c for c in schema.id_cols if "user" in c.lower() or "customer" in c.lower()), None)

    c0, c1 = pd.Timestamp(current[0]), pd.Timestamp(current[1])
    if baseline is None:
        length = c1 - c0 + pd.Timedelta(days=1)
        b0, b1 = c0 - length, c0 - pd.Timedelta(days=1)
    else:
        b0, b1 = pd.Timestamp(baseline[0]), pd.Timestamp(baseline[1])
    day = df[schema.date_col].dt.normalize()
    cur = _period_kpis(df[(day >= c0) & (day <= c1)], conv, rev, user)
    base = _period_kpis(df[(day >= b0) & (day <= b1)], conv, rev, user)
    if not cur["rows"] or not base["rows"]:
        raise DataError("Немає даних за один із періодів.")
    kpis = {}
    for k in cur:
        kpis[k] = {"current": cur[k], "previous": base[k],
                   "relative_change": round(cur[k] / base[k] - 1, 4) if base[k] else None}
        if k.endswith("_rate"):
            kpis[k]["change_pp"] = round((cur[k] - base[k]) * 100, 2)
    return {"current_period": f"{c0:%Y-%m-%d}..{c1:%Y-%m-%d}", "previous_period": f"{b0:%Y-%m-%d}..{b1:%Y-%m-%d}",
            "kpis": kpis}


def drivers_from_anomaly(anomaly_result: dict[str, Any]) -> dict[str, Any]:
    """Стискає результат explain_anomaly до того, що варто показати стейкхолдеру."""
    d = anomaly_result["decomposition"]
    out = {"metric": d["summary"]["metric"], "primary_driver": d["primary_candidate"],
           "metric_without_segment": d["metric_without_segment"]}
    if anomaly_result.get("interpretation"):
        out["analyst_hypotheses"] = [
            {"cause": h["cause"], "confidence": h["confidence"]}
            for h in anomaly_result["interpretation"]["cause_hypotheses"]
        ]
    return out


def executive_summary(metrics: dict[str, Any], llm: LLMClient, drivers: dict[str, Any] | None = None,
                      audience: str = "CEO and product leadership") -> dict[str, Any]:
    if not metrics:
        raise DataError("Порожній набір метрик — нема про що писати summary.")
    result: dict[str, Any] = {"metrics": metrics, "drivers": drivers, "summary": None, "verification": None}
    if not llm.available:
        return result
    resp = llm.call_json(
        task="executive_summary",
        system=prompts.SUMMARY_SYSTEM.format(audience=audience),
        user=prompts.SUMMARY_USER.format(
            kpis=json.dumps(metrics, ensure_ascii=False, indent=1, default=str),
            drivers=json.dumps(drivers or {"note": "no driver analysis provided"}, ensure_ascii=False, indent=1,
                               default=str),
        ),
        schema=prompts.SUMMARY_SCHEMA,
        effort="low",
        nonempty=("what_happened", "next_steps"),
    )
    result["summary"] = resp
    result["verification"] = verify_numbers(json.dumps(resp, ensure_ascii=False), [metrics, drivers], derived=True)
    return result
