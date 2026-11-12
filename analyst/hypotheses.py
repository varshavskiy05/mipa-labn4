"""Функція 2. Генерація гіпотез, обґрунтованих цифрами з датасету.

Пайплайн:
  1) `build_facts` — pandas + scipy перебирають розрізи і формують список фактів
     з id (F1, F2, ...): розриви між сегментами зі стат. тестом, аномальні дні,
     тренди, тижнева сезонність, кореляції, проблеми якості;
  2) LLM обирає найцінніші факти й формулює 5-10 гіпотез, посилаючись на id;
  3) кожна гіпотеза перевіряється: id фактів існують, а всі числа в тексті
     випливають із процитованих фактів (verify_numbers).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import prompts
from .data import Schema, infer_schema, schema_description, to_py
from .eda import detect_quality_issues, profile_dataframe
from .llm import LLMClient
from .verify import verify_numbers

MIN_SEGMENT = 200      # мінімальний розмір сегмента
ALPHA = 0.001          # жорсткий поріг через множинні порівняння
MIN_LIFT = 0.15        # мінімальне відносне відхилення від решти (15%)


def _metric_defs(df: pd.DataFrame, schema: Schema) -> list[dict[str, Any]]:
    """Метрики для скринінгу: частки (binary) і середні (value)."""
    defs = []
    for c in schema.binary_metrics:
        defs.append({"name": f"{c}_rate", "col": c, "kind": "rate"})
    for c in schema.value_metrics:
        defs.append({"name": f"avg_{c}", "col": c, "kind": "mean"})
    # спеціальний випадок: середній чек серед конверсій (якщо є converted + revenue-подібна колонка)
    conv = next((c for c in schema.binary_metrics if "convert" in c.lower() or "purchase" in c.lower()), None)
    rev = next((c for c in schema.value_metrics if any(k in c.lower() for k in ("revenue", "amount", "gmv", "price"))), None)
    if conv and rev:
        defs.append({"name": f"{rev}_per_conversion", "col": rev, "kind": "mean", "filter": conv})
    return defs


def _segment_facts(df: pd.DataFrame, schema: Schema) -> list[dict[str, Any]]:
    facts = []
    dims = [d for d in schema.dimensions if 2 <= df[d].nunique() <= 30]
    for m in _metric_defs(df, schema):
        data = df[df[m["filter"]] == 1] if "filter" in m else df
        if data.empty:
            continue
        overall = float(data[m["col"]].mean())
        for d in dims:
            if d == m["col"]:
                continue
            g = data.groupby(d, observed=True)[m["col"]]
            agg = g.agg(["mean", "count", "sum", "std"])
            for seg, row in agg.iterrows():
                n = int(row["count"])
                if n < MIN_SEGMENT:
                    continue
                rest = data.loc[data[d] != seg, m["col"]].dropna()
                if len(rest) < MIN_SEGMENT:
                    continue
                seg_val, rest_val = float(row["mean"]), float(rest.mean())
                if rest_val == 0:
                    continue
                lift = seg_val / rest_val - 1
                if abs(lift) < MIN_LIFT:
                    continue
                if m["kind"] == "rate":
                    k1, n1, k2, n2 = row["sum"], n, rest.sum(), len(rest)
                    p = (k1 + k2) / (n1 + n2)
                    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
                    pval = 2 * stats.norm.sf(abs((k1 / n1 - k2 / n2) / se))
                else:
                    seg_vals = data.loc[data[d] == seg, m["col"]].dropna()
                    pval = stats.mannwhitneyu(seg_vals, rest).pvalue
                if pval > ALPHA:
                    continue
                facts.append({
                    "type": "segment_gap", "metric": m["name"], "dimension": d, "segment": str(seg),
                    "segment_value": round(seg_val, 4), "rest_value": round(rest_val, 4),
                    "overall_value": round(overall, 4), "ratio_vs_rest": round(seg_val / rest_val, 2),
                    "segment_size": n, "segment_share_of_rows": round(n / len(data), 4),
                    "p_value": float(f"{pval:.2g}"),
                })
    # прибираємо дублікати: той самий набір рядків у різних розрізах (platform=web == app_version=web)
    seen, unique = set(), []
    for f in facts:
        key = (f["metric"], f["segment_size"], f["segment_value"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    facts = unique
    # сортуємо за «вагою»: відхилення * розмір
    facts.sort(key=lambda f: abs(np.log(f["ratio_vs_rest"])) * np.sqrt(f["segment_size"]), reverse=True)
    return facts[:40]


def _time_facts(df: pd.DataFrame, schema: Schema) -> list[dict[str, Any]]:
    if not schema.date_col:
        return []
    facts = []
    day = df[schema.date_col].dt.normalize()
    daily = df.groupby(day).size()
    med = daily.median()
    mad = (daily - med).abs().median() or 1
    z = (daily - med) / (1.4826 * mad)
    for d in daily.index[z.abs() > 4]:
        on_day = df[day == d]
        others = df[day != d]
        fact = {"type": "volume_anomaly", "date": d.strftime("%Y-%m-%d"), "weekday": d.day_name(),
                "rows": int(daily[d]), "typical_daily_rows": to_py(med), "robust_z": round(float(z[d]), 1)}
        # що змінилось у складі трафіку цього дня
        shifts = []
        for dim in schema.dimensions:
            if df[dim].nunique() > 30:
                continue
            s_day = on_day[dim].astype(str).value_counts(normalize=True)
            s_oth = others[dim].astype(str).value_counts(normalize=True)
            diff = (s_day - s_oth.reindex(s_day.index).fillna(0)).sort_values(ascending=False)
            if len(diff) and diff.iloc[0] > 0.1:
                seg = diff.index[0]
                shifts.append({"dimension": dim, "segment": seg, "share_on_day": round(float(s_day[seg]), 3),
                               "share_other_days": round(float(s_oth.get(seg, 0)), 3)})
        fact["composition_shifts"] = shifts[:4]
        for m in schema.binary_metrics[:4]:
            fact[f"{m}_rate_on_day"] = round(float(on_day[m].mean()), 4)
            fact[f"{m}_rate_other_days"] = round(float(others[m].mean()), 4)
        for m in schema.value_metrics[:3]:
            fact[f"median_{m}_on_day"] = to_py(on_day[m].median())
            fact[f"median_{m}_other_days"] = to_py(others[m].median())
        facts.append(fact)

    # тренд: останні 14 днів проти попередніх 14
    end = day.max()
    last = df[day > end - pd.Timedelta(days=14)]
    prev = df[(day <= end - pd.Timedelta(days=14)) & (day > end - pd.Timedelta(days=28))]
    if len(last) and len(prev):
        for m in schema.binary_metrics:
            a, b = last[m].mean(), prev[m].mean()
            if b and abs(a / b - 1) > 0.08:
                facts.append({"type": "trend", "metric": f"{m}_rate",
                              "last_14d": f"{(end - pd.Timedelta(days=13)):%Y-%m-%d}..{end:%Y-%m-%d}",
                              "last_14d_value": round(float(a), 4), "previous_14d_value": round(float(b), 4),
                              "relative_change": round(float(a / b - 1), 3)})
        r_last, r_prev = len(last) / 14, len(prev) / 14
        if abs(r_last / r_prev - 1) > 0.08:
            facts.append({"type": "trend", "metric": "rows_per_day", "last_14d_value": round(r_last, 1),
                          "previous_14d_value": round(r_prev, 1), "relative_change": round(r_last / r_prev - 1, 3)})

    # сезонність: вихідні проти буднів
    weekend = df[schema.date_col].dt.day_of_week >= 5
    n_days_we = day[weekend].nunique() or 1
    n_days_wd = day[~weekend].nunique() or 1
    fact = {"type": "weekly_pattern",
            "rows_per_weekend_day": round(weekend.sum() / n_days_we, 1),
            "rows_per_weekday": round((~weekend).sum() / n_days_wd, 1)}
    for m in schema.binary_metrics[:4]:
        fact[f"{m}_rate_weekend"] = round(float(df.loc[weekend, m].mean()), 4)
        fact[f"{m}_rate_weekday"] = round(float(df.loc[~weekend, m].mean()), 4)
    facts.append(fact)
    return facts


def _correlation_facts(df: pd.DataFrame, schema: Schema) -> list[dict[str, Any]]:
    cols = schema.value_metrics + schema.binary_metrics
    if len(cols) < 2:
        return []
    corr = df[cols].corr(method="spearman")
    facts = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.3 and not (a in schema.binary_metrics and b in schema.binary_metrics and abs(r) > 0.95):
                facts.append({"type": "correlation", "a": a, "b": b, "spearman_r": round(float(r), 2)})
    return sorted(facts, key=lambda f: -abs(f["spearman_r"]))[:8]


def build_facts(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema = infer_schema(df)
    prof = profile_dataframe(df, schema)
    facts = (_segment_facts(df, schema) + _time_facts(df, schema) + _correlation_facts(df, schema))
    for q in detect_quality_issues(df, prof, schema):
        facts.append({"type": "data_quality", **{k: v for k, v in q.items() if k != "id"}})
    for i, f in enumerate(facts, 1):
        f["id"] = f"F{i}"
    return facts


def suggest_hypotheses(df: pd.DataFrame, llm: LLMClient, n_min: int = 5, n_max: int = 10) -> dict[str, Any]:
    facts = build_facts(df)
    result: dict[str, Any] = {"facts": facts, "hypotheses": None}
    if not llm.available:
        return result
    if not facts:
        result["hypotheses"] = []
        result["note"] = "Скринінг не знайшов статистично значущих ефектів — гіпотез не генеруємо."
        return result

    resp = llm.call_json(
        task="suggest_hypotheses",
        system=prompts.HYPOTHESES_SYSTEM.format(n_min=n_min, n_max=n_max),
        user=prompts.HYPOTHESES_USER.format(
            schema=schema_description(df),
            facts="\n".join(json.dumps(f, ensure_ascii=False, default=str) for f in facts),
        ),
        schema=prompts.HYPOTHESES_SCHEMA,
        effort="medium",
        nonempty=("hypotheses",),
    )
    by_id = {f["id"]: f for f in facts}
    for h in resp["hypotheses"]:
        cited = [by_id[i] for i in h["evidence_fact_ids"] if i in by_id]
        h["unknown_fact_ids"] = [i for i in h["evidence_fact_ids"] if i not in by_id]
        text = " ".join([h["observation"], h["hypothesis"], h["business_impact"]])
        # числа мають випливати саме з процитованих фактів, а не з будь-яких
        h["verification"] = verify_numbers(text, cited, derived=True)
        h["grounded"] = bool(cited) and not h["unknown_fact_ids"]
    result["hypotheses"] = resp["hypotheses"]
    return result
