"""Рендеринг результатів у Markdown (для консолі та файлів-звітів)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

OFFLINE_NOTE = "> ℹ️ LLM вимкнена (офлайн-режим або немає ANTHROPIC_API_KEY) — показано лише розрахунки pandas.\n"
SEV = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def _pct(x: float | None, digits: int = 2, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def _cell(s: str) -> str:
    """Екранує | у комірках Markdown-таблиці."""
    return str(s).replace("|", "\\|")


def _img(path: str, base: Path | None) -> str:
    rel = os.path.relpath(path, base).replace("\\", "/") if base else path
    return f"![{Path(path).stem}]({rel})"


def _badge(v) -> str:
    return f"\n**Перевірка чисел:** {v.badge()}\n" + "".join(
        f"- ⚠️ `{u['number']}` — «…{u['context']}…»\n" for u in v.to_dict()["unverified"]) if v else ""


# ------------------------------------------------------------------ EDA

def render_eda(r: dict[str, Any], base: Path | None = None) -> str:
    p = r["profile"]
    out = ["## Автоматичний EDA (функція 1)\n",
           f"**Розмір:** {p['n_rows']:,} рядків × {p['n_columns']} колонок, {p['memory_mb']} MB. "
           f"**Повних дублікатів:** {p['duplicate_rows']}.\n".replace(",", " "),
           "**Ролі колонок (визначено автоматично):** " + "; ".join(
               f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in p["roles"].items() if v) + "\n",
           "| колонка | тип | пропуски | унікальних | статистики |", "|---|---|---|---|---|"]
    for c, col in p["columns"].items():
        if "top_values" in col:
            st = ", ".join(f"{k}: {_pct(v, 1)}" for k, v in list(col["top_values"].items())[:4])
        elif "median" in col:
            st = f"min {col['min']}, median {col['median']}, mean {col['mean']}, p99 {col['p99']}, max {col['max']}"
        elif "mean" in col:
            st = f"частка 1: {_pct(col['mean'])}"
        else:
            st = f"{col.get('min')} .. {col.get('max')} ({col.get('n_days')} днів)"
        out.append(f"| `{c}` | {col['dtype']} | {col['nulls']} ({_pct(col['null_share'], 1)}) | {col['unique']} | {st} |")
    out.append("\n### Проблеми якості (rule-based, pandas)\n")
    for q in r["issues"]:
        nums = ", ".join(f"{k}={v}" for k, v in q.items() if k not in {"id", "type", "column", "detail"})
        out.append(f"- **{q['id']}** `{q['column'] or '*'}` — {q['detail']} ({nums})")
    out.extend(_img(c, base) for c in r["charts"])
    it = r["interpretation"]
    if not it:
        out.append("\n" + OFFLINE_NOTE)
        return "\n".join(out)
    out += ["\n### Інтерпретація (LLM)\n", f"**Що це за дані:** {it['dataset_overview']}\n",
            f"**Один рядок =** {it['row_grain']}\n", "**Ключові колонки:**"]
    out += [f"- `{c['column']}`: {c['insight']}" for c in it["column_insights"]]
    out.append("\n**Висновки щодо якості даних:**\n")
    out.append("| | issues | висновок | помилка? | рекомендація |\n|---|---|---|---|---|")
    for f in it["quality_findings"]:
        bad = f" ⚠️ невідомі id: {f['unknown_issue_ids']}" if f.get("unknown_issue_ids") else ""
        out.append(f"| {SEV[f['severity']]} | {', '.join(f['issue_ids'])}{bad} | {f['finding']} | "
                   f"{'так' if f['is_likely_error'] else 'ні'} | {f['recommendation']} |")
    if it["extra_observations"]:
        out.append("\n**Додаткові спостереження:**")
        out += [f"- {x}" for x in it["extra_observations"]]
    out.append("\n**Наступні кроки:**")
    out += [f"1. {x}" for x in it["next_steps"]]
    out.append(_badge(r["verification"]))
    return "\n".join(out)


# ------------------------------------------------------------------ гіпотези

def render_hypotheses(r: dict[str, Any]) -> str:
    out = ["## Гіпотези для перевірки (функція 2)\n",
           f"Скринінг pandas/scipy знайшов **{len(r['facts'])} фактів** "
           f"(розриви між сегментами з p < 0.001 і відхиленням ≥ 15%, аномальні дні, тренди, кореляції, якість даних).\n"]
    hs = r["hypotheses"]
    if hs is None:
        out.append(OFFLINE_NOTE)
        out.append("Топ-10 фактів:\n")
        out += [f"- **{f['id']}** `{f['type']}` " + ", ".join(f"{k}={v}" for k, v in f.items()
                                                              if k not in {"id", "type"} and not isinstance(v, list))
                for f in r["facts"][:10]]
        return "\n".join(out)
    if not hs:
        out.append(r.get("note", "Гіпотез немає."))
    for i, h in enumerate(hs, 1):
        v = h["verification"]
        flag = "" if h["grounded"] else " ⚠️ посилається на неіснуючі факти"
        out += [f"### H{i}. {h['title']} {SEV[h['priority']]}{flag}\n",
                f"*Категорія:* {h['category']} · *пріоритет:* {h['priority']} · *впевненість:* {h['confidence']} · "
                f"*факти:* {', '.join(h['evidence_fact_ids'])} · *числа:* {v.badge()}\n",
                f"- **Спостереження:** {h['observation']}",
                f"- **Гіпотеза:** {h['hypothesis']}",
                f"- **Чому важливо:** {h['business_impact']}",
                f"- **Як перевірити:** {h['how_to_test']}"]
        out += [f"  - ⚠️ `{u['number']}` не знайдено в процитованих фактах" for u in v.to_dict()["unverified"]]
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ NL -> pandas

def render_ask(r) -> str:
    out = [f"### ❓ {r.question}\n"]
    if not r.feasible:
        out.append(f"🚫 **Неможливо відповісти на основі даних.** {r.explanation}")
        return "\n".join(out)
    out += ["```python", r.code, "```",
            f"*Логіка:* {r.explanation}"]
    if r.assumptions:
        out.append("*Припущення:* " + "; ".join(r.assumptions))
    if r.attempts > 1:
        out.append(f"*Код виправлено з {r.attempts}-ї спроби.*")
    if r.error:
        out.append(f"\n❌ **Помилка:** `{r.error}`")
        return "\n".join(out)
    out += ["\n**Результат:**\n", "```", r.result_text, "```"]
    if r.answer:
        out.append(f"\n**Відповідь:** {r.answer}")
        out += [f"- _{c}_" for c in r.caveats]
        out.append(f"\n**Перевірка чисел:** {r.verification.badge()}")
    return "\n".join(out)


# ------------------------------------------------------------------ аномалії

def _fmt_val(v, additive: bool) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f}".replace(",", " ") if additive or abs(v) > 1 else _pct(v)


def render_anomaly(r: dict[str, Any], base: Path | None = None) -> str:
    d = r["decomposition"]
    s = d["summary"]
    additive = s["metric"].startswith(("sum:", "count")) or s["metric"] in {"rows", "sessions"}
    rate_like = not additive and abs(s["baseline_value"]) <= 1
    chg = f"{s['absolute_change'] * 100:+.2f} п.п." if rate_like else f"{s['absolute_change']:+,.2f}".replace(",", " ")
    out = ["## Пояснення аномалії (функція 4)\n",
           f"**Метрика:** `{s['metric']}` ({s['metric_label']}). "
           f"**Базовий період:** {s['baseline_period']} → **поточний:** {s['current_period']}\n",
           f"**{_fmt_val(s['baseline_value'], not rate_like)} → {_fmt_val(s['current_value'], not rate_like)}** "
           f"({chg}, {_pct(s['relative_change'], 1, True)} відносно)\n",
           "### Які розрізи найкраще пояснюють зміну\n",
           "| розріз | концентрація* | головний сегмент | внесок | частка зміни |", "|---|---|---|---|---|"]
    for x in d["dimensions_ranked"] + d["dimension_pairs_ranked"]:
        t = x["top_segments"][0]
        contrib = f"{t['contribution'] * 100:+.2f} п.п." if rate_like else f"{t['contribution']:+,.1f}".replace(",", " ")
        out.append(f"| {' × '.join(x['dimensions'])} | {x['concentration']:.2f} | {_cell(t['segment'])} | {contrib} | "
                   f"{_pct(t['share_of_change'], 0)} |")
    out.append("\n\\* частка загальної зміни, яку пояснює найбільший сегмент розрізу (може бути > 1, "
               "якщо інші сегменти рухались у протилежний бік).\n")
    pc = d["primary_candidate"]
    seg = pc["segment"]
    out.append(f"**Головний кандидат:** `{seg['segment']}` — значення {_fmt_val(seg['base_value'], not rate_like)} → "
               f"{_fmt_val(seg['cur_value'], not rate_like)}, n: {seg['base_n']} → {seg['cur_n']}"
               + (", **новий сегмент** (не існував у базовому періоді)" if seg.get("new_segment") else ""))
    for cf in d["metric_without_segment"]:
        out.append(f"- Без `{cf['excluded_segment']}` метрика змінилась би на "
                   f"**{_pct(cf['relative_change_without_segment'], 1, True)}** (замість {_pct(s['relative_change'], 1, True)})")
    out.extend("\n" + _img(c, base) for c in r["charts"])
    it = r["interpretation"]
    if not it:
        out.append("\n" + OFFLINE_NOTE)
        return "\n".join(out)
    pdrv = it["primary_driver"]
    out += ["\n### Інтерпретація (LLM)\n", f"**Що сталось:** {it['what_happened']}\n",
            f"**Головний драйвер:** `{pdrv['segment']}` ({' × '.join(pdrv['dimensions'])}) — {pdrv['evidence']}\n",
            "**Гіпотези причини:**"]
    for h in it["cause_hypotheses"]:
        out.append(f"- **{h['cause']}** (впевненість: {h['confidence']})  \n  Підстави: {h['supporting_evidence']}  \n"
                   f"  Як перевірити: {h['how_to_validate']}")
    if it["ruled_out"]:
        out.append("\n**Що виключено:**")
        out += [f"- {x}" for x in it["ruled_out"]]
    if it["caveats"]:
        out.append("\n**Застереження:**")
        out += [f"- {x}" for x in it["caveats"]]
    out.append(_badge(r["verification"]))
    return "\n".join(out)


# ------------------------------------------------------------------ summary

URG = {"today": "сьогодні", "this_week": "цього тижня", "this_month": "цього місяця"}


def render_summary(r: dict[str, Any]) -> str:
    m = r["metrics"]
    out = ["## Executive summary (функція 5)\n"]
    if "kpis" in m:
        out += [f"Період: **{m['current_period']}** vs **{m['previous_period']}**\n",
                "| KPI | попередній | поточний | зміна |", "|---|---|---|---|"]
        for k, v in m["kpis"].items():
            fmt = (lambda x: _pct(x)) if k.endswith("_rate") else (lambda x: f"{x:,.2f}".rstrip("0").rstrip(".").replace(",", " "))
            ch = f"{v['change_pp']:+.2f} п.п." if "change_pp" in v else _pct(v["relative_change"], 1, True)
            out.append(f"| {k} | {fmt(v['previous'])} | {fmt(v['current'])} | {ch} |")
    s = r["summary"]
    if not s:
        out.append("\n" + OFFLINE_NOTE)
        return "\n".join(out)
    out += [f"\n> ### {s['headline']}\n", "**Що сталось:**"]
    out += [f"- {x}" for x in s["what_happened"]]
    out += [f"\n**Чому це важливо:** {s['why_it_matters']}\n", "**Що робити далі:**"]
    out += [f"1. {x['action']} — *{x['owner']}, {URG.get(x['urgency'], x['urgency'])}*" for x in s["next_steps"]]
    if s["watch_outs"]:
        out.append("\n**Застереження:**")
        out += [f"- {x}" for x in s["watch_outs"]]
    out.append(_badge(r["verification"]))
    return "\n".join(out)
