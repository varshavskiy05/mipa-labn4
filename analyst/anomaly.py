"""Функція 4. Пояснення аномалій: декомпозиція зміни метрики по розрізах.

Метрика задається рядком:
    count                       -> кількість рядків
    sum:revenue                 -> сума колонки
    mean:converted | converted  -> середнє (для 0/1 — частка, конверсія)
    ratio:revenue/converted     -> sum(revenue) / sum(converted)  (напр. AOV)

Для метрик-відношень R = sum(a)/sum(b) зміна розкладається по сегментах i:
    R = Σ w_i r_i,   w_i = b_i / B  (вага сегмента),  r_i = a_i / b_i
    ΔR = Σ c_i,      c_i = w1_i (r1_i − R0) − w0_i (r0_i − R0)
    c_i = mix_i + rate_i,  mix_i = (w1_i − w0_i)(r0_i − R0),  rate_i = w1_i (r1_i − r0_i)
Внески в межах одного розрізу точно сумуються в ΔR; сегменти, яких не було в
базовому періоді (нова версія застосунку), коректно отримують лише mix-ефект.
Для адитивних метрик (count, sum) внесок сегмента = його абсолютна зміна.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import prompts
from .data import DataError, infer_schema
from .llm import LLMClient
from .verify import verify_numbers


@dataclass
class Metric:
    spec: str
    num: str | None      # колонка чисельника (None -> рахуємо рядки)
    den: str | None      # колонка знаменника (None -> адитивна метрика); "__rows__" -> к-сть рядків
    label: str

    @property
    def additive(self) -> bool:
        return self.den is None


def parse_metric(spec: str, df: pd.DataFrame) -> Metric:
    spec = spec.strip()
    if spec in {"count", "rows", "sessions"}:
        return Metric(spec, None, None, "кількість рядків")
    kind, _, rest = spec.partition(":")
    if not rest:  # просто назва колонки -> середнє
        kind, rest = "mean", spec
    cols = rest.split("/") if kind == "ratio" else [rest]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"Немає колонок {missing}. Доступні: {list(df.columns)}")
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise DataError(f"Колонка '{c}' не числова — метрику '{spec}' порахувати неможливо.")
    if kind == "sum":
        return Metric(spec, rest, None, f"сума {rest}")
    if kind == "mean":
        return Metric(spec, rest, "__rows__", f"середнє {rest}")
    if kind == "ratio" and len(cols) == 2:
        return Metric(spec, cols[0], cols[1], f"{cols[0]} / {cols[1]}")
    raise DataError(f"Невідомий формат метрики '{spec}'. Приклади: count, sum:revenue, converted, ratio:revenue/converted")


def _agg(df: pd.DataFrame, m: Metric, by: list[str] | None = None) -> pd.DataFrame:
    tmp = pd.DataFrame(index=df.index)
    tmp["a"] = 1.0 if m.num is None else df[m.num].astype(float)
    tmp["b"] = 1.0 if m.den in (None, "__rows__") else df[m.den].astype(float)
    if by:
        for c in by:
            tmp[c] = df[c].astype(str).where(df[c].notna(), "(missing)")
        return tmp.groupby(by, observed=True)[["a", "b"]].sum()
    return tmp[["a", "b"]].sum().to_frame().T


def metric_value(df: pd.DataFrame, m: Metric) -> float:
    s = _agg(df, m).iloc[0]
    if m.additive:
        return float(s["a"])
    return float(s["a"] / s["b"]) if s["b"] else float("nan")


def _decompose(base: pd.DataFrame, cur: pd.DataFrame, m: Metric, dims: list[str]) -> pd.DataFrame:
    g0, g1 = _agg(base, m, dims), _agg(cur, m, dims)
    t = g0.add_suffix("0").join(g1.add_suffix("1"), how="outer").fillna(0.0)
    if m.additive:
        t["base_value"], t["cur_value"] = t["a0"], t["a1"]
        t["contribution"] = t["a1"] - t["a0"]
        t["mix_effect"] = np.nan
        t["rate_effect"] = np.nan
    else:
        B0, B1 = t["b0"].sum(), t["b1"].sum()
        R0 = t["a0"].sum() / B0
        w0, w1 = t["b0"] / B0, t["b1"] / B1
        r0 = np.where(t["b0"] > 0, t["a0"] / t["b0"].replace(0, np.nan), np.nan)
        r1 = np.where(t["b1"] > 0, t["a1"] / t["b1"].replace(0, np.nan), np.nan)
        t["base_share"], t["cur_share"] = w0, w1
        t["base_value"], t["cur_value"] = r0, r1
        c = w1 * (np.nan_to_num(r1) - R0) - w0 * (np.nan_to_num(r0) - R0)
        t["contribution"] = c
        new_seg = t["b0"] == 0
        t["mix_effect"] = np.where(new_seg, c, (w1 - w0) * (np.nan_to_num(r0) - R0))
        t["rate_effect"] = np.where(new_seg, 0.0, w1 * (np.nan_to_num(r1) - np.nan_to_num(r0)))
        t["new_segment"] = new_seg
    t["base_n"], t["cur_n"] = t["b0"], t["b1"]
    return t


def _without(df: pd.DataFrame, dims: list[str], seg: tuple) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for d, v in zip(dims, seg):
        mask &= df[d].astype(str).where(df[d].notna(), "(missing)").to_numpy() == v
    return df[~mask]


def explain_anomaly(
    df: pd.DataFrame,
    metric: str,
    current: tuple[str, str],
    baseline: tuple[str, str] | None = None,
    llm: LLMClient | None = None,
    dims: list[str] | None = None,
    date_col: str | None = None,
    top_k: int = 5,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    schema = infer_schema(df)
    date_col = date_col or schema.date_col
    if not date_col:
        raise DataError("У даних немає колонки з датою — неможливо порівняти періоди.")
    m = parse_metric(metric, df)
    dims = dims or [d for d in schema.dimensions if 2 <= df[d].nunique() <= 30]
    unknown = [d for d in dims if d not in df.columns]
    if unknown:
        raise DataError(f"Невідомі розрізи: {unknown}")

    c0, c1 = pd.Timestamp(current[0]), pd.Timestamp(current[1])
    if c1 < c0:
        raise DataError("Кінець періоду раніше за початок.")
    if baseline is None:  # попередній період такої ж довжини
        length = c1 - c0 + pd.Timedelta(days=1)
        b0, b1 = c0 - length, c0 - pd.Timedelta(days=1)
    else:
        b0, b1 = pd.Timestamp(baseline[0]), pd.Timestamp(baseline[1])
    day = df[date_col].dt.normalize()
    cur, base = df[(day >= c0) & (day <= c1)], df[(day >= b0) & (day <= b1)]
    if cur.empty or base.empty:
        raise DataError(
            f"Немає даних за {'поточний' if cur.empty else 'базовий'} період. "
            f"Дані доступні з {day.min():%Y-%m-%d} по {day.max():%Y-%m-%d}."
        )

    v0, v1 = metric_value(base, m), metric_value(cur, m)
    delta = v1 - v0
    summary = {
        "metric": m.spec, "metric_label": m.label,
        "baseline_period": f"{b0:%Y-%m-%d}..{b1:%Y-%m-%d}", "current_period": f"{c0:%Y-%m-%d}..{c1:%Y-%m-%d}",
        "baseline_value": round(v0, 6), "current_value": round(v1, 6),
        "absolute_change": round(delta, 6),
        "relative_change": round(delta / v0, 4) if v0 else None,
        "baseline_rows": len(base), "current_rows": len(cur),
    }

    def seg_records(t: pd.DataFrame, dlist: list[str]) -> list[dict[str, Any]]:
        t = t.reindex(t["contribution"].abs().sort_values(ascending=False).index).head(top_k)
        recs = []
        for idx, row in t.iterrows():
            seg = idx if isinstance(idx, tuple) else (idx,)
            rec = {"segment": " | ".join(f"{d}={v}" for d, v in zip(dlist, seg)),
                   "contribution": round(float(row["contribution"]), 6),
                   "share_of_change": round(float(row["contribution"] / delta), 3) if delta else None,
                   "base_value": None if pd.isna(row["base_value"]) else round(float(row["base_value"]), 6),
                   "cur_value": None if pd.isna(row["cur_value"]) else round(float(row["cur_value"]), 6),
                   "base_n": int(row["base_n"]), "cur_n": int(row["cur_n"])}
            if not m.additive:
                rec.update(base_share=round(float(row["base_share"]), 4), cur_share=round(float(row["cur_share"]), 4),
                           mix_effect=round(float(row["mix_effect"]), 6), rate_effect=round(float(row["rate_effect"]), 6),
                           new_segment=bool(row["new_segment"]))
            recs.append(rec)
        return recs

    def concentration(t: pd.DataFrame) -> float:
        """Яку частку зміни пояснює найбільший за модулем сегмент того ж знаку, що й зміна."""
        if not delta:
            return 0.0
        same = t["contribution"] * np.sign(delta)
        return float(same.max() / abs(delta))

    single = []
    for d in dims:
        t = _decompose(base, cur, m, [d])
        single.append({"dimensions": [d], "concentration": round(concentration(t), 3),
                       "top_segments": seg_records(t, [d]), "_t": t})
    single.sort(key=lambda x: -x["concentration"])

    pairs = []
    top_dims = [x["dimensions"][0] for x in single[:4]]
    for a, b in itertools.combinations(top_dims, 2):
        t = _decompose(base, cur, m, [a, b])
        pairs.append({"dimensions": [a, b], "concentration": round(concentration(t), 3),
                      "top_segments": seg_records(t, [a, b]), "_t": t})
    pairs.sort(key=lambda x: -x["concentration"])

    # Найконкретніший сильний драйвер: пара розрізів, якщо вона пояснює щонайменше 75%
    # від того, що пояснює найкращий одиночний розріз (конкретніший сегмент = краща гіпотеза).
    best = single[0]
    if pairs and pairs[0]["concentration"] >= 0.75 * single[0]["concentration"]:
        best = pairs[0]

    def counterfactual(cand: dict[str, Any]) -> dict[str, Any]:
        seg = cand["top_segments"][0]["segment"]
        vals = tuple(part.split("=", 1)[1] for part in seg.split(" | "))
        w0 = metric_value(_without(base, cand["dimensions"], vals), m)
        w1 = metric_value(_without(cur, cand["dimensions"], vals), m)
        return {"excluded_segment": seg,
                "baseline_value_without_segment": round(w0, 6), "current_value_without_segment": round(w1, 6),
                "relative_change_without_segment": round((w1 - w0) / w0, 4) if w0 else None}

    top_seg = best["top_segments"][0]
    counterfactuals = [counterfactual(single[0])] + ([counterfactual(pairs[0])] if pairs else [])

    def strip(xs):
        return [{k: v for k, v in x.items() if k != "_t"} for x in xs]

    decomposition = {
        "summary": summary,
        "dimensions_ranked": strip(single),
        "dimension_pairs_ranked": strip(pairs[:3]),
        "primary_candidate": {"dimensions": best["dimensions"], "segment": top_seg},
        "metric_without_segment": counterfactuals,
    }
    result: dict[str, Any] = {"decomposition": decomposition, "interpretation": None, "verification": None,
                              "charts": []}
    if out_dir:
        result["charts"] = _plot(df, m, date_col, (b0, b1), (c0, c1), single[0], best, out_dir)

    if llm is not None and llm.available:
        interp = llm.call_json(
            task="explain_anomaly",
            system=prompts.ANOMALY_SYSTEM,
            user=prompts.ANOMALY_USER.format(
                metric=json.dumps({"spec": m.spec, "label": m.label}, ensure_ascii=False),
                decomposition=json.dumps(decomposition, ensure_ascii=False, indent=1, default=str),
            ),
            schema=prompts.ANOMALY_SCHEMA,
            effort="medium",
            nonempty=("cause_hypotheses",),
        )
        result["interpretation"] = interp
        # перевіряємо числа тексту проти фактів, переданих моделі: summary, головний кандидат,
        # контрфакти, концентрації всіх розрізів і топ-сегменти (модель цитує й «виключені» розрізи)
        key_facts = [summary, decomposition["primary_candidate"], counterfactuals,
                     [x["concentration"] for x in single + pairs],
                     [x["top_segments"] for x in single], [x["top_segments"][:3] for x in pairs[:3]]]
        result["verification"] = verify_numbers(json.dumps(interp, ensure_ascii=False), key_facts, derived=True)
    return result


def _plot(df, m, date_col, base_p, cur_p, best_single, best, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    day = df[date_col].dt.normalize()
    lo = base_p[0] - pd.Timedelta(days=28)
    sub = df[(day >= lo) & (day <= cur_p[1])]
    g = _agg(sub.assign(_d=day), m, ["_d"])
    series = g["a"] if m.additive else g["a"] / g["b"]
    series.index = pd.to_datetime(series.index)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.values, color="#4C72B0")
    ax.axvspan(base_p[0], base_p[1] + pd.Timedelta(hours=23), color="grey", alpha=0.15, label="базовий період")
    ax.axvspan(cur_p[0], cur_p[1] + pd.Timedelta(hours=23), color="#DD8452", alpha=0.25, label="поточний період")
    ax.set_title(f"{m.label} по днях", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / "anomaly_timeseries.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    t = best["_t"]
    t = t.reindex(t["contribution"].abs().sort_values(ascending=False).index).head(8)[::-1]
    labels = [" | ".join(i) if isinstance(i, tuple) else str(i) for i in t.index]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in t["contribution"]]
    vals = t["contribution"] * (100 if not m.additive else 1)
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="black", lw=0.6)
    unit = " (п.п.)" if not m.additive else ""
    ax.set_title(f"Внесок сегментів у зміну{unit}: {' × '.join(best['dimensions'])}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / "anomaly_contributions.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))
    return paths


def find_biggest_change(df: pd.DataFrame, metric: str, freq: str = "W-SUN") -> tuple[str, str] | None:
    """Знаходить тиждень із найбільшою відносною зміною метрики до попереднього (для авто-звіту).
    Неповні тижні (менше 7 днів даних) ігноруються."""
    schema = infer_schema(df)
    m = parse_metric(metric, df)
    day = df[schema.date_col].dt.normalize()
    weeks = day.dt.to_period(freq)
    full = day.groupby(weeks).nunique()
    full = full[full == 7].index
    vals = {w: metric_value(df[weeks == w], m) for w in full}
    s = pd.Series(vals).sort_index()
    if len(s) < 2:
        return None
    rel = (s / s.shift(1) - 1).dropna()
    w = rel.abs().idxmax()
    return (w.start_time.strftime("%Y-%m-%d"), w.end_time.strftime("%Y-%m-%d"))
