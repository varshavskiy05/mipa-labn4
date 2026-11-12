"""Функція 1. Автоматичний EDA: pandas рахує, LLM інтерпретує."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import prompts
from .data import Schema, infer_schema, to_py
from .llm import LLMClient
from .verify import verify_numbers


# ---------------------------------------------------------------- профіль

def profile_dataframe(df: pd.DataFrame, schema: Schema | None = None) -> dict[str, Any]:
    schema = schema or infer_schema(df)
    n = len(df)
    prof: dict[str, Any] = {
        "n_rows": n,
        "n_columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 2**20, 1),
        "duplicate_rows": int(df.duplicated().sum()),
        "roles": {
            "date": schema.date_col,
            "dimensions": schema.dimensions,
            "binary_metrics": schema.binary_metrics,
            "value_metrics": schema.value_metrics,
            "ids": schema.id_cols,
        },
        "columns": {},
    }
    for c in df.columns:
        s = df[c]
        col: dict[str, Any] = {
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "null_share": round(float(s.isna().mean()), 4),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_datetime64_any_dtype(s):
            col.update(min=to_py(s.min()), max=to_py(s.max()), n_days=int(s.dt.normalize().nunique()))
        elif pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            q = d.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            iqr = q[0.75] - q[0.25]
            if c in schema.binary_metrics:
                col.update(mean=round(float(d.mean()), 4), sum=int(d.sum()))
            else:
                nonzero = d[d != 0]
                col.update(
                    min=to_py(d.min()), p01=to_py(q[0.01]), p25=to_py(q[0.25]), median=to_py(q[0.5]),
                    mean=round(float(d.mean()), 2), p75=to_py(q[0.75]), p99=to_py(q[0.99]),
                    max=to_py(d.max()), std=round(float(d.std()), 2),
                    skew=round(float(d.skew()), 2),
                    n_negative=int((d < 0).sum()), n_zero=int((d == 0).sum()),
                    zero_share=round(float((d == 0).mean()), 4),
                )
                if iqr > 0:
                    hi = q[0.75] + 3 * iqr
                    col["n_extreme_outliers_iqr3"] = int((d > hi).sum())
                # для «zero-inflated» колонок (виручка) — статистики по ненульових
                if len(nonzero) and (d == 0).mean() > 0.5:
                    nz = nonzero[nonzero > 0]
                    qn = nz.quantile([0.25, 0.5, 0.75, 0.99])
                    iqr_n = qn[0.75] - qn[0.25]
                    col["nonzero_stats"] = {
                        "count": int(len(nz)), "median": to_py(round(qn[0.5], 2)),
                        "p99": to_py(round(qn[0.99], 2)), "max": to_py(nz.max()),
                        "n_extreme_outliers_iqr3": int((nz > qn[0.75] + 3 * iqr_n).sum()),
                    }
        else:
            vc = s.value_counts(normalize=True, dropna=True).head(8)
            col["top_values"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        prof["columns"][c] = col

    if schema.date_col:
        daily = df.groupby(df[schema.date_col].dt.normalize()).size()
        med = daily.median()
        mad = (daily - med).abs().median() or 1
        z = (daily - med) / (1.4826 * mad)
        prof["daily_rows"] = {
            "median": to_py(med), "min": to_py(daily.min()), "max": to_py(daily.max()),
            "spike_days": {d.strftime("%Y-%m-%d"): int(daily[d]) for d in daily.index[z.abs() > 4]},
        }
    return prof


def detect_quality_issues(df: pd.DataFrame, prof: dict[str, Any], schema: Schema) -> list[dict[str, Any]]:
    """Rule-based перевірки якості даних. Кожна проблема має id і числа."""
    issues: list[dict[str, Any]] = []

    def add(kind: str, column: str | None, detail: str, **numbers):
        issues.append({"id": f"Q{len(issues) + 1}", "type": kind, "column": column,
                       "detail": detail, **numbers})

    if prof["duplicate_rows"]:
        add("duplicate_rows", None, "fully duplicated rows",
            count=prof["duplicate_rows"], share=round(prof["duplicate_rows"] / prof["n_rows"], 4))
    for c in schema.id_cols:
        dup = int(df[c].duplicated().sum())
        if dup and df[c].nunique() > 0.9 * len(df):  # колонка схожа на первинний ключ
            add("duplicate_ids", c, "values of a key-like column are repeated", count=dup)
    for c, col in prof["columns"].items():
        if col["nulls"]:
            add("missing_values", c, "missing values", count=col["nulls"], share=col["null_share"])
        if col["unique"] <= 1:
            add("constant_column", c, "column has a single value", unique=col["unique"])
        if col.get("n_negative"):
            add("negative_values", c, "negative values in a numeric column", count=col["n_negative"],
                min=col["min"])
        if col.get("n_extreme_outliers_iqr3"):
            add("outliers", c, "values above Q3 + 3*IQR", count=col["n_extreme_outliers_iqr3"],
                p99=col["p99"], max=col["max"])
        nz = col.get("nonzero_stats")
        if nz and nz.get("n_extreme_outliers_iqr3"):
            add("outliers_nonzero", c, "among non-zero values: above Q3 + 3*IQR",
                count=nz["n_extreme_outliers_iqr3"], median_nonzero=nz["median"], max=nz["max"])
    for d, cnt in prof.get("daily_rows", {}).get("spike_days", {}).items():
        add("volume_spike", schema.date_col, f"abnormal number of rows on {d}",
            rows=cnt, typical_daily_rows=prof["daily_rows"]["median"])
    return issues


# ---------------------------------------------------------------- графіки

def plot_eda(df: pd.DataFrame, schema: Schema, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if schema.value_metrics:
        cols = schema.value_metrics[:4]
        fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.2))
        for ax, c in zip(np.atleast_1d(axes), cols):
            d = df[c].dropna()
            d = d[(d > d.quantile(0.005)) & (d < d.quantile(0.995))]
            if (d == 0).mean() > 0.5:
                d = d[d != 0]
                ax.set_title(f"{c} (≠0, без 0.5% хвостів)", fontsize=9)
            else:
                ax.set_title(f"{c} (без 0.5% хвостів)", fontsize=9)
            ax.hist(d, bins=40, color="#4C72B0")
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths.append(out_dir / "eda_distributions.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)
    dims = [c for c in schema.dimensions if df[c].nunique() <= 12][:4]
    if dims:
        fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.2))
        for ax, c in zip(np.atleast_1d(axes), dims):
            vc = df[c].astype(str).value_counts(normalize=True).head(10)[::-1]
            ax.barh(vc.index, vc.values * 100, color="#4C72B0")
            ax.set_title(f"{c}, % рядків", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths.append(out_dir / "eda_categories.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)
    if schema.date_col:
        daily = df.groupby(df[schema.date_col].dt.normalize()).size()
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(daily.index, daily.values, color="#4C72B0")
        ax.set_title("Кількість рядків по днях", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths.append(out_dir / "eda_daily.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------- головна функція

def auto_eda(df: pd.DataFrame, llm: LLMClient, out_dir: Path | None = None) -> dict[str, Any]:
    """Повертає {'profile', 'issues', 'interpretation', 'verification', 'charts'}."""
    schema = infer_schema(df)
    prof = profile_dataframe(df, schema)
    issues = detect_quality_issues(df, prof, schema)
    charts = plot_eda(df, schema, out_dir) if out_dir else []

    result: dict[str, Any] = {"profile": prof, "issues": issues, "charts": [str(p) for p in charts],
                              "interpretation": None, "verification": None}
    if llm.available:
        interp = llm.call_json(
            task="auto_eda",
            system=prompts.EDA_SYSTEM,
            user=prompts.EDA_USER.format(
                profile=json.dumps(prof, ensure_ascii=False, indent=1, default=str),
                issues=json.dumps(issues, ensure_ascii=False, indent=1, default=str),
            ),
            schema=prompts.EDA_SCHEMA,
            effort="low",
            nonempty=("column_insights", "quality_findings", "next_steps") if issues else ("column_insights", "next_steps"),
        )
        # перевірка: усі числа в інтерпретації мають бути у профілі/issues
        known_ids = {i["id"] for i in issues}
        for f in interp["quality_findings"]:
            f["unknown_issue_ids"] = [i for i in f["issue_ids"] if i not in known_ids]
        text = json.dumps(interp, ensure_ascii=False)
        result["interpretation"] = interp
        result["verification"] = verify_numbers(text, {"profile": prof, "issues": issues}, derived=True)
    return result
