"""Завантаження даних і автоматичне визначення ролей колонок (без LLM)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    """Проблема з вхідними даними (порожній файл, немає потрібних колонок тощо)."""


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Файл не знайдено: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise DataError(f"Файл порожній: {path}") from e
    if df.empty:
        raise DataError(f"У файлі {path} немає жодного рядка даних.")
    return parse_dates(df)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Перетворює текстові колонки, схожі на дати, у datetime."""
    df = df.copy()
    for col in df.columns:
        if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])) and _looks_like_date(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _looks_like_date(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if sample.empty or not sample.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return False
    return pd.to_datetime(sample, errors="coerce").notna().mean() > 0.95


@dataclass
class Schema:
    date_col: str | None
    dimensions: list[str]       # категоріальні колонки з малою кардинальністю (розрізи)
    binary_metrics: list[str]   # 0/1 колонки -> метрики-частки (конверсія тощо)
    value_metrics: list[str]    # неперервні числові (виручка, тривалість)
    id_cols: list[str]


def infer_schema(df: pd.DataFrame, max_card: int = 30) -> Schema:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    dims, binary, values, ids = [], [], [], []
    n = len(df)
    for c in df.columns:
        if c in date_cols:
            continue
        s = df[c]
        nunique = s.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(s):
            uniq = set(s.dropna().unique().tolist())
            if uniq and uniq <= {0, 1}:
                binary.append(c)
            elif nunique <= 24 and pd.api.types.is_integer_dtype(s) and c.lower() in {"hour", "month", "weekday"}:
                dims.append(c)
            elif nunique > 10:
                values.append(c)
            else:
                dims.append(c)
        else:
            if nunique > 0.5 * n or c.lower().endswith("_id") or c.lower() == "id":
                ids.append(c)
            elif nunique <= max_card:
                dims.append(c)
            else:
                ids.append(c)
    return Schema(date_cols[0] if date_cols else None, dims, binary, values, ids)


def to_py(x):
    """numpy-скаляри -> звичайні python-типи (для JSON)."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return x


def schema_description(df: pd.DataFrame, n_examples: int = 5) -> str:
    """Компактний опис таблиці для промптів (типи + приклади значень)."""
    lines = [f"DataFrame `df`: {len(df):,} rows x {df.shape[1]} columns"]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            ex = f"range {s.min():%Y-%m-%d} .. {s.max():%Y-%m-%d}"
        elif pd.api.types.is_numeric_dtype(s):
            ex = f"min={to_py(s.min())}, median={to_py(s.median())}, max={to_py(s.max())}"
        else:
            vals = s.dropna().unique()[:n_examples]
            ex = f"{s.nunique()} unique, e.g. {list(map(str, vals))}"
        lines.append(f"- {c} ({s.dtype}, nulls={int(s.isna().sum())}): {ex}")
    return "\n".join(lines)
