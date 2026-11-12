"""CLI AI-асистента аналітика.

Приклади:
  python cli.py eda data/sessions.csv
  python cli.py hypotheses data/sessions.csv
  python cli.py ask data/sessions.csv "Топ-5 країн за виручкою у жовтні"
  python cli.py ask data/sessions.csv                 # інтерактивний режим
  python cli.py anomaly data/sessions.csv --metric converted --period 2026-10-19:2026-10-25
  python cli.py summary data/sessions.csv --period 2026-10-19:2026-10-25 --with-drivers converted
  python cli.py summary --metrics-json my_metrics.json
  python cli.py report data/sessions.csv              # повний «понеділковий» звіт

Глобальні опції: --offline (без LLM), --out DIR (куди зберегти звіт і графіки), --model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyst import render
from analyst.anomaly import explain_anomaly, find_biggest_change
from analyst.data import DataError, infer_schema, load_csv
from analyst.eda import auto_eda
from analyst.hypotheses import suggest_hypotheses
from analyst.llm import DEFAULT_MODEL, LLMClient, LLMRefusal, LLMUnavailable
from analyst.nl2pandas import ask
from analyst.summary import compute_kpis, drivers_from_anomaly, executive_summary


def _period(s: str | None) -> tuple[str, str] | None:
    if not s:
        return None
    a, sep, b = s.partition(":")
    if not sep:
        raise DataError(f"Період має бути у форматі YYYY-MM-DD:YYYY-MM-DD, отримано '{s}'")
    return a, b


def _last_full_week(df) -> tuple[str, str]:
    import pandas as pd

    day = df[infer_schema(df).date_col].dt.normalize()
    end = day.max()
    # остання неділя, до якої включно є дані
    end = end - pd.Timedelta(days=(end.day_of_week + 1) % 7)
    return (end - pd.Timedelta(days=6)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _save(md: str, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(md, encoding="utf-8")
    return p


def cmd_eda(a, llm):
    df = load_csv(a.file)
    r = auto_eda(df, llm, out_dir=a.out)
    return render.render_eda(r, a.out), "eda.md"


def cmd_hypotheses(a, llm):
    df = load_csv(a.file)
    r = suggest_hypotheses(df, llm, n_min=a.min, n_max=a.max)
    return render.render_hypotheses(r), "hypotheses.md"


def cmd_ask(a, llm):
    df = load_csv(a.file)
    if a.question:
        return render.render_ask(ask(df, " ".join(a.question), llm)), "ask.md"
    print("Інтерактивний режим. Порожній рядок або 'exit' — вихід.\n")
    blocks = []
    while True:
        try:
            q = input("❓ > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "вихід"}:
            break
        md = render.render_ask(ask(df, q, llm))
        print(md + "\n")
        blocks.append(md)
    return "\n\n".join(blocks), "ask.md"


def cmd_anomaly(a, llm):
    df = load_csv(a.file)
    period = _period(a.period) or find_biggest_change(df, a.metric)
    r = explain_anomaly(df, a.metric, period, _period(a.baseline), llm=llm,
                        dims=a.dims.split(",") if a.dims else None, out_dir=a.out)
    return render.render_anomaly(r, a.out), "anomaly.md"


def cmd_summary(a, llm):
    drivers = None
    if a.metrics_json:
        metrics = json.loads(Path(a.metrics_json).read_text(encoding="utf-8"))
    else:
        if not a.file:
            raise DataError("Вкажіть CSV-файл або --metrics-json.")
        df = load_csv(a.file)
        period = _period(a.period) or _last_full_week(df)
        metrics = compute_kpis(df, period, _period(a.baseline))
        if a.with_drivers:
            drivers = drivers_from_anomaly(explain_anomaly(df, a.with_drivers, period, _period(a.baseline), llm=llm))
    r = executive_summary(metrics, llm, drivers=drivers, audience=a.audience)
    return render.render_summary(r), "summary.md"


def cmd_report(a, llm):
    """Повний звіт: EDA -> гіпотези -> аномалія за останній тиждень -> executive summary."""
    df = load_csv(a.file)
    period = _period(a.period) or _last_full_week(df)
    print(f"[1/4] EDA ...", file=sys.stderr)
    eda = auto_eda(df, llm, out_dir=a.out)
    print(f"[2/4] Гіпотези ...", file=sys.stderr)
    hyp = suggest_hypotheses(df, llm)
    print(f"[3/4] Пояснення зміни {a.metric} за {period[0]}..{period[1]} ...", file=sys.stderr)
    an = explain_anomaly(df, a.metric, period, llm=llm, out_dir=a.out)
    print(f"[4/4] Executive summary ...", file=sys.stderr)
    summ = executive_summary(compute_kpis(df, period), llm, drivers=drivers_from_anomaly(an))
    md = "\n\n---\n\n".join([
        f"# Аналітичний звіт: `{Path(a.file).name}`\n\nЗгенеровано AI-асистентом аналітика "
        f"(модель: {llm.model if llm.available else 'офлайн'}).",
        render.render_summary(summ), render.render_anomaly(an, a.out),
        render.render_hypotheses(hyp), render.render_eda(eda, a.out),
    ])
    return md, "report.md"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="AI-асистент аналітика (pandas + Claude)")
    p.add_argument("--offline", action="store_true", help="не викликати LLM, лише розрахунки pandas")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=Path("reports/latest"), help="папка для звітів і графіків")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("eda", help="автоматичний EDA")
    s.add_argument("file")
    s.set_defaults(fn=cmd_eda)

    s = sub.add_parser("hypotheses", help="генерація гіпотез")
    s.add_argument("file")
    s.add_argument("--min", type=int, default=5)
    s.add_argument("--max", type=int, default=10)
    s.set_defaults(fn=cmd_hypotheses)

    s = sub.add_parser("ask", help="питання природною мовою -> pandas")
    s.add_argument("file")
    s.add_argument("question", nargs="*")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("anomaly", help="пояснення зміни метрики")
    s.add_argument("file")
    s.add_argument("--metric", default="converted", help="count | sum:col | col | ratio:a/b")
    s.add_argument("--period", help="YYYY-MM-DD:YYYY-MM-DD (за замовчуванням — тиждень з найбільшою зміною)")
    s.add_argument("--baseline", help="YYYY-MM-DD:YYYY-MM-DD (за замовчуванням — попередній період такої ж довжини)")
    s.add_argument("--dims", help="розрізи через кому (за замовчуванням — усі категоріальні)")
    s.set_defaults(fn=cmd_anomaly)

    s = sub.add_parser("summary", help="executive summary")
    s.add_argument("file", nargs="?")
    s.add_argument("--period")
    s.add_argument("--baseline")
    s.add_argument("--metrics-json", help="готовий JSON з метриками замість CSV")
    s.add_argument("--with-drivers", metavar="METRIC", help="додати аналіз драйверів зміни цієї метрики")
    s.add_argument("--audience", default="CEO and product leadership")
    s.set_defaults(fn=cmd_summary)

    s = sub.add_parser("report", help="повний звіт: summary + аномалія + гіпотези + EDA")
    s.add_argument("file")
    s.add_argument("--metric", default="converted")
    s.add_argument("--period")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    llm = LLMClient(model=a.model, offline=a.offline)
    if not llm.available and not a.offline:
        print("⚠️  Ключ LLM (ANTHROPIC_API_KEY / OPENROUTER_API_KEY) не задано — працюю в офлайн-режимі (лише pandas).", file=sys.stderr)
    try:
        md, name = a.fn(a, llm)
    except (DataError, LLMUnavailable, LLMRefusal) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — помилки API (баланс, ліміти, мережа) показуємо коротко
        if type(e).__module__.split(".")[0] in {"openai", "anthropic"}:
            print(f"❌ Помилка API LLM: {e}\n   Спробуйте --offline або перевірте ключ/баланс.", file=sys.stderr)
            return 3
        raise
    print(md)
    if md:
        path = _save(md, a.out, name)
        print(f"\n💾 Збережено: {path}", file=sys.stderr)
    if llm.available:
        print(f"🔢 Токени: input {llm.total_input_tokens:,}, output {llm.total_output_tokens:,}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
