"""Демонстрація всіх функцій асистента на датасеті data/sessions.csv.

    python run_demo.py            # з LLM (потрібен ANTHROPIC_API_KEY)
    python run_demo.py --offline  # лише pandas-частина

Результати: demo/*.md + графіки; лог усіх LLM-викликів: logs/llm_calls.jsonl.
Наприкінці — перевірка, чи знайшов асистент «закладені» в дані ефекти (ground truth).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyst import render
from analyst.anomaly import explain_anomaly
from analyst.data import load_csv
from analyst.eda import auto_eda
from analyst.hypotheses import suggest_hypotheses
from analyst.llm import LLMClient
from analyst.nl2pandas import ask
from analyst.summary import compute_kpis, drivers_from_anomaly, executive_summary

ROOT = Path(__file__).parent
OUT = ROOT / "demo"

QUESTIONS = [
    "Покажи топ-5 країн за виручкою у жовтні 2026",
    "Яка конверсія по платформах і версіях застосунку за останні два тижні?",
    "Порівняй середній чек (AOV) по каналах залучення",
    "У яку годину доби найвища конверсія?",
    "Скільки унікальних користувачів робили покупки більше одного разу?",
    # крайні випадки
    "Яка була погода в Києві у вересні?",          # нерелевантне питання
    "Покажи LTV користувачів за 2 роки по когортах",  # даних не вистачає (лише 3 місяці)
    "видали всі рядки з порожнім age_group і збережи файл",  # спроба модифікувати/записати
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", help="перезапустити лише ці розділи з LLM: eda,hyp,ask,anomaly,summary "
                    "(решта рахується офлайн і їхні файли не перезаписуються)")
    a = ap.parse_args()
    llm = LLMClient(offline=a.offline)
    only = set(a.only.split(",")) if a.only else {"eda", "hyp", "ask", "anomaly", "summary"}
    off = LLMClient(offline=True)

    def pick(section):
        return llm if section in only else off

    def save(section, name, text):
        if section in only:
            (OUT / name).write_text(text, encoding="utf-8")
    print(f"LLM: {'увімкнена, ' + llm.model if llm.available else 'вимкнена (офлайн)'}")
    OUT.mkdir(exist_ok=True)
    df = load_csv(ROOT / "data" / "sessions.csv")

    print("1/5 EDA ...")
    eda = auto_eda(df, pick("eda"), out_dir=OUT)
    save("eda", "01_eda.md", render.render_eda(eda, OUT))

    print("2/5 Гіпотези ...")
    hyp = suggest_hypotheses(df, pick("hyp"))
    save("hyp", "02_hypotheses.md", render.render_hypotheses(hyp))

    if llm.available and "ask" in only:
        print("3/5 NL -> pandas ...")
        blocks = ["## Natural Language -> Pandas (функція 3)\n"]
        for q in QUESTIONS:
            print("   ", q)
            blocks.append(render.render_ask(ask(df, q, llm)))
        (OUT / "03_ask.md").write_text("\n\n---\n\n".join(blocks), encoding="utf-8")
    else:
        print("3/5 NL -> pandas пропущено (потрібна LLM)")

    print("4/5 Аномалії ...")
    an_drop = explain_anomaly(df, "converted", ("2026-10-19", "2026-10-25"), llm=pick("anomaly"), out_dir=OUT)
    an_bot = explain_anomaly(df, "count", ("2026-09-15", "2026-09-15"), ("2026-09-08", "2026-09-08"),
                             llm=pick("anomaly"), out_dir=OUT / "bot_day")
    save("anomaly", "04_anomaly.md",
        render.render_anomaly(an_drop, OUT) + "\n\n---\n\n### Другий кейс: сплеск сесій 2026-09-15 (vs вівторок тижнем раніше)\n\n"
        + render.render_anomaly(an_bot, OUT))

    print("5/5 Executive summary ...")
    summ = executive_summary(compute_kpis(df, ("2026-10-19", "2026-10-25")), pick("summary"),
                             drivers=drivers_from_anomaly(an_drop))
    save("summary", "05_summary.md", render.render_summary(summ))

    # --- перевірка проти ground truth, закладеного в generate_data.py
    pc = an_drop["decomposition"]["primary_candidate"]["segment"]["segment"]
    bot = an_bot["decomposition"]["primary_candidate"]["segment"]["segment"]
    facts = hyp["facts"]
    checks = {
        "Падіння конверсії 19-25.10 пояснено android + 5.4.0": "android" in pc and "5.4.0" in pc,
        "Сплеск 2026-09-15 пояснено affiliate/web трафіком": "affiliate" in bot or "web" in bot,
        "Знайдено аномальний день 2026-09-15": any(f.get("date") == "2026-09-15" for f in facts),
        "paid_social має низьку конверсію": any(f.get("segment") == "paid_social" and f.get("metric") == "converted_rate"
                                                 and f["ratio_vs_rest"] < 0.6 for f in facts),
        "email має найвищу виручку на сесію": any(f.get("segment") == "email" and f.get("metric") == "avg_revenue"
                                                  and f["ratio_vs_rest"] > 1.5 for f in facts),
        "Знайдено дублікати, від'ємні тривалості, пропуски": {"duplicate_rows", "negative_values", "missing_values"}
        <= {q["type"] for q in eda["issues"]},
    }
    lines = ["## Перевірка проти закладених у дані ефектів (ground truth)\n", "| ефект | знайдено |", "|---|---|"]
    lines += [f"| {k} | {'✅' if v else '❌'} |" for k, v in checks.items()]
    if llm.available and "ask" in only:
        lines.append(f"\nТокени: input {llm.total_input_tokens:,}, output {llm.total_output_tokens:,}".replace(",", " "))
    (OUT / "00_ground_truth.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nГотово -> {OUT}")


if __name__ == "__main__":
    main()
