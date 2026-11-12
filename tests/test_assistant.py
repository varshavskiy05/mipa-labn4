"""Тести крайніх випадків. Запуск: python -m unittest -v

LLM підміняється FakeLLM, що повертає заготовлені відповіді, — так можна
перевірити логіку навколо LLM (повтори, валідацію, перевірку чисел) без ключа.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst.anomaly import _decompose, explain_anomaly, parse_metric  # noqa: E402
from analyst.data import DataError, load_csv, parse_dates  # noqa: E402
from analyst.hypotheses import build_facts, suggest_hypotheses  # noqa: E402
from analyst.llm import LLMClient  # noqa: E402
from analyst.nl2pandas import UnsafeCodeError, ask, check_code_safety  # noqa: E402
from analyst.summary import compute_kpis, executive_summary  # noqa: E402
from analyst.verify import verify_numbers  # noqa: E402

DATA = ROOT / "data" / "sessions.csv"


class FakeLLM:
    """Повертає відповіді з черги для кожної задачі."""

    available = True
    model = "fake"

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, str]] = []

    def call_json(self, *, task, system, user, schema, effort="medium", max_tokens=None, nonempty=()):
        self.calls.append((task, user))
        return self.responses[task].pop(0)


def setUpModule():
    if not DATA.exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "generate_data.py")], check=True)
    global DF
    DF = load_csv(DATA)


# ------------------------------------------------------------------ verify

class TestVerify(unittest.TestCase):
    facts = {"cr_prev": 0.0463, "cr_cur": 0.0362, "sessions": 7012}

    def test_accepts_rounding_and_derived(self):
        r = verify_numbers("Конверсія впала з 4.6% до 3.62% (-1.01 п.п., -21.8%), сесій 7 012.", self.facts)
        self.assertTrue(r.ok, r.to_dict())
        self.assertEqual(r.total, 5)

    def test_flags_hallucination(self):
        r = verify_numbers("Конверсія впала до 2.9%.", self.facts)
        self.assertFalse(r.ok)
        self.assertEqual(r.unverified[0].raw, "2.9%")

    def test_ignores_dates_versions_small_ints(self):
        r = verify_numbers("Реліз 5.4.0 від 2026-10-19, топ-5 країн, 3 сегменти, за останні 14 днів.", self.facts)
        self.assertEqual(r.total, 0)


class TestDegenerate(unittest.TestCase):
    def test_placeholders_detected(self):
        from analyst.llm import is_degenerate

        # реальна відповідь безкоштовної моделі в demo (усі поля — "...")
        self.assertTrue(is_degenerate({"what_happened": "...", "primary_driver": {"segment": "android",
                                       "evidence": "..."}, "ruled_out": ["..."], "caveats": []}))
        self.assertFalse(is_degenerate({"what_happened": "Конверсія впала", "cause": "Баг релізу",
                                        "evidence": "Конверсія 1.39% на 5.4.0", "caveats": ["n/a"]}))


# ------------------------------------------------------------------ дані

class TestData(unittest.TestCase):
    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.csv"
            p.write_text("", encoding="utf-8")
            with self.assertRaises(DataError):
                load_csv(p)

    def test_header_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.csv"
            p.write_text("a,b,c\n", encoding="utf-8")
            with self.assertRaises(DataError):
                load_csv(p)

    def test_missing_file(self):
        with self.assertRaises(DataError):
            load_csv("no_such_file.csv")


# ------------------------------------------------------------------ аномалії

class TestAnomaly(unittest.TestCase):
    def test_contributions_sum_to_delta(self):
        m = parse_metric("converted", DF)
        day = DF["session_date"]
        base = DF[(day >= "2026-10-12") & (day <= "2026-10-18")]
        cur = DF[(day >= "2026-10-19") & (day <= "2026-10-25")]
        t = _decompose(base, cur, m, ["app_version"])
        delta = cur["converted"].mean() - base["converted"].mean()
        self.assertAlmostEqual(t["contribution"].sum(), delta, places=10)

    def test_finds_planted_android_bug(self):
        r = explain_anomaly(DF, "converted", ("2026-10-19", "2026-10-25"))
        seg = r["decomposition"]["primary_candidate"]["segment"]["segment"]
        self.assertIn("android", seg)
        self.assertIn("5.4.0", seg)

    def test_period_without_data(self):
        with self.assertRaises(DataError):
            explain_anomaly(DF, "converted", ("2030-01-01", "2030-01-07"))

    def test_bad_metrics(self):
        with self.assertRaises(DataError):
            parse_metric("no_such_column", DF)
        with self.assertRaises(DataError):
            parse_metric("sum:country", DF)  # нечислова колонка

    def test_reversed_period(self):
        with self.assertRaises(DataError):
            explain_anomaly(DF, "converted", ("2026-10-25", "2026-10-19"))


# ------------------------------------------------------------------ гіпотези

class TestHypotheses(unittest.TestCase):
    def test_facts_contain_planted_effects(self):
        facts = build_facts(DF)
        self.assertTrue(any(f["type"] == "volume_anomaly" and f["date"] == "2026-09-15" for f in facts))
        self.assertTrue(any(f.get("segment") == "paid_social" and f.get("metric") == "converted_rate" for f in facts))

    def test_validation_of_llm_output(self):
        facts = build_facts(DF)
        f = next(x for x in facts if x.get("segment") == "paid_social" and x.get("metric") == "converted_rate")
        good = {"title": "t", "evidence_fact_ids": [f["id"]],
                "observation": f"Конверсія paid_social {f['segment_value'] * 100:.2f}% проти {f['rest_value'] * 100:.2f}%",
                "hypothesis": "h", "category": "conversion", "business_impact": "b", "how_to_test": "x",
                "priority": "high", "confidence": "high"}
        bad = dict(good, evidence_fact_ids=["F999"], observation="Конверсія впала на 37.3%")
        llm = FakeLLM({"suggest_hypotheses": [{"hypotheses": [good, bad]}]})
        r = suggest_hypotheses(DF, llm)
        h_good, h_bad = r["hypotheses"]
        self.assertTrue(h_good["grounded"])
        self.assertTrue(h_good["verification"].ok)
        self.assertFalse(h_bad["grounded"])
        self.assertFalse(h_bad["verification"].ok)


# ------------------------------------------------------------------ NL -> pandas

class TestNL2Pandas(unittest.TestCase):
    def test_unsafe_code_rejected(self):
        for code in ["import os\nresult = 1", "result = open('x').read()", "result = df.__class__",
                     "df.to_csv('x.csv')\nresult = 1", "result = __import__('os')"]:
            with self.assertRaises(UnsafeCodeError, msg=code):
                check_code_safety(code)

    def test_harmless_imports_allowed(self):
        # реальний випадок з demo: модель написала `import numpy as np`, хоча np уже доступний
        from analyst.nl2pandas import run_code

        self.assertEqual(run_code("import numpy as np\nimport pandas as pd\nresult = int(np.sum([1, 2]))", DF), 3)
        with self.assertRaises(UnsafeCodeError):
            check_code_safety("import numpy as os")

    def test_retry_after_error(self):
        broken = {"feasible": True, "code": "result = df.groupby('cntry')['revenue'].sum()",
                  "explanation": "e", "assumptions": []}
        fixed = {"feasible": True, "code": "result = df.groupby('country')['revenue'].sum().sort_values().tail(3)",
                 "explanation": "e", "assumptions": []}
        llm = FakeLLM({"nl2pandas": [broken, fixed], "nl2pandas_answer": [{"answer": "ok", "caveats": []}]})
        r = ask(DF, "виручка по країнах", llm)
        self.assertIsNone(r.error)
        self.assertEqual(r.attempts, 2)
        self.assertIn("KeyError", llm.calls[1][1])  # помилка передана моделі

    def test_syntax_error_is_retried(self):
        # реальний випадок з demo: модель вставила в code шматки JSON
        broken = {"feasible": True, "code": "result = df.shape[0], \"assumptions\": \"x\" }",
                  "explanation": "e", "assumptions": []}
        fixed = {"feasible": True, "code": "result = df.shape[0]", "explanation": "e", "assumptions": []}
        llm = FakeLLM({"nl2pandas": [broken, fixed], "nl2pandas_answer": [{"answer": "ok", "caveats": []}]})
        r = ask(DF, "скільки рядків", llm)
        self.assertIsNone(r.error)
        self.assertEqual(r.attempts, 2)

    def test_infeasible_question(self):
        llm = FakeLLM({"nl2pandas": [{"feasible": False, "code": "", "explanation": "Немає даних про погоду.",
                                      "assumptions": []}]})
        r = ask(DF, "Яка погода була в Києві?", llm)
        self.assertFalse(r.feasible)

    def test_empty_question(self):
        r = ask(DF, "   ", FakeLLM({}))
        self.assertFalse(r.feasible)

    def test_answer_numbers_checked_against_result(self):
        code = "result = df.groupby('channel')['converted'].mean().round(4).sort_values()"
        llm = FakeLLM({"nl2pandas": [{"feasible": True, "code": code, "explanation": "", "assumptions": []}],
                       "nl2pandas_answer": [{"answer": "Найгірша конверсія у paid_social — 1.67%, найкраща 9.9%.",
                                             "caveats": []}]})
        r = ask(DF, "конверсія по каналах", llm)
        self.assertEqual(len(r.verification.unverified), 1)  # 9.9% вигадано
        self.assertEqual(r.verification.unverified[0].raw, "9.9%")


# ------------------------------------------------------------------ summary

class TestSummary(unittest.TestCase):
    def test_empty_metrics(self):
        with self.assertRaises(DataError):
            executive_summary({}, FakeLLM({}))

    def test_kpis(self):
        k = compute_kpis(DF, ("2026-10-19", "2026-10-25"))["kpis"]
        self.assertLess(k["conversion_rate"]["relative_change"], -0.15)

    def test_offline_returns_numbers_only(self):
        r = executive_summary({"x": 1}, LLMClient(offline=True))
        self.assertIsNone(r["summary"])


if __name__ == "__main__":
    unittest.main()
