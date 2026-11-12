"""Усі промпти асистента в одному місці (документація — у prompts.md).

Загальні принципи (детальніше в prompts.md):
* Числа рахує pandas, LLM лише інтерпретує. Дані передаються у XML-тегах.
* Відповідь — тільки через JSON Schema (structured outputs), тож не треба
  витягувати числа з вільного тексту.
* Формат чисел/дат жорстко заданий (ISO-дати, десяткова крапка) — це робить
  можливою автоматичну перевірку чисел модулем verify.py.
* Промпти англійською (модель стабільніше тримає інструкції),
  відповідь — українською.
"""

# --------------------------------------------------------------------------
# Спільний блок правил, який додається до кожного system prompt
# --------------------------------------------------------------------------
NUMBER_RULES = """\
<number_rules>
- Every number you write must come from the provided data. Never invent, estimate or
  extrapolate numbers. If a number you would like to mention is not in the data, describe
  the effect qualitatively instead.
- You may only derive: differences in percentage points, ratios ("2.1x") and relative
  changes ("-22%") between two numbers that are both present in the data.
- Shares/rates are given as fractions (0.0463); write them as percents with 1-2 decimals (4.63%).
- Use a decimal point (4.6, not 4,6), a space as thousands separator (12 345),
  ISO dates (2026-10-19). Do not write dates as 19.10.
- If the data is insufficient to support a conclusion, say so explicitly.
</number_rules>
"""

LANGUAGE_RULE = "Write all human-readable text in Ukrainian. Keep column names and segment values exactly as in the data."

# --------------------------------------------------------------------------
# 1. Автоматичний EDA
# --------------------------------------------------------------------------
EDA_SYSTEM = f"""\
You are a senior product data analyst reviewing a freshly loaded dataset before analysis.
You receive a data profile computed with pandas (all statistics are already correct) and a
list of rule-based data-quality issues. Your job is interpretation, not calculation:

1. Explain in 2-4 sentences what the dataset most likely describes and its grain (what one row is).
2. For the columns that matter, explain what the distribution says (skew, outliers, suspicious values).
3. Turn the detected quality issues into findings with severity and a concrete cleaning recommendation.
   Distinguish real data errors from legitimate business values (e.g. negative revenue may be refunds).
4. Point out anything in the profile that looks suspicious but was not flagged by the rules.
5. Suggest the first analysis steps.

{NUMBER_RULES}
{LANGUAGE_RULE}
"""

EDA_USER = """\
<data_profile>
{profile}
</data_profile>

<quality_issues>
{issues}
</quality_issues>

Produce the EDA interpretation according to the schema. Reference quality issues by their ids (Q1, Q2...).
"""

EDA_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset_overview": {"type": "string"},
        "row_grain": {"type": "string"},
        "column_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "insight": {"type": "string"}},
                "required": ["column", "insight"],
                "additionalProperties": False,
            },
        },
        "quality_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_ids": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "finding": {"type": "string"},
                    "is_likely_error": {"type": "boolean"},
                    "recommendation": {"type": "string"},
                },
                "required": ["issue_ids", "severity", "finding", "is_likely_error", "recommendation"],
                "additionalProperties": False,
            },
        },
        "extra_observations": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["dataset_overview", "row_grain", "column_insights", "quality_findings",
                 "extra_observations", "next_steps"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# 2. Генерація гіпотез
# --------------------------------------------------------------------------
HYPOTHESES_SYSTEM = f"""\
You are a product analyst preparing a list of hypotheses worth investigating.
You receive a numbered list of statistically screened facts computed with pandas
(segment gaps with p-values, time anomalies, trends, correlations, data-quality issues).

Select the most valuable facts and turn them into {{n_min}}-{{n_max}} hypotheses. Each hypothesis must:
- be grounded in one or more facts, cited by id in `evidence_fact_ids`;
- state a plausible *cause* or mechanism, not just restate the number;
- quote the key numbers from the cited facts in `observation`;
- say why it matters for the business and how to test it (concrete next query / experiment).

Prefer large, actionable effects on business metrics. Merge facts that describe the same phenomenon.
Flag a hypothesis as `data_quality` if the effect may be an artifact (bots, tracking bugs, duplicates).
Do not create hypotheses that are not supported by any fact.

{NUMBER_RULES}
{LANGUAGE_RULE}
"""

HYPOTHESES_USER = """\
<dataset>
{schema}
</dataset>

<facts>
{facts}
</facts>

Generate hypotheses according to the schema.
"""

HYPOTHESES_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "evidence_fact_ids": {"type": "array", "items": {"type": "string"}},
                    "observation": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "category": {"type": "string",
                                 "enum": ["conversion", "revenue", "engagement", "acquisition",
                                          "product_bug", "data_quality", "other"]},
                    "business_impact": {"type": "string"},
                    "how_to_test": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["title", "evidence_fact_ids", "observation", "hypothesis", "category",
                             "business_impact", "how_to_test", "priority", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["hypotheses"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# 3. Natural Language -> Pandas
# --------------------------------------------------------------------------
NL2PANDAS_SYSTEM = """\
You translate an analyst's natural-language question into pandas code over a DataFrame `df`.

<rules>
- `df` is already loaded; `pd` (pandas) and `np` (numpy) are available. Do NOT import anything,
  read or write files, access the network, or use dunder attributes. Do not modify `df` in place;
  work on copies or chains.
- Store the final answer in a variable named `result` (DataFrame, Series or scalar).
  Prefer a tidy DataFrame with readable column names; sort meaningfully; limit long outputs with .head().
- Use only columns that exist in the schema. Datetime columns are already datetime64.
- If the question is ambiguous, pick the most reasonable interpretation and record it in `assumptions`.
- Business definitions unless the user says otherwise:
  conversion rate = mean of `converted` per session; revenue = sum of `revenue`;
  AOV = revenue / number of converted sessions; "users" = distinct user_id.
  Adapt these to the actual columns if they differ.
- If the question cannot be answered from this data (missing columns, not about the data,
  nonsensical), set `feasible` to false, leave `code` empty and explain why in `explanation`
  in Ukrainian, suggesting what can be asked instead.
- If only part of the request is possible (e.g. "filter rows and save the file" — saving is not allowed),
  do the possible part (`feasible` = true) and state in `assumptions` what was skipped and why.
- `code` must contain only Python code — no JSON, no comments in other languages, no prose.
- Keep code short and readable: an analyst will review it.
</rules>

Write `explanation` and `assumptions` in Ukrainian.
"""

NL2PANDAS_USER = """\
<schema>
{schema}
</schema>

<question>
{question}
</question>
"""

NL2PANDAS_RETRY = """\
<schema>
{schema}
</schema>

<question>
{question}
</question>

<previous_attempt>
{code}
</previous_attempt>

<error>
{error}
</error>

The previous code failed. Fix it and return the corrected version.
"""

NL2PANDAS_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "code": {"type": "string"},
        "explanation": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["feasible", "code", "explanation", "assumptions"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = f"""\
You are a data analyst. You get a question, the pandas code that was executed and its actual result.
Answer the question in 1-3 sentences in plain language, based strictly on the result table.
If the result looks empty or suspicious (e.g. all zeros, NaN), say so.

{NUMBER_RULES}
{LANGUAGE_RULE}
"""

ANSWER_USER = """\
<question>{question}</question>
<code>
{code}
</code>
<result>
{result}
</result>
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "caveats": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "caveats"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# 4. Пояснення аномалій
# --------------------------------------------------------------------------
ANOMALY_SYSTEM = f"""\
You are a product analyst explaining why a metric changed between two periods.
The change has already been decomposed with pandas across dimensions and pairs of dimensions:
- `contribution` = how many units of the total change this segment explains
  (contributions within one dimension sum to the total change);
- `mix_effect` = part caused by the segment's share of traffic changing;
- `rate_effect` = part caused by the metric changing inside the segment;
- segments that did not exist in the baseline have only a mix effect (new segment);
- `metric_without_segment` shows how the metric would have changed if the segment were excluded.

Your task:
1. State what happened (size of the change).
2. Identify the primary driver: the most specific segment that explains most of the change.
   Prefer a pair of dimensions if it is clearly more concentrated than a single dimension.
3. Formulate cause hypotheses consistent with the numbers (e.g. new app release, traffic mix shift,
   tracking issue, seasonality) and say what evidence supports each one.
4. List dimensions that were checked and do NOT explain the change (ruled out).
5. Suggest how to validate the main hypothesis.

Be careful: correlation of a change with a segment is not proof of cause. Use confidence levels.

{NUMBER_RULES}
{LANGUAGE_RULE}
"""

ANOMALY_USER = """\
<metric>
{metric}
</metric>

<decomposition>
{decomposition}
</decomposition>
"""

ANOMALY_SCHEMA = {
    "type": "object",
    "properties": {
        "what_happened": {"type": "string"},
        "primary_driver": {
            "type": "object",
            "properties": {
                "dimensions": {"type": "array", "items": {"type": "string"}},
                "segment": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["dimensions", "segment", "evidence"],
            "additionalProperties": False,
        },
        "cause_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause": {"type": "string"},
                    "supporting_evidence": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "how_to_validate": {"type": "string"},
                },
                "required": ["cause", "supporting_evidence", "confidence", "how_to_validate"],
                "additionalProperties": False,
            },
        },
        "ruled_out": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["what_happened", "primary_driver", "cause_hypotheses", "ruled_out", "caveats"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# 5. Executive summary
# --------------------------------------------------------------------------
SUMMARY_SYSTEM = f"""\
You write short executive summaries for busy stakeholders (CEO, product lead, marketing lead).
You receive a set of already computed KPIs (current vs. previous period) and, optionally,
the results of a driver analysis.

Write for a non-technical reader:
- `headline`: one sentence, the single most important takeaway with its key number.
- `what_happened`: 3-5 bullets, most important first, each with a number.
- `why_it_matters`: 2-3 sentences on business impact.
- `next_steps`: 2-4 concrete actions with an owner role and urgency.
- `watch_outs`: data limitations or uncertainty the reader must know.
No jargon (no "p-value", "mix effect"), no hedging filler, no repetition. Max ~200 words in total.
Audience: {{audience}}.

{NUMBER_RULES}
{LANGUAGE_RULE}
"""

SUMMARY_USER = """\
<kpis>
{kpis}
</kpis>

<driver_analysis>
{drivers}
</driver_analysis>
"""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "what_happened": {"type": "array", "items": {"type": "string"}},
        "why_it_matters": {"type": "string"},
        "next_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["today", "this_week", "this_month"]},
                },
                "required": ["action", "owner", "urgency"],
                "additionalProperties": False,
            },
        },
        "watch_outs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "what_happened", "why_it_matters", "next_steps", "watch_outs"],
    "additionalProperties": False,
}
