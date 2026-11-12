"""Функція 3. Natural Language -> Pandas.

LLM генерує код -> статична перевірка безпеки (AST) -> виконання в обмеженому
просторі імен -> при помилці одна спроба виправлення з текстом помилки ->
коротка відповідь людською мовою, числа в якій перевіряються на результаті.
Код завжди повертається користувачу, щоб аналітик міг перевірити логіку.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import prompts
from .data import schema_description
from .llm import LLMClient, LLMUnavailable
from .verify import VerificationResult, verify_numbers

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview",
    "os", "sys", "subprocess", "shutil", "pathlib", "importlib", "builtins",
}
FORBIDDEN_ATTRS_PREFIX = ("to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_json",
                          "to_feather", "to_hdf", "to_clipboard", "to_html", "to_latex", "to_stata",
                          "read_", "eval", "query_db")
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len",
    "list", "map", "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "print", "reversed", "Exception", "ValueError", "KeyError",
)}


class UnsafeCodeError(ValueError):
    pass


def check_code_safety(code: str) -> None:
    """Відхиляє імпорти, доступ до файлів/ОС, dunder-атрибути та запис у файли.

    SyntaxError навмисно НЕ перетворюється на UnsafeCodeError: зламаний синтаксис —
    це помилка генерації, яку модель може виправити з другої спроби.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not _is_allowed_import(node):
            raise UnsafeCodeError("Імпорти заборонені.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCodeError("global/nonlocal заборонені.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Заборонене ім'я: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError(f"Заборонений атрибут: {node.attr}")
            if node.attr.startswith(FORBIDDEN_ATTRS_PREFIX):
                raise UnsafeCodeError(f"Заборонена операція: {node.attr}")


ALLOWED_IMPORTS = {("pandas", "pd"), ("numpy", "np")}


def _is_allowed_import(node: ast.AST) -> bool:
    """`import pandas as pd` / `import numpy as np` — нешкідливі: pd і np вже є в просторі імен."""
    return isinstance(node, ast.Import) and all((a.name, a.asname) in ALLOWED_IMPORTS for a in node.names)


def _strip_allowed_imports(code: str) -> ast.Module:
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if not _is_allowed_import(n)]
    return tree


def run_code(code: str, df: pd.DataFrame) -> Any:
    check_code_safety(code)
    namespace = {"df": df.copy(), "pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    tree = _strip_allowed_imports(code)  # у пісочниці немає __import__, тому дозволені імпорти вирізаємо
    exec(compile(tree, "<generated>", "exec"), namespace)  # noqa: S102 — код перевірено check_code_safety
    if "result" not in namespace:
        raise ValueError("Код не створив змінну `result`.")
    return namespace["result"]


def _result_to_text(result: Any, max_rows: int = 30) -> str:
    if isinstance(result, pd.DataFrame):
        return result.head(max_rows).to_string()
    if isinstance(result, pd.Series):
        return result.head(max_rows).to_string()
    return repr(result)


def _result_facts(result: Any) -> Any:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        frame = result.to_frame() if isinstance(result, pd.Series) else result
        frame = frame.head(200)
        vals = frame.select_dtypes("number").to_numpy().ravel().tolist()
        idx = [x for x in frame.index.tolist() if isinstance(x, (int, float, np.number))]
        return vals + idx
    if isinstance(result, (int, float, np.number)):
        return [float(result)]
    return str(result)


@dataclass
class AskResult:
    question: str
    feasible: bool
    code: str = ""
    explanation: str = ""
    assumptions: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    attempts: int = 0
    answer: str | None = None
    caveats: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None

    @property
    def result_text(self) -> str:
        return "" if self.result is None else _result_to_text(self.result)


def ask(df: pd.DataFrame, question: str, llm: LLMClient, max_attempts: int = 2) -> AskResult:
    question = (question or "").strip()
    if not question:
        return AskResult(question, feasible=False, explanation="Порожній запит — сформулюйте питання до даних.")
    if not llm.available:
        raise LLMUnavailable("NL->Pandas потребує LLM (ANTHROPIC_API_KEY).")

    schema = schema_description(df)
    out = AskResult(question, feasible=True)
    user = prompts.NL2PANDAS_USER.format(schema=schema, question=question)
    for attempt in range(1, max_attempts + 1):
        out.attempts = attempt
        resp = llm.call_json(task="nl2pandas", system=prompts.NL2PANDAS_SYSTEM, user=user,
                             schema=prompts.NL2PANDAS_SCHEMA, effort="medium")
        out.feasible, out.code = resp["feasible"], resp["code"].strip()
        out.explanation, out.assumptions = resp["explanation"], resp["assumptions"]
        if not out.feasible or not out.code:
            out.feasible = False
            return out
        try:
            out.result = run_code(out.code, df)
            out.error = None
            break
        except UnsafeCodeError as e:  # небезпечний код не виправляємо — відмовляємо
            out.error = f"Код відхилено перевіркою безпеки: {e}"
            return out
        except Exception as e:  # noqa: BLE001 — показуємо будь-яку помилку виконання
            out.error = f"{type(e).__name__}: {e}"
            user = prompts.NL2PANDAS_RETRY.format(schema=schema, question=question,
                                                  code=out.code, error=out.error)
    if out.error:
        return out

    # коротка відповідь людською мовою + перевірка її чисел на фактичному результаті
    resp = llm.call_json(
        task="nl2pandas_answer", system=prompts.ANSWER_SYSTEM,
        user=prompts.ANSWER_USER.format(question=question, code=out.code, result=out.result_text),
        schema=prompts.ANSWER_SCHEMA, effort="low",
    )
    out.answer, out.caveats = resp["answer"], resp["caveats"]
    out.verification = verify_numbers(out.answer, _result_facts(out.result), derived=True)
    return out


def ask_to_dict(r: AskResult) -> dict[str, Any]:
    return {
        "question": r.question, "feasible": r.feasible, "code": r.code, "explanation": r.explanation,
        "assumptions": r.assumptions, "attempts": r.attempts, "error": r.error, "answer": r.answer,
        "caveats": r.caveats, "result": r.result_text,
        "verification": r.verification.to_dict() if r.verification else None,
    }


__all__ = ["ask", "AskResult", "ask_to_dict", "run_code", "check_code_safety", "UnsafeCodeError"]
