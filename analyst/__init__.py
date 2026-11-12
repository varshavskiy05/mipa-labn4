"""AI-асистент аналітика: pandas рахує, Claude інтерпретує, verify.py перевіряє."""
from .anomaly import explain_anomaly, find_biggest_change
from .data import load_csv
from .eda import auto_eda
from .hypotheses import build_facts, suggest_hypotheses
from .llm import LLMClient
from .nl2pandas import ask
from .summary import compute_kpis, executive_summary
from .verify import verify_numbers

__all__ = ["LLMClient", "load_csv", "auto_eda", "suggest_hypotheses", "build_facts", "ask",
           "explain_anomaly", "find_biggest_change", "compute_kpis", "executive_summary", "verify_numbers"]
