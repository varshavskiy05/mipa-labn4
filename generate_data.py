"""Генератор синтетичного датасету сесій e-commerce застосунку "ShopFlow".

Датасет навмисно містить «закладені» ефекти (ground truth), щоб можна було
перевірити, чи знаходить їх асистент, і чи не вигадує LLM того, чого немає:

1. 2026-10-19 вийшла Android-версія 5.4.0 з багом у checkout ->
   конверсія Android-сесій на 5.4.0 падає з ~4% до ~1.3%.
   Наслідок: просідання загальної конверсії на тижні 19-25.10.
2. Вівторок 2026-09-15: бот-трафік (~3500 сесій) з каналу affiliate, web,
   тривалість 1-3 с, 1 сторінка, 0 конверсій.
3. paid_social конвертує приблизно вдвічі гірше за середнє, нижчий AOV.
4. email має найвищий середній чек.
5. Проблеми якості даних: пропуски age_group / country, дублікати session_id,
   від'ємна тривалість сесії, від'ємна виручка (повернення), викиди виручки.

Запуск:  python generate_data.py  ->  data/sessions.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2026-08-01", "2026-10-31"
N_USERS = 14_000
OUT = Path(__file__).parent / "data" / "sessions.csv"

COUNTRIES = ["UA", "PL", "DE", "US", "GB", "CZ", "RO"]
COUNTRY_P = [0.38, 0.18, 0.14, 0.11, 0.08, 0.06, 0.05]
CHANNELS = ["organic", "paid_search", "paid_social", "email", "referral", "affiliate"]
CHANNEL_P = [0.34, 0.22, 0.20, 0.09, 0.08, 0.07]
PLATFORMS = ["android", "ios", "web"]
PLATFORM_P = [0.45, 0.30, 0.25]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_P = [0.22, 0.34, 0.24, 0.12, 0.08]

# базова конверсія сесії та середній чек (грн) за каналом
CHANNEL_CR = {"organic": 0.042, "paid_search": 0.047, "paid_social": 0.019,
              "email": 0.061, "referral": 0.050, "affiliate": 0.035}
CHANNEL_AOV = {"organic": 1450, "paid_search": 1380, "paid_social": 980,
               "email": 1900, "referral": 1550, "affiliate": 1250}
PLATFORM_CR_MULT = {"android": 0.95, "ios": 1.15, "web": 0.90}
COUNTRY_AOV_MULT = {"UA": 1.0, "PL": 1.25, "DE": 1.6, "US": 1.8, "GB": 1.55, "CZ": 1.2, "RO": 1.05}


def android_version(date: pd.Timestamp, rng: np.random.Generator) -> str:
    """Розкатка Android-версій: 5.3.x до 19.10, далі швидке оновлення до 5.4.0."""
    release = pd.Timestamp("2026-10-19")
    if date < release:
        return rng.choice(["5.2.3", "5.3.0", "5.3.1"], p=[0.15, 0.35, 0.50])
    days = (date - release).days
    share_new = min(0.85, 0.35 + 0.08 * days)
    return "5.4.0" if rng.random() < share_new else "5.3.1"


def ios_version(date: pd.Timestamp, rng: np.random.Generator) -> str:
    if date < pd.Timestamp("2026-09-20"):
        return rng.choice(["5.2.0", "5.3.0"], p=[0.4, 0.6])
    return rng.choice(["5.3.0", "5.4.0"], p=[0.3, 0.7])


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")

    users = pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(N_USERS)],
        "country": rng.choice(COUNTRIES, N_USERS, p=COUNTRY_P),
        "channel": rng.choice(CHANNELS, N_USERS, p=CHANNEL_P),
        "platform": rng.choice(PLATFORMS, N_USERS, p=PLATFORM_P),
        "age_group": rng.choice(AGE_GROUPS, N_USERS, p=AGE_P),
        "signup_date": rng.choice(dates[:80], N_USERS),
    })
    # активність користувача: paid_social повертається рідше
    activity = rng.gamma(2.0, 1.0, N_USERS)
    activity[users["channel"].eq("paid_social").to_numpy()] *= 0.55
    activity /= activity.sum()

    rows = []
    for d in dates:
        weekend = d.day_of_week >= 5
        n = int(rng.normal(900, 60) * (1.18 if weekend else 1.0) * (1 + 0.002 * (d - dates[0]).days))
        idx = rng.choice(N_USERS, n, p=activity)
        u = users.iloc[idx]
        for rec in u.itertuples(index=False):
            if rec.signup_date > d:  # сесія не може бути до реєстрації -> вважаємо день реєстрації
                is_new = 1
            else:
                is_new = int((d - rec.signup_date).days <= 7)
            if rec.platform == "android":
                ver = android_version(d, rng)
            elif rec.platform == "ios":
                ver = ios_version(d, rng)
            else:
                ver = "web"
            cr = CHANNEL_CR[rec.channel] * PLATFORM_CR_MULT[rec.platform]
            cr *= 0.92 if weekend else 1.0
            cr *= 1.10 if is_new == 0 else 0.85
            if ver == "5.4.0" and rec.platform == "android":
                cr *= 0.32  # баг у checkout
            pages = max(1, int(rng.poisson(6 if rec.channel != "paid_social" else 4)))
            duration = int(max(5, rng.lognormal(5.3, 0.7)))
            added = rng.random() < min(0.95, cr * 4.5)
            converted = added and rng.random() < cr / min(0.95, cr * 4.5)
            discount = int(rng.random() < (0.45 if rec.channel == "email" else 0.18))
            revenue = 0.0
            if converted:
                aov = CHANNEL_AOV[rec.channel] * COUNTRY_AOV_MULT[rec.country]
                revenue = round(float(rng.lognormal(np.log(aov), 0.45)) * (0.9 if discount else 1.0), 2)
            rows.append((rec.user_id, d, int(rng.integers(0, 24)), rec.country, rec.channel,
                         rec.platform, ver, rec.age_group, is_new, duration, pages,
                         int(added), int(converted), discount, revenue))

    # бот-трафік у вівторок 2026-09-15
    bot_day = pd.Timestamp("2026-09-15")
    for i in range(3500):
        rows.append((f"B{i:05d}", bot_day, int(rng.integers(0, 6)), rng.choice(COUNTRIES, p=COUNTRY_P),
                     "affiliate", "web", "web", None, 1, int(rng.integers(1, 4)), 1, 0, 0, 0, 0.0))

    cols = ["user_id", "session_date", "hour", "country", "channel", "platform", "app_version",
            "age_group", "is_new_user", "session_duration_sec", "pages_viewed", "added_to_cart",
            "converted", "discount_used", "revenue"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "session_id", [f"S{i:06d}" for i in range(len(df))])

    # ---- проблеми якості даних ----
    n = len(df)
    df.loc[rng.choice(n, int(n * 0.035), replace=False), "age_group"] = None
    df.loc[rng.choice(n, int(n * 0.008), replace=False), "country"] = None
    neg = rng.choice(n, 40, replace=False)
    df.loc[neg, "session_duration_sec"] = -df.loc[neg, "session_duration_sec"]
    conv_idx = df.index[df["converted"] == 1].to_numpy()
    refunds = rng.choice(conv_idx, 60, replace=False)
    df.loc[refunds, "revenue"] = -df.loc[refunds, "revenue"]
    outl = rng.choice(np.setdiff1d(conv_idx, refunds), 8, replace=False)
    df.loc[outl, "revenue"] = df.loc[outl, "revenue"] * 25
    dups = df.sample(120, random_state=1)
    df = pd.concat([df, dups]).sort_values(["session_date", "hour"]).reset_index(drop=True)

    df["session_date"] = df["session_date"].dt.strftime("%Y-%m-%d")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Збережено {len(df):,} рядків, {df.shape[1]} колонок -> {OUT}")


if __name__ == "__main__":
    main()
