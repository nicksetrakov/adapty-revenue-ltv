# adapty_revenue_export.py
from __future__ import annotations
import json, time, requests
import os
from typing import Dict, Any, List, Tuple
import pandas as pd
from config import APP_API_KEYS  # dict: {app_name: api_key}

import numpy as np

# APP_API_KEYS = {
#     "AFC Live – for Arsenal fans": os.getenv("ARSENAL_API_KEY"),
#     # Добавь остальные приложения сюда
# }

ADAPTY_URL  = "https://api-admin.adapty.io/api/v1/client-api/metrics/cohort/"
TIMEOUT     = 60
MAX_RETRIES = 5

# ---- SETTINGS ----
DATE_RANGE  = ["2024-08-01", "2025-08-31"]
COUNTRY     = "GB"               # ISO-2
# Если хочешь явно зафиксировать поведение — раскомментируй:
# PERIOD_UNIT  = "month"         # default
# PERIOD_TYPE  = "renewals"      # default
# VALUE_FIELD  = "revenue"       # default -> revenue_usd
# ACCOUNTING   = "revenue"       # default; "proceeds" для выручки после комиссий

def _post(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.post(ADAPTY_URL, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
        if r.status_code in (200, 201):
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt, 30))
            continue
        raise RuntimeError(f"Adapty API error {r.status_code}: {r.text}")
    raise RuntimeError(f"Retry limit exceeded")

def fetch_export(api_key: str) -> Dict[str, Any]:
    payload = {
        "filters": {
            "date": DATE_RANGE,
            "country": [COUNTRY],
            # без store/duration -> всё вместе
        },
        # можно оставить дефолты, но явность — наше всё:
        "period_unit": "month",
        "period_type": "renewals",
        "value_type": "absolute",
        "value_field": "revenue",      # вернётся revenue_usd
        "accounting_type": "revenue",  # поменяй на "proceeds", если нужно после комиссий
        "format": "json",
    }
    resp = _post(api_key, payload)
    print(resp)
    return resp

def parse_export_revenue(app: str, country: str, resp: Dict[str, Any], exclude_active_periods: bool = False) -> pd.DataFrame:
    """
    Разбирает Export API → data[*](type='single') → values[{'period', 'revenue_usd', 'currently_active_period', ...}]
    Строит long-таблицу с кумулятивом по revenue_usd.
    """
    rows: List[Dict[str, Any]] = []
    for seg in resp.get("data", []):
        if seg.get("type") != "single":
            continue
        cohort = seg.get("title") or seg.get("segment_start_date")
        installs = int(seg.get("total_installs") or 0)

        values = sorted(seg.get("values", []), key=lambda x: int(x.get("period", 0)))
        cum = 0.0
        for v in values:
            if exclude_active_periods and v.get("currently_active_period"):
                continue
            p = int(v.get("period", 0))
            rev = float(v.get("revenue_usd") or 0.0)
            cum += rev
            rows.append({
                "app": app,
                "country": country.upper(),
                "cohort": cohort,               # 'YYYY-MM-01'
                "period_index": p,              # 1..N
                "period_label": f"P{p}",
                "period_revenue": rev,
                "cum_revenue": cum,
                "cohort_installs": installs,
            })
    return pd.DataFrame(rows)

def make_country_pivots(df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Пивоты по стране и месяцу когорты (агрегируем по всем аппкам).
    Возвращает:
      rev_country:  cumulative revenue, index = (country, cohort)
      ltv_country:  cumulative LTV per install, index = (country, cohort)
    """
    # 1) Сумма кум.выручки по всем аппкам в разрезе (country, cohort, Pn)
    rev_by_period = (
        df_long.groupby(["country", "cohort", "period_index"], as_index=False)["cum_revenue"].sum()
    )
    rev_country = (
        rev_by_period.pivot_table(
            index=["country", "cohort"],
            columns="period_index",
            values="cum_revenue",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=1)
        .rename(columns=lambda c: f"P{int(c)}")
        .reset_index()
    )

    # 2) Правильный знаменатель для LTV: сумма инсталлов по всем аппкам в когорте
    installs_per_app = df_long[["country", "cohort", "app", "cohort_installs"]].drop_duplicates()
    installs = (
        installs_per_app.groupby(["country", "cohort"], as_index=False)["cohort_installs"].sum()
        .rename(columns={"cohort_installs": "total_installs"})
    )
    rev_country = rev_country.merge(installs, on=["country", "cohort"], how="left")

    # 3) LTV: делим каждую P-колонку на total_installs (взвешенное среднее по аппкам)
    ltv_country = rev_country.copy()
    denom = ltv_country["total_installs"].replace({0: np.nan})
    pcols = [c for c in ltv_country.columns if c.startswith("P")]
    for c in pcols:
        ltv_country[c] = ltv_country[c] / denom

    return rev_country, ltv_country


def make_country_totals(rev_country: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Свод «только страна» = сумма по всем когортам в выбранном диапазоне.
    Возвращает:
      rev_country_total: cumulative revenue by country (Pn суммируются по когортам)
      ltv_country_total: cumulative LTV per install by country (Pn = sum(rev)/sum(installs))
    """
    pcols = [c for c in rev_country.columns if c.startswith("P")]
    need = ["country", "total_installs"] + pcols

    # Суммируем по стране
    rev_country_total = (
        rev_country[need].groupby("country", as_index=False).sum(numeric_only=True)
    )

    # LTV: делим суммарную выручку на суммарные инсталлы
    ltv_country_total = rev_country_total.copy()
    denom = ltv_country_total["total_installs"].replace({0: np.nan})
    for c in pcols:
        ltv_country_total[c] = ltv_country_total[c] / denom

    return rev_country_total, ltv_country_total

def make_pivots(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      1) cumulative revenue по Pn (как в твоём sheet)
      2) cumulative LTV per install (ARPU) = cum_revenue / installs
    """
    rev_pivot = (
        df_long.pivot_table(
            index=["country", "cohort", "app", "cohort_installs"],
            columns="period_index",
            values="cum_revenue",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns=lambda c: f"P{c}" if isinstance(c, int) else c)
        .reset_index()
    )

    ltv_pivot = rev_pivot.copy()
    denom = ltv_pivot["cohort_installs"].replace({0: pd.NA})
    for c in [col for col in ltv_pivot.columns if str(col).startswith("P")]:
        ltv_pivot[c] = ltv_pivot[c] / denom
    return rev_pivot, ltv_pivot

def run() -> None:
    frames = []
    for app, key in APP_API_KEYS.items():
        if not key:
            print(f"[WARN] {app}: нет API ключа, пропускаю")
            continue
        resp = fetch_export(key)
        df = parse_export_revenue(app, COUNTRY, resp, exclude_active_periods=False)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("Пусто: не нашли строк type='single' с values[]. Проверь фильтры/диапазон.")
        return

    # long: оставляем аппку
    df_long = (pd.concat(frames, ignore_index=True)
                 .sort_values(["country","cohort","app","period_index"])
                 .reset_index(drop=True))
    df_long.to_csv("au_revenue_long.csv", index=False)

    # пивоты по стране × когорте (агрегация по всем аппкам)
    rev_country, ltv_country = make_country_pivots(df_long)
    rev_country.to_csv("au_revenue_cumulative_by_country_cohort.csv", index=False)
    ltv_country.to_csv("au_ltv_cumulative_by_country_cohort.csv", index=False)

    # пивоты «только страна» (сумма по всем когортам диапазона)
    rev_country_total, ltv_country_total = make_country_totals(rev_country)
    rev_country_total.to_csv("au_revenue_cumulative_by_country_total.csv", index=False)
    ltv_country_total.to_csv("au_ltv_cumulative_by_country_total.csv", index=False)

    print("Saved:")
    print(" - au_revenue_long.csv")
    print(" - au_revenue_cumulative_by_country_cohort.csv")
    print(" - au_ltv_cumulative_by_country_cohort.csv")
    print(" - au_revenue_cumulative_by_country_total.csv")
    print(" - au_ltv_cumulative_by_country_total.csv")

if __name__ == "__main__":
    run()
