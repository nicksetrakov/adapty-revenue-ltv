# adapty_revenue_export_all_countries_p0.py
from __future__ import annotations
import json, time, requests
import os
from typing import Dict, Any, List, Tuple
import pandas as pd
from config import APP_API_KEYS  # dict: {app_name: api_key}

ADAPTY_URL  = "https://api-admin.adapty.io/api/v1/client-api/metrics/cohort/"
TIMEOUT     = 60
MAX_RETRIES = 5

# ---- SETTINGS ----
DATE_RANGE  = ["2024-10-01", "2025-09-30"]  # inclusive
COUNTRIES   = ["GB","AU","BE","CA","FR","DE","IE","IT","MX","NG","PH","SN","SG","ZA","CH","UA","US"]
ACCOUNTING  = "revenue"    # "revenue" | "proceeds"
ROUND_TO    = 2
EPS         = 1e-8
MAX_MONTHS  = 12           # жёстко фиксируем витрину на P0..P11

os.makedirs("reports", exist_ok=True)

# ---------- ISO mapping (fallback; pycountry опционально) ----------
try:
    import pycountry
    def iso2_name(iso: str) -> str:
        c = pycountry.countries.get(alpha_2=iso.upper())
        return c.name if c else iso.upper()
except Exception:
    _FALLBACK = {
        "GB": "United Kingdom", "UK": "United Kingdom",
        "AU": "Australia", "BE": "Belgium", "CA": "Canada",
        "FR": "France", "DE": "Germany", "IE": "Ireland", "IT": "Italy",
        "MX": "Mexico", "NG": "Nigeria", "PH": "Philippines", "SN": "Senegal",
        "SG": "Singapore", "ZA": "South Africa", "CH": "Switzerland",
        "UA": "Ukraine", "US": "United States",
    }
    def iso2_name(iso: str) -> str:
        return _FALLBACK.get(iso.upper(), iso.upper())

def add_country_name(df: pd.DataFrame, iso_col: str = "country") -> pd.DataFrame:
    out = df.copy()
    out.insert(1, "country_name", out[iso_col].map(iso2_name))
    return out

# ----------------------------- HTTP ---------------------------------
def _post(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.post(ADAPTY_URL, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
        if r.status_code in (200, 201):
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt, 30))
            continue
        raise RuntimeError(f"Adapty API error {r.status_code}: {r.text}")
    raise RuntimeError("Retry limit exceeded")

def fetch_export(api_key: str, *, date_range: List[str], country: str, accounting: str = "revenue") -> Dict[str, Any]:
    payload = {
        "filters": {"date": date_range, "country": [country]},
        "period_unit": "month",
        "period_type": "renewals",
        "value_type": "absolute",
        "value_field": "revenue",
        "accounting_type": accounting,
        "format": "json",
    }
    return _post(api_key, payload)

# ----------------------------- PARSE --------------------------------
def parse_export_revenue(
    app: str,
    country: str,
    resp: Dict[str, Any],
    *,
    end_date: str,
    exclude_active_periods: bool = False,
) -> pd.DataFrame:
    """
    Парсим Adapty export; делаем 0-based индексацию периодов и режем по end_date.
    Правило включения: cohort_month + index0 <= end_date.
    """
    end_ts = pd.to_datetime(end_date)
    rows: List[Dict[str, Any]] = []

    for seg in resp.get("data", []):
        if seg.get("type") != "single":
            continue

        cohort = seg.get("title") or seg.get("segment_start_date")  # 'YYYY-MM-01'
        cohort_ts = pd.to_datetime(cohort)

        # сортируем по 1-based period от Adapty
        values = sorted(seg.get("values", []), key=lambda x: int(x.get("period", 0)))
        cum = 0.0
        for v in values:
            if exclude_active_periods and v.get("currently_active_period"):
                continue

            p1 = int(v.get("period", 0))      # 1..N из Adapty
            p0 = p1 - 1                        # делаем 0-based

            # отбрасываем всё, что выходит за правую границу по датам
            month_start = cohort_ts + pd.DateOffset(months=p0)
            if month_start > end_ts:
                continue

            # также ограничим общий горизонт витрины до MAX_MONTHS (P0..P11)
            if p0 >= MAX_MONTHS:
                continue

            rev = float(v.get("revenue_usd") or 0.0)
            cum += rev

            rows.append({
                "app": app,
                "country": country.upper(),   # ISO-2
                "cohort": cohort,             # 'YYYY-MM-01'
                "period_index": p0,           # 0..MAX_MONTHS-1
                "period_label": f"P{p0}",
                "period_revenue": rev,
                "cum_revenue": cum,
            })

    return pd.DataFrame(rows)

# -------------------------- PIVOTS/FORMAT ----------------------------
def _tail_zero_mask_from_period_sums(rev_period_nums: pd.DataFrame, eps: float) -> pd.DataFrame:
    """
    Принимает периодные суммы с числовыми колонками 0..11.
    Возвращает маску хвоста (True там, где справа нет ни одного ненулевого периода).
    """
    nz = rev_period_nums.abs().gt(eps)
    nz_right = nz.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    return nz_right.eq(0)

def build_country_tables(df_long: pd.DataFrame, *, eps: float = EPS, round_to: int = ROUND_TO
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делает два датасета:
      - rev_raw: кумулятив по странам×когортам, P0..P11
      - rev_pretty: то же, но хвост занулён и округлён
    """
    # 1) периодные суммы по всем аппам
    periodic = (df_long.groupby(["country", "cohort", "period_index"], as_index=False)
                        .agg(period_revenue=("period_revenue", "sum")))

    # 2) pivot с числовыми колонками 0..11
    rev_period_nums = (periodic.pivot_table(index=["country", "cohort"],
                                            columns="period_index",
                                            values="period_revenue",
                                            aggfunc="sum",
                                            fill_value=0.0))
    # гарантируем полный набор столбцов 0..11
    full_cols = list(range(MAX_MONTHS))  # 0..11
    rev_period_nums = rev_period_nums.reindex(columns=full_cols, fill_value=0.0).sort_index(axis=1)

    # 3) кумулятив по строкам
    rev_raw_nums = rev_period_nums.cumsum(axis=1)

    # 4) красиво занулить хвост
    tail_mask = _tail_zero_mask_from_period_sums(rev_period_nums, eps=eps)
    rev_pretty_nums = rev_raw_nums.where(~tail_mask, 0.0).round(round_to)

    # 5) переименуем столбцы в P0..P11 и вернём индекс
    name_cols = [f"P{i}" for i in full_cols]
    rev_raw = rev_raw_nums.copy()
    rev_raw.columns = name_cols
    rev_pretty = rev_pretty_nums.copy()
    rev_pretty.columns = name_cols

    rev_raw = rev_raw.reset_index()
    rev_pretty = rev_pretty.reset_index()

    return rev_raw, rev_pretty

def build_country_long(df_long: pd.DataFrame, *, eps: float = EPS, round_to: int = ROUND_TO
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    «Полудлинная» таблица: страна×когорта×(P0..P11), без разреза по аппам.
    Возвращает raw и pretty.
    """
    per = (df_long.groupby(["country", "cohort", "period_index"], as_index=False)
                  .agg(period_revenue=("period_revenue", "sum")))

    # ограничиваем period_index в 0..11 (если вдруг где-то остались хвосты)
    per = per[(per["period_index"] >= 0) & (per["period_index"] < MAX_MONTHS)]

    per = per.sort_values(["country", "cohort", "period_index"])
    per["cum_revenue"] = per.groupby(["country", "cohort"], group_keys=False)["period_revenue"].cumsum()
    per["period_label"] = "P" + per["period_index"].astype(int).astype(str)

    country_long_raw = per[["country","cohort","period_index","period_label","period_revenue","cum_revenue"]].copy()

    # хвостовое зануление кумулятива
    nz = per.loc[per["period_revenue"].abs() > eps, ["country", "cohort", "period_index"]]
    last_nz = (nz.groupby(["country", "cohort"], as_index=False)["period_index"].max()
                 .rename(columns={"period_index": "last_nz_period"}))
    pretty = country_long_raw.merge(last_nz, on=["country","cohort"], how="left")
    pretty["last_nz_period"] = pretty["last_nz_period"].fillna(-1)  # если вообще не было платежей

    tail_mask = pretty["period_index"] > pretty["last_nz_period"]
    pretty.loc[tail_mask, "cum_revenue"] = 0.0

    # округление
    pretty["period_revenue"] = pretty["period_revenue"].round(round_to)
    pretty["cum_revenue"]    = pretty["cum_revenue"].round(round_to)

    # имена стран
    raw_named    = add_country_name(country_long_raw)
    pretty_named = add_country_name(pretty)

    cols = ["country","country_name","cohort","period_index","period_label","period_revenue","cum_revenue"]
    return raw_named[cols], pretty_named[cols]

# ------------------------------ RUN ---------------------------------
def run() -> None:
    frames: List[pd.DataFrame] = []
    end_date = DATE_RANGE[1]

    for country in COUNTRIES:
        had_data = False
        for app, key in APP_API_KEYS.items():
            if not key:
                print(f"[WARN] {country} · {app}: нет API ключа, пропускаю")
                continue
            resp = fetch_export(key, date_range=DATE_RANGE, country=country, accounting=ACCOUNTING)
            df   = parse_export_revenue(app, country, resp, end_date=end_date, exclude_active_periods=False)
            if not df.empty:
                frames.append(df)
                had_data = True
        if not had_data:
            print(f"[{country}] Пусто: нет строк type='single' с values[].")

    if not frames:
        print("Пусто: ни одной страны/аппы с данными. Проверь фильтры/диапазон.")
        return

    # общий long (0-based)
    df_long = (pd.concat(frames, ignore_index=True)
                 .sort_values(["country","cohort","app","period_index"])
                 .reset_index(drop=True))
    df_long_named = add_country_name(df_long)
    df_long_named.to_csv("reports/revenue_long.csv", index=False, float_format="%.2f")

    # страна×когорта (pivot c P0..P11)
    rev_raw, rev_pretty = build_country_tables(df_long)
    rev_raw_named    = add_country_name(rev_raw)
    rev_pretty_named = add_country_name(rev_pretty)
    rev_raw_named.to_csv("reports/revenue_cumulative_by_country_cohort_raw.csv", index=False)
    rev_pretty_named.to_csv("reports/revenue_cumulative_by_country_cohort.csv", index=False, float_format="%.2f")

    # страна×когорта×период (long без аппов)
    country_long_raw, country_long = build_country_long(df_long)
    country_long_raw.to_csv("reports/revenue_country_long_raw.csv", index=False)
    country_long.to_csv     ("reports/revenue_country_long.csv",      index=False, float_format="%.2f")

    print("Saved:")
    print(" - revenue_long.csv")
    print(" - revenue_cumulative_by_country_cohort_raw.csv")
    print(" - revenue_cumulative_by_country_cohort.csv")
    print(" - revenue_country_long_raw.csv")
    print(" - revenue_country_long.csv")

if __name__ == "__main__":
    run()
