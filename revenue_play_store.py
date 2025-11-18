# adapty_revenue_export_all_countries_p0_filters.py
from __future__ import annotations
import json, time, requests
import os
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
from config import APP_API_KEYS  # dict: {app_name: api_key}

ADAPTY_URL  = "https://api-admin.adapty.io/api/v1/client-api/metrics/cohort/"
TIMEOUT     = 60
MAX_RETRIES = 5

# ---- SETTINGS ----
DATE_RANGE  = ["2024-08-01", "2025-07-31"]  # inclusive
COUNTRIES   = [
    "GB", "AU", "BE", "CA", "FR", "DE", "IE", "IT", "MX", "NG", "PH",
    "SN", "SG", "ZA", "CH", "UA", "US"
]
ACCOUNTING  = "revenue"    # "revenue" | "proceeds"
ROUND_TO    = 2
EPS         = 1e-8
MAX_MONTHS  = 12           # витрина P0..P11

# --- OPTIONAL FILTERS ---
# STORE: None | "app_store" | "play_store" | синонимы: "ios","apple","app store","google","android","play store"
STORE: Optional[str] = "play_store"
# DURATIONS: None | str | List[str] из {"weekly","monthly","quarterly","semiannual","annual"} | синонимы "month","year","3m","6m","1y" и т.п.
DURATIONS: Optional[Union[str, List[str]]] = None

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

# --------------------- Filters normalization / suffix ----------------
_ALLOWED_STORES = {"app_store", "play_store"}
_ALLOWED_DURS   = {"weekly", "monthly", "quarterly", "semiannual", "annual"}

def _norm_store(store: Optional[str]) -> Optional[str]:
    if not store:
        return None
    s = str(store).strip().lower().replace("-", "").replace(" ", "")
    mapping = {
        "appstore": "app_store", "ios": "app_store", "apple": "app_store",
        "playstore": "play_store", "googleplay": "play_store", "google": "play_store", "android": "play_store",
    }
    if s in ("app_store", "play_store"):
        return s
    return mapping.get(s)

def _ensure_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

def _norm_durations(durations: Optional[Union[str, List[str]]]) -> List[str]:
    syn = {
        "week": "weekly", "1w": "weekly",
        "month": "monthly", "1m": "monthly",
        "quarter": "quarterly", "3m": "quarterly",
        "halfyear": "semiannual", "semi-annual": "semiannual", "6m": "semiannual",
        "year": "annual", "1y": "annual", "annualy": "annual",
    }
    out = []
    for d in _ensure_list(durations):
        if not d:
            continue
        k = str(d).strip().lower().replace("-", "").replace(" ", "")
        k = syn.get(k, k)
        if k in _ALLOWED_DURS:
            out.append(k)
    # dedup preserving order
    seen = set(); dedup = []
    for v in out:
        if v not in seen:
            seen.add(v); dedup.append(v)
    return dedup

def _validate_filters(store: Optional[str], durations: Optional[Union[str, List[str]]]) -> tuple[Optional[str], List[str]]:
    s = _norm_store(store)
    if store and not s:
        print(f"[WARN] Unknown store '{store}'. Allowed: {_ALLOWED_STORES} (+ synonyms). Store will be ignored.")
    d = _norm_durations(durations)
    if durations and not d:
        print(f"[WARN] Unknown durations '{durations}'. Allowed: {_ALLOWED_DURS} (+ synonyms). Durations will be ignored.")
    return s, d

def _file_suffix(store: Optional[str], durations: Optional[Union[str, List[str]]]) -> str:
    s_norm, d_norm = _validate_filters(store, durations)
    parts = []
    if s_norm:
        parts.append(s_norm)
    if d_norm:
        parts.append("+".join(d_norm))
    return ("_" + "_".join(parts)) if parts else ""

def _fname(base: str, store: Optional[str], durations: Optional[Union[str, List[str]]]) -> str:
    return f"{base}{_file_suffix(store, durations)}.csv"

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
        # полезно увидеть, что мы реально отправили
        try:
            body_preview = json.dumps(payload)[:400]
        except Exception:
            body_preview = "<payload>"
        raise RuntimeError(f"Adapty API error {r.status_code}: {r.text}\nPayload: {body_preview}")
    raise RuntimeError("Retry limit exceeded")

def fetch_export(
    api_key: str,
    *,
    date_range: List[str],
    country: str,
    accounting: str = "revenue",
    store: Optional[str] = None,
    durations: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    # нормализуем перед отправкой
    store_norm, durations_norm = _validate_filters(store, durations)

    filters: Dict[str, Any] = {"date": date_range, "country": [country]}
    if store_norm:
        filters["store"] = [store_norm]          # список, т.к. API ждёт list
    if durations_norm:
        filters["duration"] = durations_norm     # уже список допустимых значений

    payload = {
        "filters": filters,
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
    0-based индексы (P0..), отсечка по end_date и MAX_MONTHS.
    Вход: Adapty values.period = 1..N  → period_index = period-1
    """
    end_ts = pd.to_datetime(end_date)
    rows: List[Dict[str, Any]] = []

    for seg in resp.get("data", []):
        if seg.get("type") != "single":
            continue

        cohort = seg.get("title") or seg.get("segment_start_date")  # 'YYYY-MM-01'
        cohort_ts = pd.to_datetime(cohort)
        values = sorted(seg.get("values", []), key=lambda x: int(x.get("period", 0)))

        cum = 0.0
        for v in values:
            if exclude_active_periods and v.get("currently_active_period"):
                continue
            p1 = int(v.get("period", 0))   # 1..N у Adapty
            p0 = p1 - 1                    # делаем 0-based

            month_start = cohort_ts + pd.DateOffset(months=p0)
            if month_start > end_ts or p0 >= MAX_MONTHS:
                continue

            rev = float(v.get("revenue_usd") or 0.0)
            cum += rev

            rows.append({
                "app": app,
                "country": country.upper(),
                "cohort": cohort,
                "period_index": p0,             # 0..11
                "period_label": f"P{p0}",
                "period_revenue": rev,
                "cum_revenue": cum,
            })
    return pd.DataFrame(rows)

# -------------------------- PIVOTS/FORMAT ----------------------------
def _tail_zero_mask_from_period_sums(rev_period_nums: pd.DataFrame, eps: float) -> pd.DataFrame:
    """
    rev_period_nums: числовые колонки 0..11 с ПОПЕРИОДНЫМИ суммами.
    True там, где справа нет ни одного ненулевого периода (хвост).
    """
    nz = rev_period_nums.abs().gt(eps)
    nz_right = nz.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    return nz_right.eq(0)

def build_country_tables(df_long: pd.DataFrame, *, eps: float = EPS, round_to: int = ROUND_TO
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      rev_raw    — кумулятив по странам×когортам, P0..P11
      rev_pretty — то же, но хвост занулён и округлён
    """
    periodic = (df_long.groupby(["country", "cohort", "period_index"], as_index=False)
                        .agg(period_revenue=("period_revenue", "sum")))

    rev_period_nums = (periodic.pivot_table(index=["country", "cohort"],
                                            columns="period_index",
                                            values="period_revenue",
                                            aggfunc="sum",
                                            fill_value=0.0))
    full_cols = list(range(MAX_MONTHS))  # 0..11
    rev_period_nums = rev_period_nums.reindex(columns=full_cols, fill_value=0.0).sort_index(axis=1)

    rev_raw_nums = rev_period_nums.cumsum(axis=1)
    tail_mask = _tail_zero_mask_from_period_sums(rev_period_nums, eps=eps)
    rev_pretty_nums = rev_raw_nums.where(~tail_mask, 0.0).round(round_to)

    name_cols = [f"P{i}" for i in full_cols]
    rev_raw = rev_raw_nums.copy();    rev_raw.columns = name_cols;    rev_raw = rev_raw.reset_index()
    rev_pretty = rev_pretty_nums.copy(); rev_pretty.columns = name_cols; rev_pretty = rev_pretty.reset_index()
    return rev_raw, rev_pretty

def build_country_long(
    df_long: pd.DataFrame,
    *,
    end_date: str,
    eps: float = EPS,
    round_to: int = ROUND_TO,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    end_ts = pd.to_datetime(end_date)

    # 1) периодные суммы, что реально пришло
    per = (df_long.groupby(["country", "cohort", "period_index"], as_index=False)
                  .agg(period_revenue=("period_revenue", "sum")))

    # 2) ключи (country, cohort) и макс. допустимый индекс по правой границе диапазона
    keys = df_long[["country", "cohort"]].drop_duplicates().copy()
    keys["cohort_ts"] = pd.to_datetime(keys["cohort"])

    # --- ВАЖНО: стабильный подсчёт разницы в месяцах ---
    # months_diff = (годовая разница)*12 + (разница месяцев)
    months = (
        (end_ts.year - keys["cohort_ts"].dt.year) * 12
        + (end_ts.month - keys["cohort_ts"].dt.month)
    ).astype(int)

    # обрезаем: <0 -> 0, >11 -> 11
    keys["max_idx"] = months.clip(lower=0, upper=MAX_MONTHS - 1)
    # ---------------------------------------------------

    # 3) разворачиваем в полную сетку P0..Pk
    keys["period_index"] = keys["max_idx"].apply(lambda m: list(range(m + 1)))
    grid = keys.drop(columns=["cohort_ts", "max_idx"]).explode("period_index", ignore_index=True)
    grid["period_index"] = grid["period_index"].astype(int)
    grid["period_label"] = "P" + grid["period_index"].astype(str)

    # 4) приклеиваем реальные суммы; где нет — ноль
    per_full = (grid.merge(per, on=["country", "cohort", "period_index"], how="left")
                     .fillna({"period_revenue": 0.0})
                     .sort_values(["country", "cohort", "period_index"]))

    # 5) кумулятив (raw)
    per_full["cum_revenue"] = per_full.groupby(["country", "cohort"], group_keys=False)["period_revenue"].cumsum()
    country_long_raw = per_full[["country","cohort","period_index","period_label","period_revenue","cum_revenue"]].copy()

    # 6) pretty: обнуляем хвост после последнего ненулевого period_revenue
    nz = per_full.loc[per_full["period_revenue"].abs() > eps, ["country","cohort","period_index"]]
    last_nz = (nz.groupby(["country","cohort"], as_index=False)["period_index"].max()
                 .rename(columns={"period_index": "last_nz"}))
    pretty = per_full.merge(last_nz, on=["country","cohort"], how="left")
    pretty["last_nz"] = pretty["last_nz"].fillna(-1)
    pretty.loc[pretty["period_index"] > pretty["last_nz"], "cum_revenue"] = 0.0
    country_long_pretty = pretty[["country","cohort","period_index","period_label","period_revenue","cum_revenue"]].copy()

    # 7) округление и подпись страны
    for df in (country_long_raw, country_long_pretty):
        df["period_revenue"] = df["period_revenue"].round(round_to)
        df["cum_revenue"]    = df["cum_revenue"].round(round_to)

    country_long_raw    = add_country_name(country_long_raw)
    country_long_pretty = add_country_name(country_long_pretty)
    cols = ["country","country_name","cohort","period_index","period_label","period_revenue","cum_revenue"]
    return country_long_raw[cols], country_long_pretty[cols]


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
            resp = fetch_export(
                key,
                date_range=DATE_RANGE,
                country=country,
                accounting=ACCOUNTING,
                store=STORE,
                durations=DURATIONS,
            )
            df   = parse_export_revenue(app, country, resp, end_date=end_date, exclude_active_periods=False)
            if not df.empty:
                frames.append(df); had_data = True
        if not had_data:
            print(f"[{country}] Пусто: нет строк type='single' с values[].")

    if not frames:
        print("Пусто: ни одной страны/аппы с данными. Проверь фильтры/диапазон.")
        return

    suffix = _file_suffix(STORE, DURATIONS)

    # общий long (с аппами)
    df_long = (pd.concat(frames, ignore_index=True)
                 .sort_values(["country","cohort","app","period_index"])
                 .reset_index(drop=True))
    df_long_named = add_country_name(df_long)
    df_long_named.to_csv(_fname("reports/revenue_long", STORE, DURATIONS), index=False, float_format="%.2f")

    # страна×когорта (pivot P0..P11)
    rev_raw, rev_pretty = build_country_tables(df_long)
    rev_raw_named    = add_country_name(rev_raw)
    rev_pretty_named = add_country_name(rev_pretty)
    rev_raw_named.to_csv(_fname("reports/revenue_cumulative_by_country_cohort_raw", STORE, DURATIONS), index=False)
    rev_pretty_named.to_csv(_fname("reports/revenue_cumulative_by_country_cohort",     STORE, DURATIONS), index=False, float_format="%.2f")

    # страна×когорта×период (long без аппов)
    country_long_raw, country_long = build_country_long(df_long, end_date=DATE_RANGE[1])
    country_long_raw.to_csv(_fname("reports/revenue_country_long_raw", STORE, DURATIONS), index=False)
    country_long.to_csv     (_fname("reports/revenue_country_long",      STORE, DURATIONS), index=False, float_format="%.2f")

    print("Saved:")
    print(" -", _fname("revenue_long", STORE, DURATIONS))
    print(" -", _fname("revenue_cumulative_by_country_cohort_raw", STORE, DURATIONS))
    print(" -", _fname("revenue_cumulative_by_country_cohort",     STORE, DURATIONS))
    print(" -", _fname("revenue_country_long_raw", STORE, DURATIONS))
    print(" -", _fname("revenue_country_long",      STORE, DURATIONS))

if __name__ == "__main__":
    run()
