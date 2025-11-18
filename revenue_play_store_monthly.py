from __future__ import annotations
import json, time, requests
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

# --- OPTIONAL FILTERS ---
# STORE: None | "app_store" | "play_store" | синонимы: "ios","apple","app store","google","android","play store"
STORE: Optional[str] = "play_store"
# DURATIONS: None | str | List[str]; канон Adapty + синонимы ("monthly","3m","6m","year","annual", "2m" и т.п.)
DURATIONS: Optional[Union[str, List[str]]] = ["Monthly"]

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

# --------------------- Filters normalization & suffix ----------------

def normalize_store(store: Optional[str]) -> Optional[str]:
    """ -> 'app_store' | 'play_store' | None """
    if not store:
        return None
    s = str(store).strip().lower().replace("-", "").replace(" ", "")
    mapping = {
        "appstore": "app_store", "ios": "app_store", "apple": "app_store",
        "playstore": "play_store", "googleplay": "play_store",
        "google": "play_store", "android": "play_store",
    }
    if s in ("app_store", "play_store"):
        return s
    return mapping.get(s)

# Канон Adapty для duration
_CANON_DURS = {
    "weekly": "Weekly",
    "monthly": "Monthly",
    "2 months": "2 months",
    "3 months": "3 months",
    "6 months": "6 months",
    "annual": "Annual",
    "lifetime": "Lifetime",
    "uncategorized": "Uncategorized",
}
# Синонимы → ключ к _CANON_DURS
_DUR_SYNONYMS = {
    "week": "weekly", "1w": "weekly", "w": "weekly",
    "month": "monthly", "1m": "monthly", "m": "monthly",
    "2m": "2 months", "3m": "3 months", "6m": "6 months",
    "year": "annual", "yearly": "annual", "1y": "annual", "y": "annual", "annualy": "annual",
    "life": "lifetime", "other": "uncategorized",
}

def normalize_durations(durations: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Возвращает список канонических значений Adapty или None."""
    if durations is None or durations == []:
        return None
    items = durations if isinstance(durations, list) else [durations]
    canon_values_lower = {v.lower(): v for v in _CANON_DURS.values()}
    out: List[str] = []
    for raw in items:
        if raw is None:
            continue
        k = str(raw).strip().lower().replace("_", " ").replace("-", " ")
        k = " ".join(k.split())
        if k in canon_values_lower:
            out.append(canon_values_lower[k]); continue
        if k in _DUR_SYNONYMS:
            key = _DUR_SYNONYMS[k]; out.append(_CANON_DURS[key]); continue
        if k in _CANON_DURS:
            out.append(_CANON_DURS[k]); continue
        raise ValueError(
            f"Unsupported duration '{raw}'. Use one of: "
            f"{', '.join(['Weekly','Monthly','2 months','3 months','6 months','Annual','Lifetime','Uncategorized'])} "
            f"or synonyms like 'week','2m','3m','6m','year','annual','life'."
        )
    # dedup
    seen=set(); res=[]
    for v in out:
        if v not in seen:
            seen.add(v); res.append(v)
    return res or None

def _duration_slug(v: str) -> str:
    return v.lower().replace(" ", "")

def _file_suffix(store: Optional[str], durations: Optional[Union[str, List[str]]]) -> str:
    s_norm = normalize_store(store)
    d_norm = normalize_durations(durations)
    parts: List[str] = []
    if s_norm: parts.append(s_norm)
    if d_norm: parts.append("+".join(_duration_slug(x) for x in d_norm))
    return ("_" + "_".join(parts)) if parts else ""

def _fname(base: str, store: Optional[str], durations: Optional[Union[str, List[str]]]) -> str:
    return f"{base}{_file_suffix(store, durations)}.csv"

# --- сколько месяцев в одном «индексе» (цикле) ---
def cycle_months_from_durations(d_norm: Optional[List[str]]) -> int:
    """
    Если выбран ровно один тип длительности из Monthly/2/3/6 months/Annual,
    возвращаем соответствующее число месяцев. Иначе — 1 (помесячно).
    """
    if not d_norm:
        return 1
    mapping = {
        "Monthly": 1,
        "2 months": 2,
        "3 months": 3,
        "6 months": 6,
        "Annual": 12,
    }
    candidates = [mapping[v] for v in d_norm if v in mapping]
    return candidates[0] if len(candidates)==1 else 1  # если смешанные — оставляем месяцы

# ----------------------------- HTTP ---------------------------------
def _post(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.post(ADAPTY_URL, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
        if r.status_code in (200, 201):
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt, 30)); continue
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
    filters: Dict[str, Any] = {"date": date_range, "country": [country]}
    s_norm = normalize_store(store)
    if s_norm: filters["store"] = [s_norm]
    d_norm = normalize_durations(durations)
    if d_norm: filters["duration"] = d_norm
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
    Сохраняем помесячные индексы (0-based). Агрегацию по «циклами» делаем ниже.
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
            p1 = int(v.get("period", 0))   # 1..N от Adapty
            p0 = p1 - 1                    # 0..N-1 для нас

            month_start = cohort_ts + pd.DateOffset(months=p0)
            if month_start > end_ts:
                continue

            rev = float(v.get("revenue_usd") or 0.0)
            cum += rev

            rows.append({
                "app": app,
                "country": country.upper(),
                "cohort": cohort,
                "period_index": p0,             # месячный индекс
                "period_label": f"P{p0}",
                "period_revenue": rev,
                "cum_revenue": cum,
            })
    return pd.DataFrame(rows)

# -------------------------- PIVOTS/FORMAT ----------------------------
def _tail_zero_mask_from_period_sums(rev_period_nums: pd.DataFrame, eps: float) -> pd.DataFrame:
    nz = rev_period_nums.abs().gt(eps)
    nz_right = nz.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    return nz_right.eq(0)

def _add_cycle_index(df: pd.DataFrame, cycle_months: int) -> pd.DataFrame:
    """Добавляет колонку cycle_index = floor(period_index / cycle_months)."""
    if cycle_months <= 1:
        df = df.copy()
        df["cycle_index"] = df["period_index"]
        return df
    df = df.copy()
    df["cycle_index"] = (df["period_index"] // cycle_months).astype(int)
    return df

def _max_cycle_index_per_cohort(df_long: pd.DataFrame, end_date: str, cycle_months: int) -> pd.DataFrame:
    """Для каждой (country, cohort) считает максимальный возможный индекс цикла до end_date."""
    end_ts = pd.to_datetime(end_date)
    keys = df_long[["country", "cohort"]].drop_duplicates().copy()
    keys["cohort_ts"] = pd.to_datetime(keys["cohort"])
    months = ((end_ts.year - keys["cohort_ts"].dt.year) * 12 +
              (end_ts.month - keys["cohort_ts"].dt.month)).astype(int)
    keys["max_cycle"] = (months // max(cycle_months, 1)).clip(lower=0)
    return keys[["country", "cohort", "max_cycle"]]

def build_country_tables(
    df_long: pd.DataFrame,
    *,
    end_date: str,
    cycle_months: int,
    eps: float = EPS,
    round_to: int = ROUND_TO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
      rev_raw    — кумулятив по странам×когортам в разрезе ЦИКЛОВ (P0..Pk)
      rev_pretty — то же, но хвост занулён и округлён
    """
    df_cyc = _add_cycle_index(df_long, cycle_months)

    periodic = (df_cyc.groupby(["country", "cohort", "cycle_index"], as_index=False)
                      .agg(period_revenue=("period_revenue", "sum")))

    # сколько циклов потенциально может быть у каждой когорты
    limits = _max_cycle_index_per_cohort(df_long, end_date=end_date, cycle_months=cycle_months)
    global_max = int(limits["max_cycle"].max() if len(limits) else 0)
    full_cols = list(range(global_max + 1))

    rev_period_nums = (periodic.pivot_table(index=["country", "cohort"],
                                            columns="cycle_index",
                                            values="period_revenue",
                                            aggfunc="sum",
                                            fill_value=0.0)
                               .reindex(columns=full_cols, fill_value=0.0)
                               .sort_index(axis=1))

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
    cycle_months: int,
    eps: float = EPS,
    round_to: int = ROUND_TO,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    end_ts = pd.to_datetime(end_date)
    df_cyc = _add_cycle_index(df_long, cycle_months)

    # периодные суммы по ЦИКЛАМ (а не по месяцам)
    per = (df_cyc.groupby(["country", "cohort", "cycle_index"], as_index=False)
                 .agg(period_revenue=("period_revenue", "sum")))

    # сколько циклов потенциально у каждой когорты
    limits = _max_cycle_index_per_cohort(df_long, end_date=end_date, cycle_months=cycle_months)

    # грид всех (country, cohort, 0..max_cycle)
    grid = (limits.assign(cycle_index=lambda x: x["max_cycle"].apply(lambda m: list(range(m + 1))))
                  .drop(columns=["max_cycle"])
                  .explode("cycle_index", ignore_index=True))
    grid["cycle_index"] = grid["cycle_index"].astype(int)
    grid["period_label"] = "P" + grid["cycle_index"].astype(str)

    per_full = (grid.merge(per, on=["country","cohort","cycle_index"], how="left")
                     .fillna({"period_revenue":0.0})
                     .sort_values(["country","cohort","cycle_index"]))

    # кумулятив по циклам
    per_full["cum_revenue"] = per_full.groupby(["country","cohort"], group_keys=False)["period_revenue"].cumsum()

    # pretty: обнуляем хвост после последнего ненулевого cycle
    nz = per_full.loc[per_full["period_revenue"].abs() > eps, ["country","cohort","cycle_index"]]
    last_nz = (nz.groupby(["country","cohort"], as_index=False)["cycle_index"].max()
                 .rename(columns={"cycle_index":"last_nz"}))
    pretty = per_full.merge(last_nz, on=["country","cohort"], how="left")
    pretty["last_nz"] = pretty["last_nz"].fillna(-1)
    pretty.loc[pretty["cycle_index"] > pretty["last_nz"], "cum_revenue"] = 0.0

    # финальные колонки
    country_long_raw = per_full.rename(columns={"cycle_index":"period_index"})[
        ["country","cohort","period_index","period_label","period_revenue","cum_revenue"]
    ].copy()
    country_long_pretty = pretty.rename(columns={"cycle_index":"period_index"})[
        ["country","cohort","period_index","period_label","period_revenue","cum_revenue"]
    ].copy()

    # округление + подпись страны
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

    d_norm = normalize_durations(DURATIONS)
    cycle_months = cycle_months_from_durations(d_norm)

    for country in COUNTRIES:
        had_data = False
        for app, key in APP_API_KEYS.items():
            if not key:
                print(f"[WARN] {country} · {app}: нет API ключа, пропускаю"); continue
            resp = fetch_export(
                key,
                date_range=DATE_RANGE,
                country=country,
                accounting=ACCOUNTING,
                store=STORE,
                durations=DURATIONS,
            )
            df = parse_export_revenue(app, country, resp, end_date=end_date, exclude_active_periods=False)
            if not df.empty:
                frames.append(df); had_data = True
        if not had_data:
            print(f"[{country}] Пусто: нет строк type='single' с values[].")

    if not frames:
        print("Пусто: ни одной страны/аппы с данными. Проверь фильтры/диапазон.")
        return

    # общий long (с аппами) — это всегда помесячный сырый ряд
    df_long = (pd.concat(frames, ignore_index=True)
                 .sort_values(["country","cohort","app","period_index"])
                 .reset_index(drop=True))
    df_long_named = add_country_name(df_long)
    df_long_named.to_csv(_fname("reports/revenue_long", STORE, DURATIONS), index=False, float_format="%.2f")

    # страна×когорта (P0..Pk по ЦИКЛАМ)
    rev_raw, rev_pretty = build_country_tables(df_long, end_date=end_date, cycle_months=cycle_months)
    rev_raw_named    = add_country_name(rev_raw)
    rev_pretty_named = add_country_name(rev_pretty)
    rev_raw_named.to_csv(_fname("reports/revenue_cumulative_by_country_cohort_raw", STORE, DURATIONS), index=False)
    rev_pretty_named.to_csv(_fname("reports/revenue_cumulative_by_country_cohort",     STORE, DURATIONS),
                            index=False, float_format="%.2f")

    # страна×когорта×период (long без аппок, по ЦИКЛАМ)
    country_long_raw, country_long = build_country_long(df_long, end_date=end_date, cycle_months=cycle_months)
    country_long_raw.to_csv(_fname("reports/revenue_country_long_raw", STORE, DURATIONS), index=False)
    country_long.to_csv     (_fname("reports/revenue_country_long",      STORE, DURATIONS), index=False, float_format="%.2f")

    print(f"Cycle months = {cycle_months} (derived from durations: {d_norm})")
    print("Saved:")
    print(" -", _fname("revenue_long", STORE, DURATIONS))
    print(" -", _fname("revenue_cumulative_by_country_cohort_raw", STORE, DURATIONS))
    print(" -", _fname("revenue_cumulative_by_country_cohort",     STORE, DURATIONS))
    print(" -", _fname("revenue_country_long_raw", STORE, DURATIONS))
    print(" -", _fname("revenue_country_long",      STORE, DURATIONS))

if __name__ == "__main__":
    run()
