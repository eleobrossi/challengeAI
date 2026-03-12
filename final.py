import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2

INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "dataset_all_suspicious_flagged.csv"

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
SUSPICIOUS_SCORE_MIN = 2          # abbassato da 3 a 2
SALARY_WINDOW_DAYS = 7
BURST_WINDOW_HOURS = 3
BURST_MIN_TX = 3
NEAR_THRESHOLD_VALUES = [250, 500, 1000, 3000]
NEAR_THRESHOLD_MARGIN = 0.03
SPLIT_WINDOW_HOURS = 24
SPLIT_MIN_TX = 2
SPLIT_AMOUNT_DIFF = 5.0
IMPOSSIBLE_TRAVEL_KMH = 900
MIN_FUNNEL_SENDERS = 2
NIGHT_HOUR_START = 0
NIGHT_HOUR_END = 5
MAX_SCORE = 60


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_str(series):
    return (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "Unknown": pd.NA})
    )

def normalize_description(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    return re.sub(r"\s+", " ", x)

def semantic_label(desc):
    d = normalize_description(desc)
    if d == "":
        return "missing"
    if "salary" in d:
        return "salary"
    if "rent" in d:
        return "rent"
    if "tax" in d:
        return "tax"
    if "subscription" in d or "streaming" in d:
        return "subscription"
    if "gym" in d:
        return "gym"
    if "app fee" in d or "monthly fee" in d or "scheduling app" in d:
        return "app_fee"
    if any(k in d for k in ["marketplace", "e-commerce", "ecommerce", "shop", "store"]):
        return "commerce"
    return "other"

def is_generic_description(desc):
    d = normalize_description(desc)
    generic = {"", "payment", "transfer", "invoice", "purchase", "fee",
                "services", "monthly payment", "misc", "other"}
    return d in generic or len(d) < 6

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    except Exception:
        return None

def is_near_threshold(x):
    if pd.isna(x):
        return False
    for t in NEAR_THRESHOLD_VALUES:
        if abs(x - t) / t <= NEAR_THRESHOLD_MARGIN:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

required = [
    "transaction_id", "timestamp", "date_only", "time_only", "transaction_type",
    "amount", "balance_after", "description", "sender_id", "sender_iban",
    "sender_name", "sender_job", "sender_residence_city", "sender_salary",
    "sender_last_city", "sender_last_lat", "sender_last_lng",
    "sender_last_location_time", "sender_location_hours_before_tx",
    "recipient_id", "recipient_iban", "recipient_name", "recipient_job",
    "recipient_salary", "payment_method", "transaction_location"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Colonne mancanti: {missing}")

for col in [
    "transaction_id", "transaction_type", "description", "sender_id", "sender_iban",
    "sender_name", "sender_job", "sender_residence_city", "sender_last_city",
    "sender_location_hours_before_tx", "recipient_id", "recipient_iban",
    "recipient_name", "recipient_job", "payment_method", "transaction_location",
    "time_only", "date_only"
]:
    df[col] = clean_str(df[col])

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["sender_last_location_time"] = pd.to_datetime(df["sender_last_location_time"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
df["sender_salary"] = pd.to_numeric(df["sender_salary"], errors="coerce")
df["sender_last_lat"] = pd.to_numeric(df["sender_last_lat"], errors="coerce")
df["sender_last_lng"] = pd.to_numeric(df["sender_last_lng"], errors="coerce")

df["description_norm"] = df["description"].apply(normalize_description)
df["semantic_label"] = df["description"].apply(semantic_label)
df["is_generic_description"] = df["description"].apply(is_generic_description)

df["is_salary_like"] = df["description_norm"].str.contains("salary", na=False)
df["is_rent_like"] = df["description_norm"].str.contains("rent", na=False)
df["is_tax_like"] = df["description_norm"].str.contains("tax", na=False)
df["is_subscription_like"] = df["description_norm"].str.contains(
    "subscription|streaming|gym|monthly fee|app fee|scheduling app", na=False)
df["is_ecommerce_like"] = df["description_norm"].str.contains(
    "marketplace|shop|store|e-commerce|ecommerce", na=False)
df["is_missing_description"] = df["semantic_label"] == "missing"
df["hour_of_day"] = pd.to_datetime(df["time_only"], format="%H:%M:%S", errors="coerce").dt.hour

df = df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE GLOBAL STATS
# ─────────────────────────────────────────────────────────────────────────────
funnel_map = (
    df.groupby("recipient_iban", dropna=False)["sender_id"].nunique()
      .rename("distinct_senders").to_dict()
)

recipient_risk = (
    df.groupby("recipient_id", dropna=False)
      .agg(
          rec_distinct_senders=("sender_id", "nunique"),
          rec_distinct_descriptions=("description_norm", "nunique"),
          rec_distinct_semantic_labels=("semantic_label", "nunique"),
          rec_tx_count=("transaction_id", "count"),
          rec_distinct_payment_methods=("payment_method", "nunique"),
          rec_total_amount=("amount", "sum"),
          rec_amount_mean=("amount", "mean"),
          rec_amount_std=("amount", "std"),
      )
      .reset_index()
)
recipient_risk_map = recipient_risk.set_index("recipient_id").to_dict(orient="index")

salary_tx = df[df["is_salary_like"]].copy()[
    ["recipient_id", "timestamp", "amount"]
].rename(columns={"recipient_id": "person_id", "timestamp": "salary_ts", "amount": "salary_amt"})

sender_stats = (
    df.groupby("sender_id", dropna=False)["amount"]
      .agg(sender_amount_mean="mean", sender_amount_std="std", sender_tx_count="count")
      .reset_index()
)
df = df.merge(sender_stats, on="sender_id", how="left")
df["sender_amount_std"] = df["sender_amount_std"].replace(0, np.nan)
df["sender_amount_zscore"] = (df["amount"] - df["sender_amount_mean"]) / df["sender_amount_std"]
df["sender_amount_percentile"] = df.groupby("sender_id")["amount"].rank(pct=True, method="average")

pair_stats = (
    df.groupby(["sender_id", "recipient_id"], dropna=False)["amount"]
      .agg(pair_amount_mean="mean", pair_amount_std="std", pair_tx_count="count")
      .reset_index()
)
df = df.merge(pair_stats, on=["sender_id", "recipient_id"], how="left")
df["pair_amount_std"] = df["pair_amount_std"].replace(0, np.nan)
df["pair_amount_zscore"] = (df["amount"] - df["pair_amount_mean"]) / df["pair_amount_std"]
df["pair_growth_ratio"] = np.where(df["pair_amount_mean"] > 0, df["amount"] / df["pair_amount_mean"], np.nan)

df["amount_plus_balance"] = df["amount"] + df["balance_after"]
df["amount_over_total_funds"] = np.where(df["amount_plus_balance"] > 0, df["amount"] / df["amount_plus_balance"], np.nan)
df["amount_over_balance_after"] = np.where(df["balance_after"] > 0, df["amount"] / df["balance_after"], np.nan)
df["amount_over_salary"] = np.where(df["sender_salary"] > 0, df["amount"] / df["sender_salary"], np.nan)

sender_type_counts = df.groupby(["sender_id", "transaction_type"], dropna=False).size().reset_index(name="cnt")
sender_pm_counts = df.groupby(["sender_id", "payment_method"], dropna=False).size().reset_index(name="cnt")

df["near_threshold"] = df["amount"].apply(is_near_threshold)

df = df.sort_values(["sender_id", "recipient_id", "timestamp"]).reset_index(drop=True)
split_flags = set()
for (sid, rid), g in df.groupby(["sender_id", "recipient_id"], sort=False):
    g = g.sort_values("timestamp")
    idxs = list(g.index)
    for i in range(len(idxs)):
        cur = df.loc[idxs[i]]
        similar = 1
        for j in range(i + 1, len(idxs)):
            nxt = df.loc[idxs[j]]
            hrs = (nxt["timestamp"] - cur["timestamp"]).total_seconds() / 3600
            if hrs > SPLIT_WINDOW_HOURS:
                break
            if abs(nxt["amount"] - cur["amount"]) <= SPLIT_AMOUNT_DIFF:
                similar += 1
        if similar >= SPLIT_MIN_TX:
            split_flags.add(idxs[i])
df["possible_split_payment"] = df.index.isin(split_flags)

df = df.sort_values("timestamp").reset_index(drop=True)
sender_cities_map = df.groupby("sender_id")["sender_last_city"].apply(lambda x: set(x.dropna())).to_dict()

desc_stats = (
    df.groupby("description_norm", dropna=False)["amount"]
      .agg(desc_amount_mean="mean", desc_amount_std="std", desc_tx_count="count")
      .reset_index()
)
df = df.merge(desc_stats, on="description_norm", how="left")
df["desc_amount_std"] = df["desc_amount_std"].replace(0, np.nan)
df["desc_amount_zscore"] = (df["amount"] - df["desc_amount_mean"]) / df["desc_amount_std"]

type_stats = (
    df.groupby("transaction_type", dropna=False)["amount"]
      .agg(type_amount_mean="mean", type_amount_std="std", type_tx_count="count")
      .reset_index()
)
df = df.merge(type_stats, on="transaction_type", how="left")
df["type_amount_std"] = df["type_amount_std"].replace(0, np.nan)
df["type_amount_zscore"] = (df["amount"] - df["type_amount_mean"]) / df["type_amount_std"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING LOOP
# ─────────────────────────────────────────────────────────────────────────────
all_scores = []
all_reasons = []

for sender_id, g in df.groupby("sender_id", sort=False):
    g = g.sort_values("timestamp").copy()

    seen_sender_ibans = set()
    seen_recipient_ids = set()
    seen_recipient_ibans = set()
    seen_payment_methods = set()
    seen_type_pm = set()
    seen_city_pm = set()
    seen_recipient_pm = set()
    seen_semantic_labels = set()   # ← usa label, non stringa raw
    seen_hours = set()
    stable_recipient_counts = {}

    st = sender_type_counts[sender_type_counts["sender_id"] == sender_id]
    sp = sender_pm_counts[sender_pm_counts["sender_id"] == sender_id]

    dominant_type = st.sort_values("cnt", ascending=False).iloc[0]["transaction_type"] if not st.empty else None
    dominant_type_cnt = int(st.sort_values("cnt", ascending=False).iloc[0]["cnt"]) if not st.empty else 0
    dominant_pm = sp.sort_values("cnt", ascending=False).iloc[0]["payment_method"] if not sp.empty else None
    dominant_pm_cnt = int(sp.sort_values("cnt", ascending=False).iloc[0]["cnt"]) if not sp.empty else 0

    person_salary_events = salary_tx[salary_tx["person_id"] == sender_id].sort_values("salary_ts")

    row_results = {}

    for idx, row in g.iterrows():
        score = 0
        reasons = []

        sender_iban = row["sender_iban"]
        recipient_id = row["recipient_id"]
        recipient_iban = row["recipient_iban"]
        pm = row["payment_method"]
        tx_type = row["transaction_type"]
        city = row["sender_last_city"]
        hour = row["hour_of_day"]
        desc_norm = row["description_norm"]
        sem_label = row["semantic_label"]
        ts = row["timestamp"]
        lat = row["sender_last_lat"]
        lng = row["sender_last_lng"]
        loc_ts = row["sender_last_location_time"]
        residence = row["sender_residence_city"]
        salary = row["sender_salary"]
        job = row["sender_job"]

        prev_count_for_recipient = stable_recipient_counts.get(recipient_id, 0)
        is_new_recipient = pd.notna(recipient_id) and recipient_id not in seen_recipient_ids

        # ── AREA 1: IDENTITY & ACCOUNT GRAPH ─────────────────────────────────
        if pd.notna(sender_iban) and len(seen_sender_ibans) > 0 and sender_iban not in seen_sender_ibans:
            score += 4
            reasons.append(f"A1:new_sender_iban:{sender_iban}")
            if prev_count_for_recipient >= 3:
                score += 2
                reasons.append(f"A1:new_iban_on_stable_recipient:{recipient_id}")

        if is_new_recipient:
            score += 3
            reasons.append(f"A1:new_recipient_id:{recipient_id}")

        if pd.notna(recipient_iban) and recipient_iban in funnel_map:
            n_senders = funnel_map[recipient_iban]
            if n_senders >= MIN_FUNNEL_SENDERS:
                score += 2
                reasons.append(f"A1:funnel_recipient_iban:senders={n_senders}")

        if is_new_recipient and not person_salary_events.empty:
            recent_salary = person_salary_events[
                (person_salary_events["salary_ts"] <= ts) &
                ((ts - person_salary_events["salary_ts"]).dt.days <= SALARY_WINDOW_DAYS)
            ]
            if not recent_salary.empty:
                days_after = (ts - recent_salary.iloc[-1]["salary_ts"]).total_seconds() / 86400
                score += 3
                reasons.append(f"A1:new_recipient_after_salary:{days_after:.1f}d")

        # ── AREA 2: TEMPORAL ──────────────────────────────────────────────────
        if pd.notna(hour) and len(seen_hours) >= 3 and NIGHT_HOUR_START <= hour <= NIGHT_HOUR_END:
            if hour not in seen_hours:
                score += 2
                reasons.append(f"A2:new_night_hour:{hour}h")

        same_day_tx = g[g["date_only"] == row["date_only"]]
        if len(same_day_tx) >= BURST_MIN_TX:
            recent_burst = g[
                (g["timestamp"] <= ts) &
                (g["timestamp"] >= ts - pd.Timedelta(hours=BURST_WINDOW_HOURS))
            ]
            if len(recent_burst) >= BURST_MIN_TX:
                score += 2
                reasons.append(f"A2:burst_tx:{len(recent_burst)}_in_{BURST_WINDOW_HOURS}h")

        if not person_salary_events.empty and not row["is_salary_like"]:
            very_recent = person_salary_events[
                (person_salary_events["salary_ts"] <= ts) &
                ((ts - person_salary_events["salary_ts"]).dt.total_seconds() <= 12 * 3600)
            ]
            if not very_recent.empty:
                score += 1
                reasons.append("A2:tx_within_12h_after_salary")

        # ── AREA 3: AMOUNT & BALANCE ──────────────────────────────────────────
        if pd.notna(row["sender_amount_zscore"]) and row["sender_tx_count"] >= 5 and row["sender_amount_zscore"] >= 2.5:
            score += 2
            reasons.append(f"A3:sender_amount_zscore:{row['sender_amount_zscore']:.2f}")

        if pd.notna(row["pair_amount_zscore"]) and row["pair_tx_count"] >= 3 and row["pair_amount_zscore"] >= 2.0:
            score += 3
            reasons.append(f"A3:pair_amount_zscore:{row['pair_amount_zscore']:.2f}")

        if pd.notna(row["sender_amount_percentile"]) and row["sender_amount_percentile"] >= 0.98 and row["sender_tx_count"] >= 5:
            score += 1
            reasons.append("A3:sender_percentile_top2pct")

        if pd.notna(row["amount_over_salary"]) and row["amount_over_salary"] >= 0.35:
            score += 2
            reasons.append(f"A3:amount_over_salary:{row['amount_over_salary']:.2f}")

        if pd.notna(row["amount_over_total_funds"]) and row["amount_over_total_funds"] >= 0.45:
            score += 2
            reasons.append(f"A3:amount_over_total_funds:{row['amount_over_total_funds']:.2f}")

        if pd.notna(row["amount_over_balance_after"]) and row["amount_over_balance_after"] >= 0.8:
            score += 2
            reasons.append(f"A3:amount_over_balance_after:{row['amount_over_balance_after']:.2f}")

        if pd.notna(row["pair_growth_ratio"]) and row["pair_tx_count"] >= 3 and row["pair_growth_ratio"] >= 1.8:
            score += 2
            reasons.append(f"A3:pair_growth_ratio:{row['pair_growth_ratio']:.2f}")

        if pd.notna(row["desc_amount_zscore"]) and row["desc_tx_count"] >= 3 and row["desc_amount_zscore"] >= 2.5:
            score += 2
            reasons.append(f"A3:desc_amount_zscore:{row['desc_amount_zscore']:.2f}")

        if row["near_threshold"]:
            score += 1
            reasons.append("A3:near_threshold_amount")

        if row["possible_split_payment"]:
            score += 2
            reasons.append("A3:possible_split_payment")

        # ── AREA 4: GEOGRAPHIC ────────────────────────────────────────────────
        if pd.notna(city) and pd.notna(residence) and city != residence:
            score += 1
            reasons.append(f"A4:city_mismatch:{city}!={residence}")

        if pd.notna(row["sender_location_hours_before_tx"]):
            try:
                loc_hrs = float(str(row["sender_location_hours_before_tx"]).replace("h", ""))
                if loc_hrs > 48:
                    score += 1
                    reasons.append(f"A4:stale_location:{loc_hrs:.0f}h")
            except Exception:
                pass

        # ── AREA 5: PAYMENT METHOD ────────────────────────────────────────────
        if pd.notna(pm) and len(seen_payment_methods) > 0 and pm not in seen_payment_methods:
            score += 3
            reasons.append(f"A5:first_payment_method:{pm}")

        type_pm_key = (tx_type, pm)
        if pd.notna(tx_type) and pd.notna(pm) and len(seen_type_pm) > 0 and type_pm_key not in seen_type_pm:
            score += 2
            reasons.append(f"A5:new_type_pm_combo:{tx_type}|{pm}")

        city_pm_key = (city, pm)
        if pd.notna(city) and pd.notna(pm) and len(seen_city_pm) > 0 and city_pm_key not in seen_city_pm:
            score += 2
            reasons.append(f"A5:new_city_pm:{city}|{pm}")

        rec_pm_key = (recipient_id, pm)
        if pd.notna(recipient_id) and pd.notna(pm) and len(seen_recipient_pm) > 0 and rec_pm_key not in seen_recipient_pm:
            score += 2
            reasons.append(f"A5:new_recipient_pm:{recipient_id}|{pm}")

        if dominant_type and dominant_type_cnt >= 5 and tx_type != dominant_type:
            score += 2
            reasons.append(f"A5:type_drift:{dominant_type}->{tx_type}")

        if dominant_pm and dominant_pm_cnt >= 5 and pd.notna(pm) and pm != dominant_pm:
            score += 2
            reasons.append(f"A5:pm_drift:{dominant_pm}->{pm}")

        if tx_type == "transfer" and pd.notna(pm):
            score += 2
            reasons.append(f"A5:pm_on_transfer:{pm}")

        if tx_type == "e-commerce" and pd.isna(pm):
            score += 2
            reasons.append("A5:missing_pm_for_ecommerce")

        if pd.notna(hour) and tx_type in ["e-commerce", "direct debit"] and (hour < 5 or hour > 23):
            score += 1
            reasons.append(f"A5:odd_hour_{tx_type}@{hour}h")

        # ── AREA 6: PROFILE CONSISTENCY ───────────────────────────────────────
        if pd.notna(salary) and salary > 0 and pd.notna(row["amount"]):
            if row["amount"] > salary * 0.8 and not row["is_salary_like"]:
                score += 2
                reasons.append(f"A6:amount_exceeds_80pct_salary")

        if pd.notna(job) and str(job).lower() == "student" and pd.notna(row["amount"]) and row["amount"] > 2000:
            score += 1
            reasons.append(f"A6:student_high_amount:{row['amount']:.0f}")

        if pd.notna(job) and str(job).lower() == "student" and pd.notna(salary) and salary == 0:
            if pd.notna(row["amount"]) and row["amount"] > 500:
                score += 1
                reasons.append(f"A6:student_zero_salary_high_amount")

        if pd.notna(recipient_id) and recipient_id in recipient_risk_map:
            rr = recipient_risk_map[recipient_id]
            if rr["rec_distinct_senders"] >= 3 and rr["rec_distinct_semantic_labels"] >= 3:
                score += 2
                reasons.append(f"A6:recipient_multi_profile:senders={rr['rec_distinct_senders']}")

        # ── AREA 7: SEMANTIC SIGNALS ──────────────────────────────────────────
        # Confronto su semantic_label, non stringa raw
        if sem_label not in ("missing",) and len(seen_semantic_labels) > 0 and sem_label not in seen_semantic_labels:
            score += 2
            reasons.append(f"A7:new_semantic_label:{sem_label}")

        if row["is_missing_description"]:
            score += 3
            reasons.append("A7:missing_description")
        elif row["is_generic_description"]:
            score += 2
            reasons.append("A7:generic_description")

        if sem_label == "salary" and tx_type != "transfer":
            score += 3
            reasons.append(f"A7:salary_type_mismatch:{tx_type}")

        if sem_label == "rent" and tx_type not in ["transfer", "direct debit"]:
            score += 2
            reasons.append(f"A7:rent_type_mismatch:{tx_type}")

        if sem_label == "salary" and pd.notna(pm):
            score += 2
            reasons.append(f"A7:salary_with_pm:{pm}")

        if sem_label == "commerce" and tx_type != "e-commerce":
            score += 2
            reasons.append(f"A7:commerce_type_mismatch:{tx_type}")

        # ── AREA 8: RECIPIENT RISK ────────────────────────────────────────────
        if pd.notna(recipient_id) and recipient_id in recipient_risk_map:
            rr = recipient_risk_map[recipient_id]
            if rr["rec_distinct_senders"] >= MIN_FUNNEL_SENDERS:
                score += 1
                reasons.append(f"A8:recipient_shared_senders:{rr['rec_distinct_senders']}")
            if rr["rec_distinct_payment_methods"] >= 3:
                score += 2
                reasons.append(f"A8:recipient_many_pm:{rr['rec_distinct_payment_methods']}")
            if rr["rec_distinct_descriptions"] >= 5:
                score += 2
                reasons.append(f"A8:recipient_many_descriptions:{rr['rec_distinct_descriptions']}")
            if rr["rec_tx_count"] >= 5 and pd.notna(rr["rec_amount_std"]) and rr["rec_amount_std"] > rr["rec_amount_mean"] * 0.5:
                score += 1
                reasons.append("A8:recipient_high_amount_variance")

        # ── MITIGATIONS ───────────────────────────────────────────────────────
        # Ridotte rispetto alla versione precedente
        mitigation = 0

        if row["is_salary_like"] and tx_type == "transfer" and pd.isna(pm):
            mitigation += 2          # era 5
            reasons.append("MIT:salary_transfer")

        if row["is_rent_like"] and tx_type == "transfer":
            mitigation += 1          # era 3
            reasons.append("MIT:rent_transfer")

        if row["is_tax_like"] and tx_type in ["transfer", "direct debit"]:
            mitigation += 2
            reasons.append("MIT:tax_classic")

        if row["is_subscription_like"] and tx_type == "direct debit":
            mitigation += 2          # era 3
            reasons.append("MIT:subscription_direct_debit")

        if row["is_ecommerce_like"] and tx_type == "e-commerce":
            mitigation += 1
            reasons.append("MIT:ecommerce_match")

        final_score = max(score - mitigation, 0)

        row_results[idx] = {
            "score": final_score,
            "reasons": " | ".join(reasons) if reasons else ""
        }

        # Update state
        if pd.notna(sender_iban):
            seen_sender_ibans.add(sender_iban)
        if pd.notna(recipient_id):
            seen_recipient_ids.add(recipient_id)
            stable_recipient_counts[recipient_id] = stable_recipient_counts.get(recipient_id, 0) + 1
        if pd.notna(recipient_iban):
            seen_recipient_ibans.add(recipient_iban)
        if pd.notna(pm):
            seen_payment_methods.add(pm)
        if pd.notna(tx_type) and pd.notna(pm):
            seen_type_pm.add((tx_type, pm))
        if pd.notna(city) and pd.notna(pm):
            seen_city_pm.add((city, pm))
        if pd.notna(recipient_id) and pd.notna(pm):
            seen_recipient_pm.add((recipient_id, pm))
        if sem_label not in ("missing",):
            seen_semantic_labels.add(sem_label)
        if pd.notna(hour):
            seen_hours.add(hour)

    for idx, v in row_results.items():
        all_scores.append((idx, v["score"], v["reasons"]))


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
score_df = pd.DataFrame(all_scores, columns=["_idx", "raw_score", "suspicious_reasons"]).set_index("_idx")
df["raw_score"] = score_df["raw_score"]
df["suspicious_reasons"] = score_df["suspicious_reasons"]
df["raw_score"] = df["raw_score"].fillna(0)
df["suspicious_reasons"] = df["suspicious_reasons"].fillna("")

df["suspicious_score_pct"] = (df["raw_score"].clip(upper=MAX_SCORE) / MAX_SCORE * 100).round(1)
df["suspicious"] = df["raw_score"].apply(lambda x: "yes" if x >= SUSPICIOUS_SCORE_MIN else "false")

drop_cols = [
    "description_norm", "is_generic_description", "is_salary_like", "is_rent_like",
    "is_tax_like", "is_subscription_like", "is_ecommerce_like", "is_missing_description",
    "hour_of_day", "semantic_label", "near_threshold", "possible_split_payment",
    "amount_plus_balance", "amount_over_total_funds", "amount_over_balance_after",
    "amount_over_salary", "sender_amount_mean", "sender_amount_std", "sender_tx_count",
    "sender_amount_zscore", "sender_amount_percentile",
    "pair_amount_mean", "pair_amount_std", "pair_tx_count", "pair_amount_zscore",
    "pair_growth_ratio", "desc_amount_mean", "desc_amount_std", "desc_tx_count",
    "desc_amount_zscore", "type_amount_mean", "type_amount_std", "type_tx_count",
    "type_amount_zscore", "raw_score"
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

sus_cols = ["suspicious_reasons", "suspicious_score_pct", "suspicious"]
other_cols = [c for c in df.columns if c not in sus_cols]
df = df[other_cols + sus_cols]

df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
total = len(df)
yes = (df["suspicious"] == "yes").sum()
false_c = (df["suspicious"] == "false").sum()

print(f"\n=== RISULTATO FINALE ===")
print(f"Totale transazioni : {total}")
print(f"yes   = {yes} ({yes/total*100:.1f}%)")
print(f"false = {false_c} ({false_c/total*100:.1f}%)")
print(f"\nFile salvato: {OUTPUT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILE — solo transaction_id sospetti (formato challenge)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_IDS_FILE = "output.txt"

suspicious_ids = df[df["suspicious"] == "yes"]["transaction_id"].tolist()

with open(OUTPUT_IDS_FILE, "w") as f:
    for tid in suspicious_ids:
        f.write(str(tid) + "\n")

print(f"\n=== OUTPUT CHALLENGE: {OUTPUT_IDS_FILE} ===")
print(f"Transazioni sospette: {len(suspicious_ids)}")
for tid in suspicious_ids:
    print(tid)
