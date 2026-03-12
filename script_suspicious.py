import pandas as pd
import numpy as np
import re

INPUT_FILE = "transactions_enriched_with_locations.csv"
OUTPUT_FILE = "transactions_enriched_with_locations_all_suspicious_flags.csv"

IDENTITY_THRESHOLD = 4
PAYMENT_THRESHOLD = 4
SEMANTIC_THRESHOLD = 4
BALANCE_THRESHOLD = 4

MIN_FUNNEL_SENDERS = 2
SALARY_WINDOW_DAYS = 7

NEAR_THRESHOLD_VALUES = [250, 500, 1000, 3000]
NEAR_THRESHOLD_MARGIN = 0.03
SPLIT_WINDOW_HOURS = 24
SPLIT_MIN_TX = 2
SPLIT_AMOUNT_DIFF = 5.0


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
    x = re.sub(r"\s+", " ", x)
    return x


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
    if "marketplace" in d or "e-commerce" in d or "ecommerce" in d or "shop" in d or "store" in d:
        return "commerce"
    return "other"


def is_generic_description(desc):
    d = normalize_description(desc)
    generic_values = {
        "", "payment", "transfer", "invoice", "purchase", "fee", "services",
        "monthly payment", "misc", "other"
    }
    return d in generic_values or len(d) < 6


def is_near_threshold(x):
    if pd.isna(x):
        return False
    for t in NEAR_THRESHOLD_VALUES:
        if abs(x - t) / t <= NEAR_THRESHOLD_MARGIN:
            return True
    return False


df = pd.read_csv(INPUT_FILE)

required = [
    "transaction_id", "timestamp", "time_only", "transaction_type", "amount",
    "balance_after", "description", "sender_id", "sender_iban", "sender_salary",
    "sender_last_city", "recipient_id", "recipient_iban", "payment_method"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Colonne mancanti: {missing}")

for col in [
    "transaction_id", "time_only", "transaction_type", "description",
    "sender_id", "sender_iban", "sender_last_city",
    "recipient_id", "recipient_iban", "payment_method"
]:
    df[col] = clean_str(df[col])

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
df["sender_salary"] = pd.to_numeric(df["sender_salary"], errors="coerce")

df["description_lc"] = df["description"].fillna("").str.lower()
df["description_norm"] = df["description"].apply(normalize_description)
df["semantic_label"] = df["description"].apply(semantic_label)
df["is_generic_description"] = df["description"].apply(is_generic_description)

df["is_salary_like"] = df["description_lc"].str.contains("salary", na=False)
df["is_rent_like"] = df["description_lc"].str.contains("rent", na=False)
df["is_tax_like"] = df["description_lc"].str.contains("tax", na=False)
df["is_subscription_like"] = df["description_lc"].str.contains(
    "subscription|streaming|gym|monthly fee|app fee|scheduling app", na=False
)
df["is_ecommerce_like"] = df["description_lc"].str.contains(
    "marketplace|shop|store|e-commerce|ecommerce", na=False
)
df["is_missing_description"] = df["semantic_label"] == "missing"

df["hour_of_day"] = pd.to_datetime(df["time_only"], format="%H:%M:%S", errors="coerce").dt.hour
df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# Identity graph prep
# -----------------------------
funnel = (
    df.groupby("recipient_iban", dropna=False)
      .agg(
          distinct_sender_id=("sender_id", "nunique"),
          tx_count=("transaction_id", "count"),
          total_amount=("amount", "sum")
      )
      .reset_index()
)
funnel_map = funnel.set_index("recipient_iban").to_dict(orient="index")

salary_tx = df[df["is_salary_like"]].copy()
salary_tx = salary_tx[["recipient_id", "timestamp", "transaction_id", "amount"]].rename(columns={
    "recipient_id": "person_id",
    "timestamp": "salary_timestamp",
    "transaction_id": "salary_transaction_id",
    "amount": "salary_amount"
})

identity_rows = []

for sender_id, g in df.groupby("sender_id", sort=False):
    g = g.sort_values("timestamp").copy()
    seen_sender_ibans = set()
    seen_recipient_ids = set()
    stable_recipient_counts = {}
    person_salary_events = salary_tx[salary_tx["person_id"] == sender_id].sort_values("salary_timestamp")

    for _, row in g.iterrows():
        score = 0
        reasons = []

        sender_iban = row["sender_iban"]
        recipient_id = row["recipient_id"]
        recipient_iban = row["recipient_iban"]
        prev_count_for_recipient = stable_recipient_counts.get(recipient_id, 0)

        if pd.notna(sender_iban) and len(seen_sender_ibans) > 0 and sender_iban not in seen_sender_ibans:
            score += 4
            reasons.append(f"new_sender_iban:{sender_iban}")
            if prev_count_for_recipient >= 3:
                score += 2
                reasons.append(f"new_sender_iban_on_stable_recipient:{recipient_id}")

        is_new_recipient = pd.notna(recipient_id) and recipient_id not in seen_recipient_ids
        if is_new_recipient:
            score += 3
            reasons.append(f"new_recipient_id:{recipient_id}")

        if pd.notna(recipient_iban) and recipient_iban in funnel_map:
            sender_count = int(funnel_map[recipient_iban]["distinct_sender_id"])
            if sender_count >= MIN_FUNNEL_SENDERS:
                score += 2
                reasons.append(f"shared_recipient_iban:senders={sender_count}")

        if is_new_recipient and not person_salary_events.empty:
            recent_salary = person_salary_events[
                (person_salary_events["salary_timestamp"] <= row["timestamp"]) &
                ((row["timestamp"] - person_salary_events["salary_timestamp"]).dt.days <= SALARY_WINDOW_DAYS)
            ]
            if not recent_salary.empty:
                last_salary = recent_salary.iloc[-1]
                score += 3
                reasons.append(
                    f"new_recipient_after_salary:{(row['timestamp'] - last_salary['salary_timestamp']).total_seconds()/86400.0:.1f}d"
                )

        mitigation = 0
        if row["is_rent_like"]:
            mitigation += 2
            reasons.append("mitigation:rent_like")
        if row["is_tax_like"]:
            mitigation += 2
            reasons.append("mitigation:tax_like")
        if row["is_subscription_like"]:
            mitigation += 2
            reasons.append("mitigation:subscription_like")

        final_score = max(score - mitigation, 0)

        identity_rows.append({
            "transaction_id": row["transaction_id"],
            "identity_graph_score": final_score,
            "identity_graph_reasons": " | ".join(reasons) if reasons else ""
        })

        if pd.notna(sender_iban):
            seen_sender_ibans.add(sender_iban)
        if pd.notna(recipient_id):
            seen_recipient_ids.add(recipient_id)
            stable_recipient_counts[recipient_id] = stable_recipient_counts.get(recipient_id, 0) + 1

identity_df = pd.DataFrame(identity_rows)
df = df.merge(identity_df, on="transaction_id", how="left")
df["identity_graph_score"] = df["identity_graph_score"].fillna(0).astype(int)
df["identity_graph_reasons"] = df["identity_graph_reasons"].fillna("")
df["suspicious_identity_graph"] = df["identity_graph_score"].apply(
    lambda x: "yes" if x >= IDENTITY_THRESHOLD else "false"
)

# -----------------------------
# Payment method misuse
# -----------------------------
sender_type_counts = (
    df.groupby(["sender_id", "transaction_type"], dropna=False)
      .size()
      .reset_index(name="cnt")
)
sender_pm_counts = (
    df.groupby(["sender_id", "payment_method"], dropna=False)
      .size()
      .reset_index(name="cnt")
)

payment_rows = []

for sender_id, g in df.groupby("sender_id", sort=False):
    g = g.sort_values("timestamp").copy()

    seen_payment_methods = set()
    seen_type_pm = set()
    seen_city_pm = set()
    seen_recipient_pm = set()

    sender_type_profile = sender_type_counts[sender_type_counts["sender_id"] == sender_id]
    sender_pm_profile = sender_pm_counts[sender_pm_counts["sender_id"] == sender_id]

    dominant_type = None
    dominant_type_cnt = 0
    if not sender_type_profile.empty:
        top_type_row = sender_type_profile.sort_values("cnt", ascending=False).iloc[0]
        dominant_type = top_type_row["transaction_type"]
        dominant_type_cnt = int(top_type_row["cnt"])

    dominant_pm = None
    dominant_pm_cnt = 0
    if not sender_pm_profile.empty:
        top_pm_row = sender_pm_profile.sort_values("cnt", ascending=False).iloc[0]
        dominant_pm = top_pm_row["payment_method"]
        dominant_pm_cnt = int(top_pm_row["cnt"])

    for _, row in g.iterrows():
        score = 0
        reasons = []

        pm = row["payment_method"]
        tx_type = row["transaction_type"]
        city = row["sender_last_city"]
        recipient_id = row["recipient_id"]
        hour = row["hour_of_day"]

        if pd.notna(pm) and len(seen_payment_methods) > 0 and pm not in seen_payment_methods:
            score += 3
            reasons.append(f"first_payment_method_use:{pm}")

        type_pm_key = (tx_type, pm)
        if pd.notna(tx_type) and pd.notna(pm) and len(seen_type_pm) > 0 and type_pm_key not in seen_type_pm:
            score += 2
            reasons.append(f"new_type_payment_combo:{tx_type}|{pm}")

        city_pm_key = (city, pm)
        if pd.notna(city) and pd.notna(pm) and len(seen_city_pm) > 0 and city_pm_key not in seen_city_pm:
            score += 2
            reasons.append(f"new_city_payment_combo:{city}|{pm}")

        recipient_pm_key = (recipient_id, pm)
        if pd.notna(recipient_id) and pd.notna(pm) and len(seen_recipient_pm) > 0 and recipient_pm_key not in seen_recipient_pm:
            score += 2
            reasons.append(f"new_recipient_payment_combo:{recipient_id}|{pm}")

        if dominant_type is not None and dominant_type_cnt >= 5 and tx_type != dominant_type:
            score += 2
            reasons.append(f"transaction_type_drift:{dominant_type}->{tx_type}")

        if dominant_pm is not None and dominant_pm_cnt >= 5 and pd.notna(pm) and pm != dominant_pm:
            score += 2
            reasons.append(f"payment_method_drift:{dominant_pm}->{pm}")

        if pd.notna(hour) and pd.notna(pm):
            if pm in ["PayPal"] and (hour < 6 or hour > 23):
                score += 1
                reasons.append(f"odd_hour_for_payment_method:{pm}@{hour}")
            if tx_type in ["e-commerce", "direct debit"] and (hour < 5 or hour > 23):
                score += 1
                reasons.append(f"odd_hour_for_transaction_type:{tx_type}@{hour}")

        if tx_type == "transfer" and pd.notna(pm):
            score += 2
            reasons.append(f"payment_method_present_on_transfer:{pm}")

        if tx_type == "e-commerce" and pd.isna(pm):
            score += 2
            reasons.append("missing_payment_method_for_ecommerce")

        if tx_type == "direct debit" and pd.notna(pm) and str(pm).lower() in ["paypal"]:
            score += 2
            reasons.append(f"direct_debit_with_unusual_payment_method:{pm}")

        if row["is_salary_like"] and tx_type != "transfer":
            score += 3
            reasons.append(f"salary_description_mismatch:{tx_type}")

        if row["is_rent_like"] and tx_type not in ["transfer", "direct debit"]:
            score += 2
            reasons.append(f"rent_description_mismatch:{tx_type}")

        if row["is_subscription_like"] and tx_type == "transfer":
            score += 1
            reasons.append("subscription_with_transfer")

        mitigation = 0
        if row["is_subscription_like"] and tx_type == "direct debit":
            mitigation += 2
            reasons.append("mitigation:subscription_direct_debit")
        if row["is_tax_like"] and tx_type == "transfer":
            mitigation += 1
            reasons.append("mitigation:tax_transfer")
        if row["is_rent_like"] and tx_type == "transfer":
            mitigation += 2
            reasons.append("mitigation:rent_transfer")
        if row["is_salary_like"] and tx_type == "transfer":
            mitigation += 3
            reasons.append("mitigation:salary_transfer")
        if row["is_ecommerce_like"] and tx_type == "e-commerce":
            mitigation += 1
            reasons.append("mitigation:ecommerce_semantic_match")

        final_score = max(score - mitigation, 0)

        payment_rows.append({
            "transaction_id": row["transaction_id"],
            "payment_method_score": final_score,
            "payment_method_reasons": " | ".join(reasons) if reasons else ""
        })

        if pd.notna(pm):
            seen_payment_methods.add(pm)
        if pd.notna(tx_type) and pd.notna(pm):
            seen_type_pm.add((tx_type, pm))
        if pd.notna(city) and pd.notna(pm):
            seen_city_pm.add((city, pm))
        if pd.notna(recipient_id) and pd.notna(pm):
            seen_recipient_pm.add((recipient_id, pm))

payment_df = pd.DataFrame(payment_rows)
df = df.merge(payment_df, on="transaction_id", how="left")
df["payment_method_score"] = df["payment_method_score"].fillna(0).astype(int)
df["payment_method_reasons"] = df["payment_method_reasons"].fillna("")
df["suspicious_payment"] = df["payment_method_score"].apply(
    lambda x: "yes" if x >= PAYMENT_THRESHOLD else "false"
)

# -----------------------------
# Semantic signals
# -----------------------------
recipient_desc_stats = (
    df.groupby("recipient_id", dropna=False)
      .agg(
          distinct_descriptions=("description_norm", "nunique"),
          distinct_semantic_labels=("semantic_label", "nunique"),
          tx_count=("transaction_id", "count"),
          distinct_senders=("sender_id", "nunique")
      )
      .reset_index()
)
recipient_desc_map = recipient_desc_stats.set_index("recipient_id").to_dict(orient="index")

desc_recipient_stats = (
    df.groupby(["recipient_id", "description_norm"], dropna=False)
      .agg(
          tx_count=("transaction_id", "count"),
          avg_amount=("amount", "mean")
      )
      .reset_index()
)

semantic_rows = []

for sender_id, g in df.groupby("sender_id", sort=False):
    g = g.sort_values("timestamp").copy()
    seen_descriptions = set()

    for _, row in g.iterrows():
        score = 0
        reasons = []

        desc_norm = row["description_norm"]
        recipient_id = row["recipient_id"]
        tx_type = row["transaction_type"]
        pm = row["payment_method"]

        if desc_norm != "" and len(seen_descriptions) > 0 and desc_norm not in seen_descriptions:
            score += 2
            reasons.append(f"new_description_for_sender:{desc_norm}")

        if row["is_missing_description"]:
            score += 3
            reasons.append("missing_description")
        elif row["is_generic_description"]:
            score += 2
            reasons.append("generic_description")

        if pd.notna(recipient_id) and recipient_id in recipient_desc_map:
            rec_info = recipient_desc_map[recipient_id]
            if rec_info["tx_count"] >= 2 and rec_info["distinct_semantic_labels"] >= 3:
                score += 2
                reasons.append(f"recipient_semantic_diversity:labels={int(rec_info['distinct_semantic_labels'])}")
            if rec_info["distinct_descriptions"] >= 4 and rec_info["distinct_senders"] >= 2:
                score += 2
                reasons.append(f"shared_recipient_with_many_descriptions:{int(rec_info['distinct_descriptions'])}")

        if row["semantic_label"] == "salary" and tx_type != "transfer":
            score += 3
            reasons.append(f"salary_type_mismatch:{tx_type}")

        if row["semantic_label"] == "rent" and tx_type not in ["transfer", "direct debit"]:
            score += 2
            reasons.append(f"rent_type_mismatch:{tx_type}")

        if row["semantic_label"] == "tax" and tx_type not in ["transfer", "direct debit"]:
            score += 2
            reasons.append(f"tax_type_mismatch:{tx_type}")

        if row["semantic_label"] in ["subscription", "gym", "app_fee"] and tx_type == "transfer":
            score += 1
            reasons.append("subscription_transfer_unusual")

        if row["semantic_label"] == "salary" and pd.notna(pm):
            score += 2
            reasons.append(f"salary_with_payment_method:{pm}")

        if row["semantic_label"] == "commerce" and tx_type != "e-commerce":
            score += 2
            reasons.append(f"commerce_description_type_mismatch:{tx_type}")

        if row["semantic_label"] == "other" and pd.notna(recipient_id):
            rec_desc_rows = desc_recipient_stats[
                (desc_recipient_stats["recipient_id"] == recipient_id) &
                (desc_recipient_stats["description_norm"] == desc_norm)
            ]
            if not rec_desc_rows.empty and int(rec_desc_rows.iloc[0]["tx_count"]) == 1:
                score += 1
                reasons.append("one_off_unclassified_description")

        same_desc = df[df["description_norm"] == desc_norm]
        if len(same_desc) >= 3:
            mean_amt = same_desc["amount"].mean()
            std_amt = same_desc["amount"].std()
            if pd.notna(std_amt) and std_amt > 0:
                z = (row["amount"] - mean_amt) / std_amt
                if z >= 2.5:
                    score += 2
                    reasons.append(f"description_amount_zscore:{z:.2f}")

        mitigation = 0
        if row["semantic_label"] == "salary" and tx_type == "transfer" and pd.isna(pm):
            mitigation += 3
            reasons.append("mitigation:classic_salary_pattern")
        if row["semantic_label"] == "rent" and tx_type == "transfer":
            mitigation += 2
            reasons.append("mitigation:classic_rent_pattern")
        if row["semantic_label"] == "tax" and tx_type == "transfer":
            mitigation += 2
            reasons.append("mitigation:classic_tax_pattern")
        if row["semantic_label"] in ["subscription", "gym", "app_fee"] and tx_type == "direct debit":
            mitigation += 2
            reasons.append("mitigation:classic_subscription_pattern")
        if row["semantic_label"] == "commerce" and tx_type == "e-commerce":
            mitigation += 1
            reasons.append("mitigation:classic_commerce_pattern")

        final_score = max(score - mitigation, 0)

        semantic_rows.append({
            "transaction_id": row["transaction_id"],
            "semantic_score": final_score,
            "semantic_reasons": " | ".join(reasons) if reasons else ""
        })

        if desc_norm != "":
            seen_descriptions.add(desc_norm)

semantic_df = pd.DataFrame(semantic_rows)
df = df.merge(semantic_df, on="transaction_id", how="left")
df["semantic_score"] = df["semantic_score"].fillna(0).astype(int)
df["semantic_reasons"] = df["semantic_reasons"].fillna("")
df["suspicious_semantic"] = df["semantic_score"].apply(
    lambda x: "yes" if x >= SEMANTIC_THRESHOLD else "false"
)

# -----------------------------
# Amount & balance behaviour
# -----------------------------
df["amount_plus_balance"] = df["amount"] + df["balance_after"]
df["amount_over_total_funds"] = np.where(
    df["amount_plus_balance"] > 0,
    df["amount"] / df["amount_plus_balance"],
    np.nan
)
df["amount_over_balance_after"] = np.where(
    df["balance_after"] > 0,
    df["amount"] / df["balance_after"],
    np.nan
)
df["amount_over_salary"] = np.where(
    df["sender_salary"] > 0,
    df["amount"] / df["sender_salary"],
    np.nan
)

sender_stats = (
    df.groupby("sender_id", dropna=False)["amount"]
      .agg(sender_amount_mean="mean", sender_amount_std="std", sender_tx_count="count")
      .reset_index()
)
df = df.merge(sender_stats, on="sender_id", how="left")
df["sender_amount_std"] = df["sender_amount_std"].replace(0, np.nan)
df["sender_amount_zscore"] = (df["amount"] - df["sender_amount_mean"]) / df["sender_amount_std"]

pair_stats = (
    df.groupby(["sender_id", "recipient_id"], dropna=False)["amount"]
      .agg(pair_amount_mean="mean", pair_amount_std="std", pair_tx_count="count")
      .reset_index()
)
df = df.merge(pair_stats, on=["sender_id", "recipient_id"], how="left")
df["pair_amount_std"] = df["pair_amount_std"].replace(0, np.nan)
df["pair_amount_zscore"] = (df["amount"] - df["pair_amount_mean"]) / df["pair_amount_std"]

type_stats = (
    df.groupby("transaction_type", dropna=False)["amount"]
      .agg(type_amount_mean="mean", type_amount_std="std", type_tx_count="count")
      .reset_index()
)
df = df.merge(type_stats, on="transaction_type", how="left")
df["type_amount_std"] = df["type_amount_std"].replace(0, np.nan)
df["type_amount_zscore"] = (df["amount"] - df["type_amount_mean"]) / df["type_amount_std"]

desc_stats = (
    df.groupby("description", dropna=False)["amount"]
      .agg(desc_amount_mean="mean", desc_amount_std="std", desc_tx_count="count")
      .reset_index()
)
df = df.merge(desc_stats, on="description", how="left")
df["desc_amount_std"] = df["desc_amount_std"].replace(0, np.nan)
df["desc_amount_zscore"] = (df["amount"] - df["desc_amount_mean"]) / df["desc_amount_std"]

df["sender_amount_percentile"] = (
    df.groupby("sender_id")["amount"].rank(pct=True, method="average")
)
df["pair_growth_ratio"] = np.where(
    df["pair_amount_mean"] > 0,
    df["amount"] / df["pair_amount_mean"],
    np.nan
)
df["near_threshold"] = df["amount"].apply(is_near_threshold)

df = df.sort_values(["sender_id", "recipient_id", "timestamp"]).reset_index(drop=True)
split_flags = {}

for (sender_id, recipient_id), g in df.groupby(["sender_id", "recipient_id"], sort=False):
    g = g.sort_values("timestamp")
    idxs = list(g.index)

    for i in range(len(idxs)):
        current_idx = idxs[i]
        current_row = df.loc[current_idx]
        similar_count = 1

        for j in range(i + 1, len(idxs)):
            next_idx = idxs[j]
            next_row = df.loc[next_idx]
            hours_diff = (next_row["timestamp"] - current_row["timestamp"]).total_seconds() / 3600.0
            if hours_diff > SPLIT_WINDOW_HOURS:
                break
            if abs(next_row["amount"] - current_row["amount"]) <= SPLIT_AMOUNT_DIFF:
                similar_count += 1

        if similar_count >= SPLIT_MIN_TX:
            split_flags[current_idx] = True

df["possible_split_payment"] = df.index.map(lambda x: split_flags.get(x, False))

balance_scores = []
balance_reasons = []

for _, row in df.iterrows():
    score = 0
    reasons = []

    if pd.notna(row["sender_amount_zscore"]) and row["sender_tx_count"] >= 5 and row["sender_amount_zscore"] >= 2.5:
        score += 2
        reasons.append(f"sender_amount_zscore:{row['sender_amount_zscore']:.2f}")

    if pd.notna(row["pair_amount_zscore"]) and row["pair_tx_count"] >= 3 and row["pair_amount_zscore"] >= 2.0:
        score += 3
        reasons.append(f"pair_amount_zscore:{row['pair_amount_zscore']:.2f}")

    if pd.notna(row["sender_amount_percentile"]) and row["sender_amount_percentile"] >= 0.98:
        score += 1
        reasons.append(f"sender_amount_percentile:{row['sender_amount_percentile']:.2f}")

    if pd.notna(row["amount_over_salary"]) and row["amount_over_salary"] >= 0.35:
        score += 2
        reasons.append(f"amount_over_salary:{row['amount_over_salary']:.2f}")

    if pd.notna(row["amount_over_total_funds"]) and row["amount_over_total_funds"] >= 0.45:
        score += 2
        reasons.append(f"amount_over_total_funds:{row['amount_over_total_funds']:.2f}")

    if pd.notna(row["amount_over_balance_after"]) and row["amount_over_balance_after"] >= 0.8:
        score += 2
        reasons.append(f"amount_over_balance_after:{row['amount_over_balance_after']:.2f}")

    if pd.notna(row["pair_growth_ratio"]) and row["pair_tx_count"] >= 3 and row["pair_growth_ratio"] >= 1.8:
        score += 2
        reasons.append(f"pair_growth_ratio:{row['pair_growth_ratio']:.2f}")

    if pd.notna(row["type_amount_zscore"]) and row["type_tx_count"] >= 5 and row["type_amount_zscore"] >= 2.5:
        score += 1
        reasons.append(f"type_amount_zscore:{row['type_amount_zscore']:.2f}")

    if pd.notna(row["desc_amount_zscore"]) and row["desc_tx_count"] >= 3 and row["desc_amount_zscore"] >= 2.5:
        score += 2
        reasons.append(f"description_amount_zscore:{row['desc_amount_zscore']:.2f}")

    if row["near_threshold"]:
        score += 1
        reasons.append("near_threshold_amount")

    if row["possible_split_payment"]:
        score += 2
        reasons.append("possible_split_payment")

    mitigation = 0
    if row["is_salary_like"]:
        mitigation += 3
        reasons.append("mitigation:salary_like")
    if row["is_rent_like"]:
        mitigation += 2
        reasons.append("mitigation:rent_like")
    if row["is_tax_like"]:
        mitigation += 2
        reasons.append("mitigation:tax_like")
    if row["is_subscription_like"]:
        mitigation += 1
        reasons.append("mitigation:subscription_like")

    final_score = max(score - mitigation, 0)
    balance_scores.append(final_score)
    balance_reasons.append(" | ".join(reasons) if reasons else "")

df["amount_balance_score"] = balance_scores
df["amount_balance_reasons"] = balance_reasons
df["suspicious_balance"] = df["amount_balance_score"].apply(
    lambda x: "yes" if x >= BALANCE_THRESHOLD else "false"
)

# Restore chronological order
df = df.sort_values("timestamp").reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n=== FILE CREATO ===")
print(OUTPUT_FILE)

print("\n=== DISTRIBUZIONE suspicious_identity_graph ===")
print(df["suspicious_identity_graph"].value_counts(dropna=False).to_string())

print("\n=== DISTRIBUZIONE suspicious_payment ===")
print(df["suspicious_payment"].value_counts(dropna=False).to_string())

print("\n=== DISTRIBUZIONE suspicious_semantic ===")
print(df["suspicious_semantic"].value_counts(dropna=False).to_string())

print("\n=== DISTRIBUZIONE suspicious_balance ===")
print(df["suspicious_balance"].value_counts(dropna=False).to_string())

print("\n=== TOP RIGHE SOSPETTE ===")
top = df[
    (df["suspicious_identity_graph"] == "yes") |
    (df["suspicious_payment"] == "yes") |
    (df["suspicious_semantic"] == "yes") |
    (df["suspicious_balance"] == "yes")
].copy()

if top.empty:
    print("Nessuna transazione marcata suspicious")
else:
    top["total_suspicious_flags"] = (
        (top["suspicious_identity_graph"] == "yes").astype(int) +
        (top["suspicious_payment"] == "yes").astype(int) +
        (top["suspicious_semantic"] == "yes").astype(int) +
        (top["suspicious_balance"] == "yes").astype(int)
    )

    print(
        top[
            [
                "transaction_id", "timestamp", "sender_id", "recipient_id", "amount",
                "suspicious_identity_graph", "suspicious_payment",
                "suspicious_semantic", "suspicious_balance",
                "total_suspicious_flags", "description"
            ]
        ]
        .sort_values(["total_suspicious_flags", "timestamp"], ascending=[False, True])
        .head(40)
        .to_string(index=False)
    )
