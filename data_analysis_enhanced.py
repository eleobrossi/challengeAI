"""
Fraud Detection with Statistical Analysis
uses transactions_with_stats.csv with enhanced statistical features
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Load enriched data with statistical features
print("Loading enriched transaction data with statistics...")
df = pd.read_csv("transactions_with_stats.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

print(f"Loaded {len(df)} transactions with {len(df.columns)} features\n")

# =====================================================
# STATISTICAL ANOMALY DETECTION
# =====================================================

print("Computing statistical anomalies...")

# 1. IQR-Based Outlier Detection
df["amount_numeric"] = pd.to_numeric(df["amount"], errors="coerce")
df["balance_numeric"] = pd.to_numeric(df["balance_after"], errors="coerce")
df["flag_iqr_outlier"] = df["amount_iqr_outlier"].fillna(False).astype(bool)

# 2. Rolling Statistics Deviation
df["sender_roll_mean"] = pd.to_numeric(df["sender_roll_mean"], errors="coerce")
df["sender_roll_std"] = pd.to_numeric(df["sender_roll_std"], errors="coerce")
df["sender_amount_stds_from_mean"] = (
    (df["amount_numeric"] - df["sender_roll_mean"]) / (df["sender_roll_std"] + 1e-9)
)
df["flag_rolling_deviation_sender"] = abs(df["sender_amount_stds_from_mean"]) > 2.5

df["recipient_roll_mean"] = pd.to_numeric(df["recipient_roll_mean"], errors="coerce")
df["recipient_roll_std"] = pd.to_numeric(df["recipient_roll_std"], errors="coerce")
df["recipient_amount_stds_from_mean"] = (
    (df["amount_numeric"] - df["recipient_roll_mean"]) / (df["recipient_roll_std"] + 1e-9)
)
df["flag_rolling_deviation_recipient"] = abs(df["recipient_amount_stds_from_mean"]) > 2.5

# 3. Mahalanobis Distance Anomaly
df["mahalanobis"] = pd.to_numeric(df["mahalanobis"], errors="coerce")
mahal_q95 = df["mahalanobis"].quantile(0.95)
df["flag_mahalanobis_outlier"] = df["mahalanobis"] > mahal_q95

# 4. Entropy-Based Pattern Anomaly
df["sender_desc_entropy"] = pd.to_numeric(df["sender_desc_entropy"], errors="coerce")
entropy_q90 = df["sender_desc_entropy"].quantile(0.90)
df["flag_high_entropy_descriptions"] = df["sender_desc_entropy"] > entropy_q90

# 5. KL Divergence Distribution Shift
df["kl_divergence"] = pd.to_numeric(df["kl_divergence"], errors="coerce")
kl_q90 = df["kl_divergence"].quantile(0.90)
df["flag_kl_divergence_high"] = df["kl_divergence"] > kl_q90

print("✓ Statistical features computed")

# =====================================================
# TRADITIONAL FRAUD DETECTION RULES
# =====================================================

print("Computing traditional fraud indicators...")

# 1. Amount Spike Detection
sender_stats = df.groupby("sender_name")["amount_numeric"].agg(["mean", "std"]).reset_index()
sender_stats.columns = ["sender_name", "amount_mean", "amount_std"]
df = df.merge(sender_stats, on="sender_name", how="left")
df["flag_amount_spike"] = df["amount_numeric"] > (df["amount_mean"] + 3 * df["amount_std"])

# 2. Unusual Time of Day
sender_common_hour = (
    df.groupby(["sender_name", "hour"])
    .size()
    .reset_index(name="count")
)
idx = sender_common_hour.groupby("sender_name")["count"].idxmax()
common_hours = sender_common_hour.loc[idx][["sender_name", "hour"]].rename(columns={"hour": "usual_hour"})
df = df.merge(common_hours, on="sender_name", how="left")
df["flag_abnormal_hour"] = abs(df["hour"] - df["usual_hour"]) > 3

# 3. Unusual Transfer Frequency
df["flag_rapid_small_transfers"] = False
for sender in df["sender_name"].unique():
    sender_median = df[df["sender_name"] == sender]["amount_numeric"].median()
    sender_df = df[df["sender_name"] == sender].copy()
    
    for idx in sender_df.index:
        current_time = df.loc[idx, "timestamp"]
        time_window = current_time - pd.Timedelta(hours=1)
        
        small_in_window = len(df[
            (df["sender_name"] == sender) &
            (df["timestamp"] >= time_window) &
            (df["timestamp"] <= current_time) &
            (df["amount_numeric"] < sender_median)
        ])
        
        if small_in_window > 3 and df.loc[idx, "amount_numeric"] < sender_median:
            df.loc[idx, "flag_rapid_small_transfers"] = True

# 4. Cross-city transfers (different from residence)
df["flag_unusual_source_city"] = df["transaction_location"] != df["sender_residence_city"]
df["flag_unusual_destination_city"] = df["transaction_location"] != df["recipient_residence_city"]

# 5. Large transfers to unknown parties
df["flag_large_to_unknown"] = (df["amount_numeric"] > df["amount_mean"]) & (df["recipient_name"] == "Unknown")

# 6. Duplicate pattern
dup_key = df.groupby(["sender_name", "recipient_name", "amount_numeric"]).size().reset_index(name="dup_count")
dup_key = dup_key[dup_key["dup_count"] > 1]
df = df.merge(dup_key[["sender_name", "recipient_name", "amount_numeric", "dup_count"]], 
              on=["sender_name", "recipient_name", "amount_numeric"], how="left", suffixes=("", "_dup"))
df["flag_duplicate_pattern"] = df["dup_count"] > 1
df = df.drop(columns=["dup_count"], errors="ignore")

print("✓ Traditional indicators computed")

# =====================================================
# COMBINED FRAUD SCORE
# =====================================================

df["fraud_score"] = (
    df["flag_amount_spike"].astype(int)
    + df["flag_abnormal_hour"].astype(int)
    + df["flag_rapid_small_transfers"].astype(int)
    + df["flag_unusual_source_city"].astype(int) * 0.2
    + df["flag_unusual_destination_city"].astype(int) * 0.2
    + df["flag_large_to_unknown"].astype(int)
    + df["flag_duplicate_pattern"].astype(int)
    # STATISTICAL FEATURES WITH WEIGHTS
    + (df["flag_iqr_outlier"].astype(int) * 2)
    + (df["flag_rolling_deviation_sender"].astype(int) * 2)
    + (df["flag_rolling_deviation_recipient"].astype(int) * 2)
    + (df["flag_mahalanobis_outlier"].astype(int) * 3)
    + (df["flag_kl_divergence_high"].astype(int) * 2)
    + (df["flag_high_entropy_descriptions"].astype(int))
)

# =====================================================
# FRAUD REASONS
# =====================================================

fraud_reasons = []
for idx, row in df.iterrows():
    reasons = []
    if row["flag_amount_spike"]:
        reasons.append("Amount Spike")
    if row["flag_abnormal_hour"]:
        reasons.append("Abnormal Hour")
    if row["flag_rapid_small_transfers"]:
        reasons.append("Rapid Small Transfers")
    if row["flag_unusual_source_city"]:
        reasons.append("Unusual Source City")
    if row["flag_unusual_destination_city"]:
        reasons.append("Unusual Destination City")
    if row["flag_large_to_unknown"]:
        reasons.append("Large to Unknown")
    if row["flag_duplicate_pattern"]:
        reasons.append("Duplicate Pattern")
    # Statistical
    if row["flag_iqr_outlier"]:
        reasons.append("IQR Outlier [STAT]")
    if row["flag_rolling_deviation_sender"]:
        reasons.append("Sender Deviation [STAT]")
    if row["flag_rolling_deviation_recipient"]:
        reasons.append("Recipient Deviation [STAT]")
    if row["flag_mahalanobis_outlier"]:
        reasons.append("Multivariate Anomaly [STAT]")
    if row["flag_kl_divergence_high"]:
        reasons.append("Distribution Shift [STAT]")
    if row["flag_high_entropy_descriptions"]:
        reasons.append("Unusual Patterns [STAT]")
    
    fraud_reasons.append(" | ".join(reasons) if reasons else "None")

df["fraud_reason"] = fraud_reasons
df["flag_suspicious"] = df["fraud_score"] >= 2

# =====================================================
# OUTPUT
# =====================================================

# Save results
df.to_csv("transactions_flagged.csv", index=False)

print("\n" + "="*80)
print("FRAUD DETECTION COMPLETE (Enhanced with Statistical Analysis)")
print("="*80)

# Show samples
print("\nTop 10 Suspicious Transactions:")
suspicious = df[df["flag_suspicious"]].nlargest(10, "fraud_score")[
    ["transaction_id", "sender_name", "recipient_name", "amount", "fraud_score", "fraud_reason"]
]
print(suspicious.to_string())

# Calculate statistics
fraud_count = df["flag_suspicious"].sum()
total_count = len(df)
fraud_percentage = (fraud_count / total_count) * 100

print(f"\n--- SUMMARY ---")
print(f"Total transactions: {total_count}")
print(f"Flagged as suspicious: {fraud_count}")
print(f"Detection rate: {fraud_percentage:.2f}%")

print(f"\n--- Statistical Feature Breakdown ---")
print(f"IQR Outliers: {df['flag_iqr_outlier'].sum()}")
print(f"Sender Rolling Deviations: {df['flag_rolling_deviation_sender'].sum()}")
print(f"Recipient Rolling Deviations: {df['flag_rolling_deviation_recipient'].sum()}")
print(f"Mahalanobis Outliers: {df['flag_mahalanobis_outlier'].sum()}")
print(f"High KL Divergence: {df['flag_kl_divergence_high'].sum()}")
print(f"High Entropy: {df['flag_high_entropy_descriptions'].sum()}")

print(f"\n--- Traditional Feature Breakdown ---")
print(f"Amount Spikes: {df['flag_amount_spike'].sum()}")
print(f"Abnormal Hours: {df['flag_abnormal_hour'].sum()}")
print(f"Rapid Small Transfers: {df['flag_rapid_small_transfers'].sum()}")
print(f"Unusual Source City: {df['flag_unusual_source_city'].sum()}")
print(f"Unusual Destination City: {df['flag_unusual_destination_city'].sum()}")
print(f"Large to Unknown: {df['flag_large_to_unknown'].sum()}")
print(f"Duplicate Patterns: {df['flag_duplicate_pattern'].sum()}")

print(f"\nFraud Score Distribution:")
print(df["fraud_score"].describe())

# Save JSON with details
print(f"\n✓ Saving detailed fraud stack traces...")
flagged = df[df["flag_suspicious"]].copy()

fraud_details = []
for idx, row in flagged.iterrows():
    detail = {
        "transaction_id": row["transaction_id"],
        "fraud_score": int(row["fraud_score"]),
        "sender": row["sender_name"],
        "sender_job": row["sender_job"],
        "recipient": row["recipient_name"],
        "amount": float(row["amount"]),
        "timestamp": row["timestamp"].isoformat(),
        "source_city": row["sender_residence_city"],
        "dest_city": row["recipient_residence_city"],
        "transaction_city": row["transaction_location"],
        "fraud_reasons": row["fraud_reason"],
        "statistical_evidence": {
            "iqr_outlier": bool(row["flag_iqr_outlier"]),
            "sender_deviation_stds": float(row["sender_amount_stds_from_mean"]) if pd.notna(row["sender_amount_stds_from_mean"]) else None,
            "recipient_deviation_stds": float(row["recipient_amount_stds_from_mean"]) if pd.notna(row["recipient_amount_stds_from_mean"]) else None,
            "mahalanobis_distance": float(row["mahalanobis"]) if pd.notna(row["mahalanobis"]) else None,
            "kl_divergence": float(row["kl_divergence"]) if pd.notna(row["kl_divergence"]) else None,
            "description_entropy": float(row["sender_desc_entropy"]) if pd.notna(row["sender_desc_entropy"]) else None,
        },
        "rolling_stats": {
            "sender_roll_mean": float(row["sender_roll_mean"]) if pd.notna(row["sender_roll_mean"]) else None,
            "sender_roll_std": float(row["sender_roll_std"]) if pd.notna(row["sender_roll_std"]) else None,
            "recipient_roll_mean": float(row["recipient_roll_mean"]) if pd.notna(row["recipient_roll_mean"]) else None,
            "recipient_roll_std": float(row["recipient_roll_std"]) if pd.notna(row["recipient_roll_std"]) else None,
        }
    }
    fraud_details.append(detail)

with open("fraud_analysis_detailed.json", "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.now().isoformat(),
        "total_flagged": len(fraud_details),
        "detection_rate": f"{fraud_percentage:.2f}%",
        "flagged_transactions": fraud_details
    }, f, indent=2, ensure_ascii=False)

print("✓ Results saved to:")
print("  - transactions_flagged.csv")
print("  - fraud_analysis_detailed.json")
print("\n" + "="*80)
