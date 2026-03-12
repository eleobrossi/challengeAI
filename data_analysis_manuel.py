import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
# df = pd.read_csv("The_Truman_Show_train\\public\\transactions.csv")
df = pd.read_csv("transactions_with_stats.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract hour
df["hour"] = df["timestamp"].dt.hour


# -----------------------------
# 1. Amount Spike Detection
# -----------------------------
sender_stats = df.groupby("sender_id")["amount"].agg(["mean", "std"]).reset_index()
sender_stats.columns = ["sender_id", "amount_mean", "amount_std"]

df = df.merge(sender_stats, on="sender_id", how="left")

# Flag if amount > mean + 3*std
df["flag_amount_spike"] = df["amount"] > (df["amount_mean"] + 3 * df["amount_std"])


# -----------------------------
# 2. Unusual Transaction Type
# -----------------------------
# Most common transaction type per sender
sender_common_type = (
    df.groupby(["sender_id", "transaction_type"])
    .size()
    .reset_index(name="count")
)

idx = sender_common_type.groupby("sender_id")["count"].idxmax()
common_types = sender_common_type.loc[idx][["sender_id", "transaction_type"]]
common_types.columns = ["sender_id", "usual_transaction_type"]

df = df.merge(common_types, on="sender_id", how="left")

df["flag_unusual_transaction_type"] = (
    df["transaction_type"] != df["usual_transaction_type"]
)


# -----------------------------
# 3. New Payment Method
# -----------------------------
# First payment method used by sender
first_payment_method = (
    df.sort_values("timestamp")
      .groupby("sender_id")["payment_method"]
      .first()
      .reset_index()
)

first_payment_method.columns = ["sender_id", "usual_payment_method"]

df = df.merge(first_payment_method, on="sender_id", how="left")

# df["flag_new_payment_method"] = (
#     df["payment_method"] != df["usual_payment_method"]
# )


# -----------------------------
# 4. Abnormal Time of Day
# -----------------------------
# Calculate mean hour per sender
sender_mean_hour = df.groupby("sender_id")["hour"].mean().reset_index()
sender_mean_hour.columns = ["sender_id", "mean_hour"]

df = df.merge(sender_mean_hour, on="sender_id", how="left")

# Flag if transaction happens outside ±10 hours from mean
df["flag_abnormal_hour"] = (
    abs(df["hour"] - df["mean_hour"]) > 10
)


# -----------------------------
# 5. System Manipulation Detection
# Balance Math Validation
# -----------------------------
# Sort by sender and time for sequential analysis
df_sorted = df.sort_values(["timestamp","sender_id"]).reset_index(drop=True)

df_sorted["flag_balance_mismatch"] = False
df_sorted["flag_sequential_inconsistency"] = False

# Check balance math for each sender
# Note: balance_after represents the sender's account balance after the transaction
# All transaction types (bank transfer, in-person payment, e-commerce, direct debit, withdrawal) 
# are outgoing from the sender's perspective, so balance should decrease by the amount
for sender_id in df_sorted["sender_id"].unique():
    sender_txns = df_sorted[df_sorted["sender_id"] == sender_id].index
    
    for i in sender_txns:
        if i == sender_txns[0]:
            # First transaction - cannot validate against previous
            continue
        
        current_txn = df_sorted.loc[i]
        previous_txn = df_sorted.loc[i - 1]
        
        # For sender: balance_after = previous_balance - amount (all transactions are outgoing)
        expected_balance = previous_txn["balance_after"] - current_txn["amount"]
        
        # Round to 2 decimal places to avoid floating point precision errors
        expected_balance_rounded = round(expected_balance, 2)
        actual_balance_rounded = round(current_txn["balance_after"], 2)
        
        # Flag if balance math doesn't match (allow 0.005 for rounding, which rounds to 0.01)
        if abs(expected_balance_rounded - actual_balance_rounded) > 0.005:
            df_sorted.loc[i, "flag_balance_mismatch"] = True
        
        # Flag sequential inconsistencies (balance increased unexpectedly for outgoing txns)
        if current_txn["balance_after"] > previous_txn["balance_after"]:
            df_sorted.loc[i, "flag_sequential_inconsistency"] = True

# Update original dataframe with flags
df = df_sorted.copy()


# -----------------------------
# 6. Duplicate Transactions
# (sender_id + recipient_id + amount - Many identical amounts)
# -----------------------------
# Count transactions with same sender, recipient, and amount
duplicate_key = df.groupby(["sender_id", "recipient_id", "amount"]).size().reset_index(name="count")
duplicate_key = duplicate_key[duplicate_key["count"] > 1]

df = df.merge(
    duplicate_key[["sender_id", "recipient_id", "amount", "count"]],
    on=["sender_id", "recipient_id", "amount"],
    how="left"
)

df["flag_duplicate_transactions"] = df["count"] > 1
df = df.drop(columns=["count"])


# -----------------------------
# 7. Rapid Small Transfers
# (sender_id + timestamp - Many small transfers quickly)
# -----------------------------
# Identify small transfers (below median for each sender)
sender_median_amount = df.groupby("sender_id")["amount"].median().reset_index()
sender_median_amount.columns = ["sender_id", "median_amount"]

df = df.merge(sender_median_amount, on="sender_id", how="left")

# Flag as small transfer if below median
df["is_small_transfer"] = df["amount"] < df["median_amount"]

# Count small transfers per sender within 1-hour windows
df["rapid_transfer_count"] = 0
for sender_id in df["sender_id"].unique():
    sender_df = df[df["sender_id"] == sender_id].copy()
    
    for idx in sender_df.index:
        current_time = df.loc[idx, "timestamp"]
        time_window_start = current_time - pd.Timedelta(hours=1)
        
        # Count small transfers from this sender in the 1-hour window
        count = len(df[
            (df["sender_id"] == sender_id) &
            (df["timestamp"] >= time_window_start) &
            (df["timestamp"] <= current_time) &
            (df["is_small_transfer"] == True)
        ])
        
        df.loc[idx, "rapid_transfer_count"] = count

# Flag if more than 3 small transfers in 1 hour
df["flag_rapid_small_transfers"] = (df["rapid_transfer_count"] > 3) & (df["is_small_transfer"])

df = df.drop(columns=["is_small_transfer", "rapid_transfer_count", "median_amount"])


# -----------------------------
# 8. Amounts Near Reporting Threshold
# (e.g., €9,900 repeatedly - amounts just below common thresholds)
# -----------------------------
# Common reporting thresholds: 10,000, 5,000, 1,000
thresholds = [10000, 5000, 1000]
threshold_buffer = 500  # Flag amounts within €500 of threshold

def is_near_threshold(amount):
    for threshold in thresholds:
        if threshold - threshold_buffer < amount < threshold:
            return True
    return False

df["flag_reporting_threshold"] = df["amount"].apply(is_near_threshold)


# -----------------------------
# 9. New or Rare Location
# (sender_id + location - New or rare location)
# -----------------------------
# Count occurrences of each location per sender
sender_location_freq = df[df["transaction_location"].notna()].groupby(["sender_id", "transaction_location"]).size().reset_index(name="location_count")
sender_location_freq["is_rare"] = sender_location_freq["location_count"] <= 1

df = df.merge(
    sender_location_freq[["sender_id", "transaction_location", "is_rare"]],
    on=["sender_id", "transaction_location"],
    how="left"
)

df["flag_new_rare_location"] = df["is_rare"] == True
df = df.drop(columns=["is_rare"])


# -----------------------------
# 10. Impossible Travel Detection
# (location + timestamp - Impossible travel: Rome → Tokyo in 1h)
# -----------------------------
# City coordinates (latitude, longitude)
city_coordinates = {
    "Rome": (41.9028, 12.4964),
    "Tokyo": (35.6762, 139.6503),
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Milan": (45.4642, 9.1900),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3874, 2.1686),
    "Amsterdam": (52.3676, 4.9041),
    "Vienna": (48.2082, 16.3738),
    "Dublin": (53.3498, -6.2603),
    "Prague": (50.0755, 14.4378),
    "Moscow": (55.7558, 37.6173),
    "Istanbul": (41.0082, 28.9784),
    "Dubai": (25.2048, 55.2708),
    "Hong Kong": (22.3193, 114.1694),
    "Singapore": (1.3521, 103.8198),
    "Sydney": (33.8688, 151.2093),
    "Los Angeles": (34.0522, -118.2437),
}

from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Max travel speed: 900 km/h (approximately speed of commercial flights)
max_travel_speed = 900

df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
df["flag_impossible_travel"] = False

for i in range(1, len(df)):
    current_row = df.loc[i]
    previous_row = df.loc[i-1]
    
    # Only check if same sender and both have valid locations
    if (current_row["sender_id"] == previous_row["sender_id"] and
        pd.notna(current_row["transaction_location"]) and pd.notna(previous_row["transaction_location"]) and
        current_row["transaction_location"] in city_coordinates and previous_row["transaction_location"] in city_coordinates):
        
        lat1, lon1 = city_coordinates[previous_row["transaction_location"]]
        lat2, lon2 = city_coordinates[current_row["transaction_location"]]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Time difference in hours
        time_diff = (current_row["timestamp"] - previous_row["timestamp"]).total_seconds() / 3600
        
        # Flag if impossible to travel this distance in the given time
        if time_diff > 0 and distance / time_diff > max_travel_speed:
            df.loc[i, "flag_impossible_travel"] = True


# -----------------------------
# 11. Unusual Location + Payment Method Combination
# (sender_id + location + payment_method - Card used in unusual place)
# -----------------------------
# For each sender, find common payment methods used in each location
sender_loc_payment = df[df["transaction_location"].notna() & df["payment_method"].notna()].groupby(
    ["sender_id", "transaction_location", "payment_method"]
).size().reset_index(name="count")

# Get the most common payment method per sender/location
sender_loc_payment_sorted = sender_loc_payment.sort_values(["sender_id", "transaction_location", "count"], ascending=[True, True, False])
usual_payment = sender_loc_payment_sorted.drop_duplicates(["sender_id", "transaction_location"], keep="first")[["sender_id", "transaction_location", "payment_method"]]
usual_payment.columns = ["sender_id", "transaction_location", "usual_payment_method"]

df = df.merge(
    usual_payment,
    on=["sender_id", "transaction_location"],
    how="left"
)

# Ensure the column exists (in case merge didn't create it)
if "usual_payment_method" not in df.columns:
    df["usual_payment_method"] = None

df["flag_unusual_location_payment"] = (
    df["payment_method"].notna() & 
    df["usual_payment_method"].notna() & 
    (df["payment_method"] != df["usual_payment_method"])
)


# -----------------------------
# 12. Amount IQR Outlier Detection
# (Interquartile Range based outlier detection)
# -----------------------------
df["flag_amount_iqr_outlier"] = False

for sender_id in df["sender_id"].unique():
    sender_amounts = df[df["sender_id"] == sender_id]["amount"]
    Q1 = sender_amounts.quantile(0.25)
    Q3 = sender_amounts.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df.loc[
        (df["sender_id"] == sender_id) & ((df["amount"] < lower_bound) | (df["amount"] > upper_bound)),
        "flag_amount_iqr_outlier"
    ] = True


# -----------------------------
# 13. Mahalanobis Distance (Multivariate Outlier Detection)
# (Combines amount, hour, payment_method encoded)
# -----------------------------
df["flag_mahalanobis_outlier"] = False

# Encode categorical features
from concurrent.futures import ThreadPoolExecutor
payment_method_encoded = pd.factorize(df["payment_method"].fillna("unknown"))[0]
transaction_type_encoded = pd.factorize(df["transaction_type"].fillna("unknown"))[0]

# Prepare features for Mahalanobis
features = np.column_stack([
    df["amount"],
    df["hour"],
    payment_method_encoded,
    transaction_type_encoded
])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Calculate Mahalanobis distance per sender
for sender_id in df["sender_id"].unique():
    sender_mask = df["sender_id"] == sender_id
    sender_features = features_scaled[sender_mask]
    
    if len(sender_features) > 1:
        mean = np.mean(sender_features, axis=0)
        try:
            cov = np.cov(sender_features.T)
            # Add small regularization to avoid singular matrix
            cov += np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov)
            
            # Calculate Mahalanobis distance for each transaction
            distances = []
            for i in range(len(sender_features)):
                diff = sender_features[i] - mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                distances.append(mahal_dist)
            
            # Flag if distance > 3 standard deviations
            distances = np.array(distances)
            threshold = np.mean(distances) + 3 * np.std(distances)
            
            sender_indices = df[sender_mask].index
            for idx, dist in zip(sender_indices, distances):
                if dist > threshold:
                    df.loc[idx, "flag_mahalanobis_outlier"] = True
        except:
            pass  # Skip if covariance matrix is singular



# -----------------------------
# 15. KL Divergence (Distribution Shift Detection)
# (Compare recent transactions to historical baseline)
# -----------------------------
df["flag_kl_divergence"] = False

# Use first 60% of sender's transactions as baseline
for sender_id in df["sender_id"].unique():
    sender_transactions = df[df["sender_id"] == sender_id].sort_values("timestamp").reset_index(drop=True)
    
    if len(sender_transactions) >= 10:  # Need at least 10 transactions
        split_point = int(len(sender_transactions) * 0.6)
        
        baseline = sender_transactions.iloc[:split_point]
        recent = sender_transactions.iloc[split_point:]
        
        # Create amount distribution bins
        all_amounts = pd.concat([baseline["amount"], recent["amount"]])
        bins = np.histogram_bin_edges(all_amounts, bins=10)
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline["amount"], bins=bins)
        baseline_dist = baseline_dist / baseline_dist.sum()
        
        recent_dist, _ = np.histogram(recent["amount"], bins=bins)
        recent_dist = recent_dist / recent_dist.sum()
        
        # Add small epsilon to avoid log(0)
        baseline_dist = baseline_dist + 1e-10
        recent_dist = recent_dist + 1e-10
        
        # Calculate KL divergence
        kl_div = np.sum(recent_dist * (np.log(recent_dist) - np.log(baseline_dist)))
        
        # Flag if KL divergence > 0.5 (significant distribution shift)
        if kl_div > 0.5:
            df.loc[df["sender_id"] == sender_id, "flag_kl_divergence"] = True


# -----------------------------
# 16. Velocity Checks
# (Transaction frequency spikes - rapid/burst activity)
# -----------------------------
df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
df["flag_velocity_spike_10min"] = False
df["flag_velocity_spike_1hour"] = False
df["flag_velocity_spike_1day"] = False

for sender_id in df["sender_id"].unique():
    sender_txns = df[df["sender_id"] == sender_id].index
    
    for idx in sender_txns:
        current_time = df.loc[idx, "timestamp"]
        
        # 10-minute window: flag if >5 transactions
        window_10min = df[
            (df["sender_id"] == sender_id) &
            (df["timestamp"] >= current_time - pd.Timedelta(minutes=10)) &
            (df["timestamp"] <= current_time)
        ]
        if len(window_10min) > 5:
            df.loc[idx, "flag_velocity_spike_10min"] = True
        
        # 1-hour window: flag if >10 transactions
        window_1hour = df[
            (df["sender_id"] == sender_id) &
            (df["timestamp"] >= current_time - pd.Timedelta(hours=1)) &
            (df["timestamp"] <= current_time)
        ]
        if len(window_1hour) > 10:
            df.loc[idx, "flag_velocity_spike_1hour"] = True
        
        # 1-day window: flag if >20 transactions
        window_1day = df[
            (df["sender_id"] == sender_id) &
            (df["timestamp"] >= current_time - pd.Timedelta(days=1)) &
            (df["timestamp"] <= current_time)
        ]
        if len(window_1day) > 20:
            df.loc[idx, "flag_velocity_spike_1day"] = True

# Combine velocity flags into one
df["flag_velocity_spike"] = (
    df["flag_velocity_spike_10min"] | 
    df["flag_velocity_spike_1hour"] | 
    df["flag_velocity_spike_1day"]
)
df = df.drop(columns=["flag_velocity_spike_10min", "flag_velocity_spike_1hour", "flag_velocity_spike_1day"])


# -----------------------------
# 17. Round Number Detection
# (Structuring: exact round amounts like €1000, €5000, €100)
# -----------------------------
df["flag_round_number_amount"] = False

round_amounts = [100, 500, 1000, 2000, 5000, 10000]

for amount in round_amounts:
    df.loc[df["amount"] == amount, "flag_round_number_amount"] = True


# Combined Fraud Score with Weighted System
# Weight assignments based on reliability and severity:
# - CRITICAL (3.0): Direct evidence of manipulation/impossibility
# - HIGH (2.0): Strong statistical indicators
# - MEDIUM (1.5): Behavioral anomalies
# - LOW (1.0): Weak individual indicators

df["fraud_score"] = (
    # CRITICAL WEIGHT (3.0) - Direct evidence of manipulation or physical impossibility
    df["flag_balance_mismatch"].astype(float) * 3.0           # Account math doesn't add up - manipulation
    + df["flag_sequential_inconsistency"].astype(float) * 3.0  # Balance increased on outgoing - impossible
    + df["flag_impossible_travel"].astype(float) * 3.0         # Physical impossibility - strong evidence
    + df["flag_duplicate_transactions"].astype(float) * 3.0    # Exact duplicate - suspicious pattern
    
    # HIGH WEIGHT (2.0) - Strong statistical/behavioral indicators
    + df["flag_amount_spike"].astype(float) * 2.0              # Significant deviation from normal
    + df["flag_amount_iqr_outlier"].astype(float) * 2.0        # Outlier by quartile analysis
    + df["flag_mahalanobis_outlier"].astype(float) * 2.0       # Multivariate anomaly
    + df["flag_reporting_threshold"].astype(float) * 2.0       # Classic structuring pattern
    + df["flag_rapid_small_transfers"].astype(float) * 2.0     # Smurfing indicator
    + df["flag_kl_divergence"].astype(float) * 2.0             # Distribution shift (account compromise)
    + df["flag_velocity_spike"].astype(float) * 2.0            # Velocity spike - rapid transaction burst
    + df["flag_round_number_amount"].astype(float) * 2.0       # Round amounts - classic structuring
    
    # MEDIUM WEIGHT (1.5) - Behavioral anomalies
    + df["flag_new_rare_location"].astype(float) * 1.5         # New location (could be legitimate travel)
    + df["flag_unusual_location_payment"].astype(float) * 1.5  # Unusual payment method combo
    
    # LOW WEIGHT (1.0) - Weak individual indicators
    + df["flag_unusual_transaction_type"].astype(float) * 1.0  # People change transaction types
    + df["flag_abnormal_hour"].astype(float) * 1.0             # Schedules vary legitimately
)
# Create fraud reason column
fraud_reasons = []
for idx, row in df.iterrows():
    reasons = []
    if row["flag_amount_spike"]:
        reasons.append("Amount Spike")
    if row["flag_unusual_transaction_type"]:
        reasons.append("Unusual Transaction Type")
    # if row["flag_new_payment_method"]:
    #     reasons.append("New Payment Method")
    if row["flag_abnormal_hour"]:
        reasons.append("Abnormal Time of Day")
    if row["flag_balance_mismatch"]:
        reasons.append("Balance Math Mismatch")
    if row["flag_sequential_inconsistency"]:
        reasons.append("Sequential Inconsistency")
    if row["flag_duplicate_transactions"]:
        reasons.append("Duplicate Transactions")
    if row["flag_rapid_small_transfers"]:
        reasons.append("Rapid Small Transfers")
    if row["flag_reporting_threshold"]:
        reasons.append("Reporting Threshold Amount")
    if row["flag_new_rare_location"]:
        reasons.append("New/Rare Location")
    if row["flag_impossible_travel"]:
        reasons.append("Impossible Travel")
    if row["flag_unusual_location_payment"]:
        reasons.append("Unusual Location-Payment Combo")
    if row["flag_amount_iqr_outlier"]:
        reasons.append("Amount IQR Outlier")
    if row["flag_mahalanobis_outlier"]:
        reasons.append("Mahalanobis Outlier")
    if row["flag_kl_divergence"]:
        reasons.append("KL Divergence Distribution Shift")
    if row["flag_velocity_spike"]:
        reasons.append("Velocity Spike")
    if row["flag_round_number_amount"]:
        reasons.append("Round Number Amount")
    
    fraud_reasons.append(" | ".join(reasons) if reasons else "None")

df["fraud_reason"] = fraud_reasons
# Flag as suspicious if weighted score >= 6
# Score scale: ~30 max (4 critical @ 3 + 6 high @ 2 + 3 medium @ 1.5 + 2 low @ 1)
# Threshold of 6 = approximately 2 critical flags or equivalent combination
df["flag_suspicious"] = df["fraud_score"] >= 6

# Drop intermediate column
if "usual_payment_method" in df.columns:
    df = df.drop(columns=["usual_payment_method"])

# Save results
df.to_csv("transactions_flagged.csv", index=False)

print("Fraud detection complete.")
print(df[["transaction_id", "fraud_score", "fraud_reason", "flag_suspicious"]].head())

# Calculate and print percentage of frauds
fraud_count = df["flag_suspicious"].sum()
total_count = len(df)
fraud_percentage = (fraud_count / total_count) * 100

print(f"\nFraud Summary:")
print(f"Total transactions: {total_count}")
print(f"Fraudulent transactions (threshold=6): {fraud_count}")
print(f"Fraud percentage: {fraud_percentage:.2f}%")

# Threshold Variance Analysis
print(f"\nThreshold Variance Analysis (Weighted Scoring):")
print(f"{'Threshold':<12} {'Flagged':<12} {'Percentage':<12} {'Interpretation':<30}")
print("-" * 66)
thresholds_with_labels = [
    (2, "Very Permissive"),
    (4, "Permissive"),
    (6, "Recommended"),
    (8, "Moderate"),
    (10, "Conservative"),
    (12, "Very Conservative")
]
for threshold, label in thresholds_with_labels:
    flagged_count = (df["fraud_score"] >= threshold).sum()
    pct = (flagged_count / total_count) * 100
    print(f"{threshold:<12} {flagged_count:<12} {pct:>10.2f}%     {label:<30}")

# Validation Metrics & Detection Analysis
print(f"\n" + "="*70)
print(f"VALIDATION METRICS & FRAUD FLAG ANALYSIS")
print(f"="*70)

# Fraud Score Distribution
print(f"\nFraud Score Distribution:")
print(f"{'Score Range':<20} {'Count':<12} {'Percentage':<12}")
print("-" * 44)
score_ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 15), (15, 35)]
for low, high in score_ranges:
    count = ((df["fraud_score"] >= low) & (df["fraud_score"] < high)).sum()
    pct = (count / total_count) * 100
    print(f"{low}-{high}:<20 {count:<12} {pct:>10.2f}%")

# Flag Detection Frequency (most common flags)
print(f"\nMost Common Fraud Flags:")
flag_columns = [
    col for col in df.columns if col.startswith("flag_")
    and col not in ["flag_suspicious"]
]
flag_counts = {}
for flag_col in flag_columns:
    flag_counts[flag_col.replace("flag_", "")] = df[flag_col].sum()

# Sort by frequency
sorted_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
print(f"{'Flag Name':<35} {'Count':<12} {'Percentage':<12}")
print("-" * 59)
for flag_name, count in sorted_flags:
    pct = (count / total_count) * 100
    print(f"{flag_name:<35} {count:<12} {pct:>10.2f}%")

# Flag Co-occurrence Analysis (correlations)
print(f"\nTop Flag Combinations (for flagged transactions):")
flagged_df = df[df["flag_suspicious"] == True]
if len(flagged_df) > 0:
    flag_sum_per_tx = flagged_df[[col for col in flag_columns]].sum(axis=1)
    flag_combo_dist = flag_sum_per_tx.value_counts().sort_index()
    print(f"{'Flags/Transaction':<20} {'Flagged Txns':<15} {'Percentage':<15}")
    print("-" * 50)
    for num_flags, count in flag_combo_dist.items():
        pct = (count / len(flagged_df)) * 100
        print(f"{int(num_flags):<20} {count:<15} {pct:>12.2f}%")
else:
    print("No transactions flagged at current threshold.")

print(f"\n" + "="*70)


# Create detailed JSON with stack traces for fraudulent transactions
import json
from datetime import datetime

# Helper function to convert non-serializable types to JSON-compatible types
def convert_to_serializable(obj):
    """Convert pandas/numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy/pandas scalar types
        return obj.item()
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj

fraud_stack_traces = []

# Filter fraudulent transactions
fraudulent_transactions = df[df["flag_suspicious"] == True].copy()

for idx, row in fraudulent_transactions.iterrows():
    stack_trace = {
        "transaction_id": row["transaction_id"],
        "fraud_score": int(row["fraud_score"]),
        "sender_id": row["sender_id"],
        "recipient_id": row["recipient_id"],
        "amount": float(row["amount"]),
        "timestamp": row["timestamp"].isoformat(),
        "location": row["transaction_location"] if pd.notna(row["transaction_location"]) else None,
        "payment_method": row["payment_method"] if pd.notna(row["payment_method"]) else None,
        "fraud_flags": []
    }
    
    # 1. Amount Spike
    if row["flag_amount_spike"]:
        stack_trace["fraud_flags"].append({
            "flag": "Amount Spike",
            "description": "Transaction amount significantly exceeds sender's typical pattern",
            "evidence": {
                "sender_mean_amount": float(row["amount_mean"]),
                "sender_std_amount": float(row["amount_std"]),
                "threshold_calculation": f"mean + 3*std = {float(row['amount_mean']):.2f} + 3*{float(row['amount_std']):.2f}",
                "threshold": float(row["amount_mean"] + 3 * row["amount_std"]),
                "transaction_amount": float(row["amount"]),
                "exceeds_threshold_by": float(row["amount"] - (row["amount_mean"] + 3 * row["amount_std"])),
                "variance": f"{(row['amount'] / row['amount_mean']):.2f}x the sender's average"
            }
        })
    
    # 2. Unusual Transaction Type
    if row["flag_unusual_transaction_type"]:
        stack_trace["fraud_flags"].append({
            "flag": "Unusual Transaction Type",
            "description": "Sender using unfamiliar transaction type",
            "evidence": {
                "usual_type": row["usual_transaction_type"],
                "used_type": row["transaction_type"],
                "mismatch": f"Expected {row['usual_transaction_type']}, used {row['transaction_type']}"
            }
        })
    
    # 3. Abnormal Hour
    if row["flag_abnormal_hour"]:
        stack_trace["fraud_flags"].append({
            "flag": "Abnormal Time of Day",
            "description": "Transaction at unusual hour for this sender",
            "evidence": {
                "sender_mean_hour": round(float(row["mean_hour"]), 2),
                "transaction_hour": int(row["hour"]),
                "hour_difference": round(abs(row["hour"] - row["mean_hour"]), 2),
                "time_deviation": f"{round(abs(row['hour'] - row['mean_hour']), 2)} hours away from sender's mean transaction time",
                "threshold": "More than 10 hours"
            }
        })
    
    # 4. Balance Math Mismatch
    if row["flag_balance_mismatch"]:
        # Find previous transaction for this sender
        sender_txns = df[df["sender_id"] == row["sender_id"]].sort_values("timestamp")
        current_idx_in_sorted = sender_txns[sender_txns["transaction_id"] == row["transaction_id"]].index[0]
        
        # Get the previous transaction
        current_position = sender_txns.index.get_loc(current_idx_in_sorted)
        if current_position > 0:
            prev_txn = sender_txns.iloc[current_position - 1]
            expected_balance = round(float(prev_txn["balance_after"] - row["amount"]), 2)
            actual_balance = round(float(row["balance_after"]), 2)
            discrepancy = round(actual_balance - expected_balance, 2)
            
            stack_trace["fraud_flags"].append({
                "flag": "Balance Math Mismatch",
                "description": "Account balance does not match expected value after transaction",
                "evidence": {
                    "previous_balance": round(float(prev_txn["balance_after"]), 2),
                    "transaction_amount": round(float(row["amount"]), 2),
                    "expected_calculation": f"{round(float(prev_txn['balance_after']), 2):.2f} - {round(float(row['amount']), 2):.2f}",
                    "expected_balance": expected_balance,
                    "reported_balance": actual_balance,
                    "discrepancy": discrepancy,
                    "formula": f"{actual_balance:.2f} - {expected_balance:.2f} = {discrepancy:.2f}",
                    "error_percentage": f"{abs(discrepancy/expected_balance)*100:.2f}%" if expected_balance != 0 else "N/A"
                }
            })
        else:
            stack_trace["fraud_flags"].append({
                "flag": "Balance Math Mismatch",
                "description": "Account balance does not match expected value after transaction",
                "note": "Indicates potential system manipulation or data tampering"
            })
    
    # 5. Sequential Inconsistency
    if row["flag_sequential_inconsistency"]:
        # Find previous transaction for this sender
        sender_txns = df[df["sender_id"] == row["sender_id"]].sort_values("timestamp")
        current_idx_in_sorted = sender_txns[sender_txns["transaction_id"] == row["transaction_id"]].index[0]
        
        current_position = sender_txns.index.get_loc(current_idx_in_sorted)
        if current_position > 0:
            prev_txn = sender_txns.iloc[current_position - 1]
            prev_balance = round(float(prev_txn["balance_after"]), 2)
            curr_balance = round(float(row["balance_after"]), 2)
            balance_change = round(curr_balance - prev_balance, 2)
            
            stack_trace["fraud_flags"].append({
                "flag": "Sequential Inconsistency",
                "description": "Account balance increased unexpectedly in outgoing transaction",
                "evidence": {
                    "previous_balance": prev_balance,
                    "current_balance": curr_balance,
                    "balance_change": balance_change,
                    "expected": "Balance should decrease (be negative)",
                    "actual": f"Balance increased by {balance_change:.2f}",
                    "note": "All transactions are outgoing from sender's perspective"
                }
            })
        else:
            stack_trace["fraud_flags"].append({
                "flag": "Sequential Inconsistency",
                "description": "Account balance increased unexpectedly in outgoing transaction",
                "note": "Balance should only decrease for outgoing transactions"
            })
    
    # 6. Duplicate Transactions
    if row["flag_duplicate_transactions"]:
        # Find duplicate transactions
        duplicates = df[
            (df["sender_id"] == row["sender_id"]) &
            (df["recipient_id"] == row["recipient_id"]) &
            (df["amount"] == row["amount"])
        ][["transaction_id", "timestamp", "amount", "recipient_id"]].sort_values("timestamp")
        
        stack_trace["fraud_flags"].append({
            "flag": "Duplicate Transactions",
            "description": "Identical transactions (sender, recipient, amount) detected",
            "evidence": {
                "sender_id": row["sender_id"],
                "recipient_id": row["recipient_id"],
                "amount": float(row["amount"]),
                "duplicate_count": len(duplicates),
                "matching_transactions": convert_to_serializable(duplicates.to_dict(orient="records"))
            }
        })
    
    # 7. Rapid Small Transfers
    if row["flag_rapid_small_transfers"]:
        # Find all small transfers from this sender in 1-hour windows
        current_time = row["timestamp"]
        time_window_start = current_time - pd.Timedelta(hours=1)
        
        # Get sender's median for comparison
        sender_median = df[df["sender_id"] == row["sender_id"]]["amount"].median()
        
        small_transfers_in_window = df[
            (df["sender_id"] == row["sender_id"]) &
            (df["timestamp"] >= time_window_start) &
            (df["timestamp"] <= current_time) &
            (df["amount"] < sender_median)
        ][["transaction_id", "timestamp", "amount", "recipient_id"]].sort_values("timestamp")
        
        stack_trace["fraud_flags"].append({
            "flag": "Rapid Small Transfers",
            "description": "Multiple small transfers from sender within short timeframe",
            "evidence": {
                "sender_median_amount": float(sender_median),
                "transaction_amount": float(row["amount"]),
                "is_below_median": float(row["amount"]) < sender_median,
                "time_window": "1 hour",
                "small_transfers_in_window": len(small_transfers_in_window),
                "threshold": "More than 3 small transfers",
                "pattern": "Structuring/smurfing - breaking down large amounts",
                "transfers": convert_to_serializable(small_transfers_in_window.to_dict(orient="records"))
            }
        })
    
    # 8. Reporting Threshold Amount
    if row["flag_reporting_threshold"]:
        matching_threshold = None
        for threshold in [10000, 5000, 1000]:
            if threshold - 500 < row["amount"] < threshold:
                matching_threshold = threshold
                break
        
        stack_trace["fraud_flags"].append({
            "flag": "Reporting Threshold Amount",
            "description": "Transaction amount just below reporting threshold - structuring pattern",
            "evidence": {
                "transaction_amount": float(row["amount"]),
                "near_threshold": matching_threshold,
                "distance_from_threshold": float(matching_threshold - row["amount"]),
                "threshold_buffer": 500,
                "formula": f"{matching_threshold} - {float(row['amount'])} = {float(matching_threshold - row['amount'])}",
                "threshold_list": [10000, 5000, 1000],
                "pattern": "Classic structuring to avoid €10,000 AML reporting requirement"
            }
        })
    
    # 9. New/Rare Location
    if row["flag_new_rare_location"]:
        sender_location_count = len(df[(df["sender_id"] == row["sender_id"]) & (df["transaction_location"] == row["transaction_location"])])
        
        stack_trace["fraud_flags"].append({
            "flag": "New/Rare Location",
            "description": "Transaction at new or rarely-used location for this sender",
            "evidence": {
                "location": row["transaction_location"],
                "times_used_by_sender": sender_location_count,
                "threshold": "1 (new/first time)",
                "is_first_time": sender_location_count == 1,
                "note": "Sender has only transacted at this location once"
            }
        })
    
    # 10. Impossible Travel
    if row["flag_impossible_travel"]:
        # Find previous transaction for this sender with a known location
        sender_txns = df[df["sender_id"] == row["sender_id"]].sort_values("timestamp")
        current_idx_in_sorted = sender_txns[sender_txns["transaction_id"] == row["transaction_id"]].index[0]
        current_position = sender_txns.index.get_loc(current_idx_in_sorted)
        
        previous_location = None
        distance = None
        time_diff = None
        required_speed = None
        
        if current_position > 0:
            prev_txn = sender_txns.iloc[current_position - 1]
            previous_location = prev_txn["transaction_location"] if pd.notna(prev_txn["transaction_location"]) else None
            
            if previous_location and previous_location in city_coordinates and row["transaction_location"] in city_coordinates:
                lat1, lon1 = city_coordinates[previous_location]
                lat2, lon2 = city_coordinates[row["transaction_location"]]
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                time_diff = (row["timestamp"] - prev_txn["timestamp"]).total_seconds() / 3600
                required_speed = distance / time_diff if time_diff > 0 else float('inf')
        
        stack_trace["fraud_flags"].append({
            "flag": "Impossible Travel",
            "description": "Sender would need to travel impossibly fast between locations",
            "evidence": {
                "previous_location": previous_location,
                "current_location": row["transaction_location"],
                "distance_km": round(distance, 2) if distance else None,
                "time_difference_hours": round(time_diff, 2) if time_diff else None,
                "required_speed_kmh": round(required_speed, 2) if required_speed else None,
                "max_realistic_speed_kmh": max_travel_speed,
                "formula": f"{distance:.2f} km / {time_diff:.2f} hours = {required_speed:.2f} km/h" if distance and time_diff else "N/A",
                "exceeds_possible": f"Yes, {required_speed:.2f} > {max_travel_speed} km/h" if required_speed and required_speed > max_travel_speed else "N/A"
            }
        })
    
    # 11. Unusual Location-Payment Combination
    if row["flag_unusual_location_payment"]:
        # Find all transactions at this location for this sender
        sender_loc_txns = df[
            (df["sender_id"] == row["sender_id"]) &
            (df["transaction_location"] == row["transaction_location"]) &
            (df["payment_method"].notna())
        ]["payment_method"].value_counts()
        
        # Most common payment method at this location
        usual_payment_at_location = sender_loc_txns.idxmax() if len(sender_loc_txns) > 0 else None
        usage_count = sender_loc_txns.get(usual_payment_at_location, 0) if usual_payment_at_location else 0
        
        stack_trace["fraud_flags"].append({
            "flag": "Unusual Location-Payment Combination",
            "description": "Different payment method used at this location than usual",
            "evidence": {
                "location": row["transaction_location"],
                "used_payment_method": row["payment_method"],
                "usual_payment_method": usual_payment_at_location,
                "usual_method_usage_count": int(usage_count),
                "payment_methods_at_location": convert_to_serializable(sender_loc_txns.to_dict()),
                "mismatch": f"Sender typically uses {usual_payment_at_location} at {row['transaction_location']}, but used {row['payment_method']} this time"
            }
        })
    
    # 12. Amount IQR Outlier
    if row["flag_amount_iqr_outlier"]:
        sender_amounts = df[df["sender_id"] == row["sender_id"]]["amount"]
        Q1 = sender_amounts.quantile(0.25)
        Q3 = sender_amounts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        stack_trace["fraud_flags"].append({
            "flag": "Amount IQR Outlier",
            "description": "Transaction amount falls outside the Interquartile Range (IQR)",
            "evidence": {
                "transaction_amount": round(float(row["amount"]), 2),
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "is_below_lower": float(row["amount"]) < lower_bound,
                "is_above_upper": float(row["amount"]) > upper_bound,
                "pattern": "Transaction amount deviates significantly from sender's typical range"
            }
        })
    
    # 13. Mahalanobis Distance Outlier
    if row["flag_mahalanobis_outlier"]:
        stack_trace["fraud_flags"].append({
            "flag": "Mahalanobis Outlier",
            "description": "Multivariate outlier: unusual combination of amount, hour, payment method, and transaction type",
            "evidence": {
                "amount": round(float(row["amount"]), 2),
                "hour": int(row["hour"]),
                "payment_method": row["payment_method"],
                "transaction_type": row["transaction_type"],
                "pattern": "Unusual in combination, though individual features may seem normal"
            }
        })
    
    # 14. KL Divergence Distribution Shift
    if row["flag_kl_divergence"]:
        stack_trace["fraud_flags"].append({
            "flag": "KL Divergence Distribution Shift",
            "description": "Recent transaction distribution differs significantly from historical baseline",
            "evidence": {
                "amount": round(float(row["amount"]), 2),
                "pattern": "Recent transactions show different distribution than sender's historical behavior",
                "note": "May indicate account compromise or unauthorized user activity"
            }
        })
    
    # 15. Velocity Spike
    if row["flag_velocity_spike"]:
        # Count transactions in different windows
        current_time = row["timestamp"]
        txns_10min = len(df[
            (df["sender_id"] == row["sender_id"]) &
            (df["timestamp"] >= current_time - pd.Timedelta(minutes=10)) &
            (df["timestamp"] <= current_time)
        ])
        txns_1hour = len(df[
            (df["sender_id"] == row["sender_id"]) &
            (df["timestamp"] >= current_time - pd.Timedelta(hours=1)) &
            (df["timestamp"] <= current_time)
        ])
        txns_1day = len(df[
            (df["sender_id"] == row["sender_id"]) &
            (df["timestamp"] >= current_time - pd.Timedelta(days=1)) &
            (df["timestamp"] <= current_time)
        ])
        
        stack_trace["fraud_flags"].append({
            "flag": "Velocity Spike",
            "description": "Rapid burst of transactions from this sender",
            "evidence": {
                "txns_in_10min": txns_10min,
                "txns_in_1hour": txns_1hour,
                "txns_in_1day": txns_1day,
                "thresholds": {
                    "10_min_threshold": 5,
                    "1_hour_threshold": 10,
                    "1_day_threshold": 20
                },
                "pattern": "Account takeover indicator - automated/bot activity would show high velocity"
            }
        })
    
    # 16. Round Number Amount
    if row["flag_round_number_amount"]:
        stack_trace["fraud_flags"].append({
            "flag": "Round Number Amount",
            "description": "Transaction uses exact round amount (classic structuring pattern)",
            "evidence": {
                "amount": float(row["amount"]),
                "is_round": row["amount"] in [100, 500, 1000, 2000, 5000, 10000],
                "round_amounts_watched": [100, 500, 1000, 2000, 5000, 10000],
                "pattern": "Just-below-threshold or exact round amounts used to avoid reporting requirements"
            }
        })
    
    fraud_stack_traces.append(stack_trace)

# Save to JSON file
output_file = "fraud_stack_traces.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(convert_to_serializable({
        "generated_at": datetime.now().isoformat(),
        "total_fraudulent_transactions": len(fraud_stack_traces),
        "fraud_percentage": round(fraud_percentage, 2),
        "fraudulent_transactions": fraud_stack_traces
    }), f, indent=2, ensure_ascii=False)

print(f"\nDetailed fraud stack traces saved to: {output_file}")