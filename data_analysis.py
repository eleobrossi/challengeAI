import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("The_Truman_Show_train\\public\\transactions.csv")

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
# Most common hour per sender
sender_common_hour = (
    df.groupby(["sender_id", "hour"])
    .size()
    .reset_index(name="count")
)

idx = sender_common_hour.groupby("sender_id")["count"].idxmax()
common_hours = sender_common_hour.loc[idx][["sender_id", "hour"]]
common_hours.columns = ["sender_id", "usual_hour"]

df = df.merge(common_hours, on="sender_id", how="left")

# Flag if transaction happens outside +-3 hours
df["flag_abnormal_hour"] = (
    abs(df["hour"] - df["usual_hour"]) > 3
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
        
        # Flag if balance math doesn't match
        if abs(expected_balance - current_txn["balance_after"]) > 0.01:
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
sender_location_freq = df[df["location"].notna()].groupby(["sender_id", "location"]).size().reset_index(name="location_count")
sender_location_freq["is_rare"] = sender_location_freq["location_count"] <= 1

df = df.merge(
    sender_location_freq[["sender_id", "location", "is_rare"]],
    on=["sender_id", "location"],
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
        pd.notna(current_row["location"]) and pd.notna(previous_row["location"]) and
        current_row["location"] in city_coordinates and previous_row["location"] in city_coordinates):
        
        lat1, lon1 = city_coordinates[previous_row["location"]]
        lat2, lon2 = city_coordinates[current_row["location"]]
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
sender_loc_payment = df[df["location"].notna() & df["payment_method"].notna()].groupby(
    ["sender_id", "location", "payment_method"]
).size().reset_index(name="count")

# Get the most common payment method per sender/location
sender_loc_payment_sorted = sender_loc_payment.sort_values(["sender_id", "location", "count"], ascending=[True, True, False])
usual_payment = sender_loc_payment_sorted.drop_duplicates(["sender_id", "location"], keep="first")[["sender_id", "location", "payment_method"]]
usual_payment.columns = ["sender_id", "location", "usual_payment_method"]

df = df.merge(
    usual_payment,
    on=["sender_id", "location"],
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


# Combined Fraud Score
# ----
df["fraud_score"] = (
    df["flag_amount_spike"].astype(int)
    + df["flag_unusual_transaction_type"].astype(int)
    # + df["flag_new_payment_method"].astype(int)
    + df["flag_abnormal_hour"].astype(int)
    + df["flag_balance_mismatch"].astype(int)
    + df["flag_sequential_inconsistency"].astype(int)
    + df["flag_duplicate_transactions"].astype(int)
    + df["flag_rapid_small_transfers"].astype(int)
    + df["flag_reporting_threshold"].astype(int)
    + df["flag_new_rare_location"].astype(int)
    + df["flag_impossible_travel"].astype(int)
    + df["flag_unusual_location_payment"].astype(int)
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
    
    fraud_reasons.append(" | ".join(reasons) if reasons else "None")

df["fraud_reason"] = fraud_reasons
# Flag as suspicious if score >= 2
df["flag_suspicious"] = df["fraud_score"] >= 2

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
print(f"Fraudulent transactions: {fraud_count}")
print(f"Fraud percentage: {fraud_percentage:.2f}%")


# Create detailed JSON with stack traces for fraudulent transactions
import json
from datetime import datetime

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
        "location": row["location"] if pd.notna(row["location"]) else None,
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
                "sender_usual_hour": int(row["usual_hour"]),
                "transaction_hour": int(row["hour"]),
                "hour_difference": int(abs(row["hour"] - row["usual_hour"])),
                "time_deviation": f"{int(abs(row['hour'] - row['usual_hour']))} hours away from typical time"
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
            expected_balance = float(prev_txn["balance_after"] - row["amount"])
            actual_balance = float(row["balance_after"])
            discrepancy = actual_balance - expected_balance
            
            stack_trace["fraud_flags"].append({
                "flag": "Balance Math Mismatch",
                "description": "Account balance does not match expected value after transaction",
                "evidence": {
                    "previous_balance": float(prev_txn["balance_after"]),
                    "transaction_amount": float(row["amount"]),
                    "expected_calculation": f"{float(prev_txn['balance_after']):.2f} - {float(row['amount']):.2f}",
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
            balance_change = float(row["balance_after"] - prev_txn["balance_after"])
            
            stack_trace["fraud_flags"].append({
                "flag": "Sequential Inconsistency",
                "description": "Account balance increased unexpectedly in outgoing transaction",
                "evidence": {
                    "previous_balance": float(prev_txn["balance_after"]),
                    "current_balance": float(row["balance_after"]),
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
                "matching_transactions": duplicates.to_dict(orient="records")
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
                "transfers": small_transfers_in_window.to_dict(orient="records")
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
        sender_location_count = len(df[(df["sender_id"] == row["sender_id"]) & (df["location"] == row["location"])])
        
        stack_trace["fraud_flags"].append({
            "flag": "New/Rare Location",
            "description": "Transaction at new or rarely-used location for this sender",
            "evidence": {
                "location": row["location"],
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
            previous_location = prev_txn["location"] if pd.notna(prev_txn["location"]) else None
            
            if previous_location and previous_location in city_coordinates and row["location"] in city_coordinates:
                lat1, lon1 = city_coordinates[previous_location]
                lat2, lon2 = city_coordinates[row["location"]]
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                time_diff = (row["timestamp"] - prev_txn["timestamp"]).total_seconds() / 3600
                required_speed = distance / time_diff if time_diff > 0 else float('inf')
        
        stack_trace["fraud_flags"].append({
            "flag": "Impossible Travel",
            "description": "Sender would need to travel impossibly fast between locations",
            "evidence": {
                "previous_location": previous_location,
                "current_location": row["location"],
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
            (df["location"] == row["location"]) &
            (df["payment_method"].notna())
        ]["payment_method"].value_counts()
        
        # Most common payment method at this location
        usual_payment_at_location = sender_loc_txns.idxmax() if len(sender_loc_txns) > 0 else None
        usage_count = sender_loc_txns.get(usual_payment_at_location, 0) if usual_payment_at_location else 0
        
        stack_trace["fraud_flags"].append({
            "flag": "Unusual Location-Payment Combination",
            "description": "Different payment method used at this location than usual",
            "evidence": {
                "location": row["location"],
                "used_payment_method": row["payment_method"],
                "usual_payment_method": usual_payment_at_location,
                "usual_method_usage_count": int(usage_count),
                "payment_methods_at_location": sender_loc_txns.to_dict(),
                "mismatch": f"Sender typically uses {usual_payment_at_location} at {row['location']}, but used {row['payment_method']} this time"
            }
        })
    
    fraud_stack_traces.append(stack_trace)

# Save to JSON file
output_file = "fraud_stack_traces.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.now().isoformat(),
        "total_fraudulent_transactions": len(fraud_stack_traces),
        "fraud_percentage": round(fraud_percentage, 2),
        "fraudulent_transactions": fraud_stack_traces
    }, f, indent=2, ensure_ascii=False)

print(f"\nDetailed fraud stack traces saved to: {output_file}")