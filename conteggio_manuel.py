import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILE — solo transaction_id sospetti (formato challenge)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_IDS_FILE = "output.txt"
df = pd.read_csv("transactions_flagged.csv")

suspicious_ids = df[df["flag_suspicious"] == True]["transaction_id"].tolist()

with open(OUTPUT_IDS_FILE, "w") as f:
    for tid in suspicious_ids:
        f.write(str(tid) + "\n")

print(f"\n=== OUTPUT CHALLENGE: {OUTPUT_IDS_FILE} ===")
print(f"Transazioni sospette: {len(suspicious_ids)}")
for tid in suspicious_ids:
    print(tid)
