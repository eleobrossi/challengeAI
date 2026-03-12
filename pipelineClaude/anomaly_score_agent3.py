#!/usr/bin/env python3
"""
anomaly_score_agent3.py

Pure-statistical anomaly scoring agent for the Reply Mirror AI Agent Challenge.

Reads the feature-engineered master table produced by feature_engineer_agent2.py
(feature_outputs/features_master.csv) or falls back to raw transactions.csv,
and computes a composite, multi-signal anomaly score for every transaction.

No LLM is used – all signals are derived from statistics alone, making this
agent fast, reproducible and interpretable.

Output:
    <dataset_dir>/anomaly_outputs/
        anomaly_scores.csv          – per-transaction scores + flag
        signal_weights_report.json  – how each signal contributed
        top_suspicious.csv          – top-N most suspicious transactions

Usage:
    python anomaly_score_agent3.py --dataset datasets/level_1
    python anomaly_score_agent3.py --dataset datasets/level_1 --top-n 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("anomaly_score_agent3")

# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def percentile_rank(series: pd.Series) -> pd.Series:
    """Return per-element percentile rank within the series (0-1)."""
    return series.rank(pct=True, na_option="bottom")


def sigmoid(x: np.ndarray, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Saturating sigmoid to map arbitrary values into (0,1)."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def robust_clamp(series: pd.Series, lo_pct: float = 1.0, hi_pct: float = 99.0) -> pd.Series:
    """Clamp series at given percentiles to reduce outlier influence on normalisation."""
    lo = series.quantile(lo_pct / 100.0)
    hi = series.quantile(hi_pct / 100.0)
    return series.clip(lower=lo, upper=hi)


def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


# =============================================================================
# Signal builders
# =============================================================================

class SignalBuilder:
    """
    Computes individual fraud-risk signals (each in 0-1 scale) from the
    feature-engineered DataFrame. Each method returns a named pd.Series.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------
    # Amount signals
    # ------------------------------------------------------------------
    def amount_global_mad_z_signal(self) -> pd.Series:
        """High absolute MAD z-score → unusual amount globally."""
        col = "amount_global_mad_z"
        if col not in self.df.columns:
            # compute from scratch
            x = pd.to_numeric(self.df.get("amount", pd.Series()), errors="coerce")
            med = x.median()
            mad = (x - med).abs().median()
            mad = mad if not pd.isna(mad) and mad > 1e-9 else 1.0
            z = 0.6745 * (x - med).abs() / mad
        else:
            z = pd.to_numeric(self.df[col], errors="coerce").abs()

        return minmax_norm(robust_clamp(z.fillna(0.0), 0, 99)).rename("sig_amount_global_mad_z")

    def amount_vs_sender_mean_signal(self) -> pd.Series:
        """Transaction amount greatly exceeds sender's historical mean."""
        ratio_col = "amount_over_sender_mean_ratio"
        if ratio_col in self.df.columns:
            ratio = pd.to_numeric(self.df[ratio_col], errors="coerce").fillna(1.0)
        else:
            ratio = pd.Series(1.0, index=self.df.index)
        # Cap at 20x to avoid single outlier domination
        ratio = ratio.clip(0, 20)
        return minmax_norm(ratio).rename("sig_amount_vs_sender_mean")

    def amount_vs_pair_mean_signal(self) -> pd.Series:
        """Transaction amount greatly exceeds this sender→recipient pair's history."""
        ratio_col = "amount_over_pair_mean_ratio"
        if ratio_col in self.df.columns:
            ratio = pd.to_numeric(self.df[ratio_col], errors="coerce").fillna(1.0)
        else:
            ratio = pd.Series(1.0, index=self.df.index)
        ratio = ratio.clip(0, 20)
        return minmax_norm(ratio).rename("sig_amount_vs_pair_mean")

    def amount_absolute_percentile_signal(self) -> pd.Series:
        """Raw amount absolute percentile – very large absolute amounts are suspicious."""
        amt = pd.to_numeric(self.df.get("amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        return percentile_rank(amt).rename("sig_amount_absolute_percentile")

    def balance_drain_signal(self) -> pd.Series:
        """
        After-transaction balance drain signal.
        Low balance_after / amount ratio → dangerous drain of account.
        """
        bal_col_candidates = ["balance_after", "balance", "Balance"]
        bal_col = next((c for c in bal_col_candidates if c in self.df.columns), None)
        if bal_col is None:
            return pd.Series(0.0, index=self.df.index, name="sig_balance_drain")

        bal = pd.to_numeric(self.df[bal_col], errors="coerce")
        amt = pd.to_numeric(self.df.get("amount", pd.Series(dtype=float)), errors="coerce")

        # ratio of balance to amount: very low ratio = drained account
        ratio = bal / (amt.clip(lower=1e-3))
        # low ratio → high risk; invert and clamp
        drain = 1.0 / (1.0 + ratio.clip(lower=0))
        return minmax_norm(drain.fillna(0.0)).rename("sig_balance_drain")

    # ------------------------------------------------------------------
    # Velocity signals
    # ------------------------------------------------------------------
    def velocity_1h_signal(self) -> pd.Series:
        col = "sender_tx_last_1h"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_velocity_1h")
        v = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        # 3+ tx in 1h is very suspicious
        return minmax_norm(v.clip(0, 10)).rename("sig_velocity_1h")

    def velocity_24h_signal(self) -> pd.Series:
        col = "sender_tx_last_24h"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_velocity_24h")
        v = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        return minmax_norm(v.clip(0, 20)).rename("sig_velocity_24h")

    def velocity_amt_24h_signal(self) -> pd.Series:
        """Total amount sent in last 24h vs sender's typical daily spend."""
        col = "sender_amt_last_24h"
        mean_col = "sender_amount_mean"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_velocity_amt_24h")
        amt_24h = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        mean_amt = pd.to_numeric(self.df.get(mean_col, pd.Series(dtype=float)), errors="coerce").fillna(1.0).clip(lower=1.0)
        ratio = amt_24h / mean_amt
        return minmax_norm(ratio.clip(0, 50)).rename("sig_velocity_amt_24h")

    def gap_hours_signal(self) -> pd.Series:
        """Very short gap after previous transaction is suspicious."""
        col = "sender_gap_hours"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_gap_hours")
        gap = pd.to_numeric(self.df[col], errors="coerce")
        # very short gap → high risk (inverse: 0h → 1.0, 24h → ~0)
        gap_inv = 1.0 / (1.0 + gap.clip(lower=0).fillna(24))
        return minmax_norm(gap_inv).rename("sig_gap_hours")

    # ------------------------------------------------------------------
    # Novelty / behavioural signals
    # ------------------------------------------------------------------
    def new_recipient_signal(self) -> pd.Series:
        col = "is_new_recipient_for_sender"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_new_recipient")
        return pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).clip(0, 1).rename("sig_new_recipient")

    def new_recipient_high_amount_signal(self) -> pd.Series:
        col = "new_recipient_high_amount_flag"
        if col not in self.df.columns:
            new_rec = self.new_recipient_signal()
            amt_p95 = self.df.get("amount_gt_sender_p95", pd.Series(0.0, index=self.df.index))
            return (new_rec * pd.to_numeric(amt_p95, errors="coerce").fillna(0)).rename("sig_new_recipient_high_amount")
        return pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).rename("sig_new_recipient_high_amount")

    def night_transaction_signal(self) -> pd.Series:
        col = "is_night"
        if col not in self.df.columns:
            if "hour" in self.df.columns:
                return self.df["hour"].between(0, 5).astype(float).rename("sig_night_tx")
            return pd.Series(0.0, index=self.df.index, name="sig_night_tx")
        return pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).rename("sig_night_tx")

    def night_new_high_signal(self) -> pd.Series:
        col = "night_new_high_flag"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_night_new_high")
        return pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).rename("sig_night_new_high")

    def burst_signal(self) -> pd.Series:
        b1 = pd.to_numeric(self.df.get("burst_1h_flag", pd.Series(0, index=self.df.index)), errors="coerce").fillna(0)
        b24 = pd.to_numeric(self.df.get("burst_24h_flag", pd.Series(0, index=self.df.index)), errors="coerce").fillna(0)
        return ((b1 * 0.6 + b24 * 0.4).clip(0, 1)).rename("sig_burst")

    # ------------------------------------------------------------------
    # IBAN / identity signals
    # ------------------------------------------------------------------
    def iban_mismatch_signal(self) -> pd.Series:
        """Sender or recipient IBAN doesn't match the registered IBAN."""
        s_col = "sender_iban_consistent"
        r_col = "recipient_iban_consistent"
        s = pd.to_numeric(self.df.get(s_col, pd.Series(1, index=self.df.index)), errors="coerce").fillna(1.0)
        r = pd.to_numeric(self.df.get(r_col, pd.Series(1, index=self.df.index)), errors="coerce").fillna(1.0)
        # inconsistent = 0, so mismatch risk = 1 - consistent
        mismatch = 1.0 - (s * 0.5 + r * 0.5)
        return mismatch.clip(0, 1).rename("sig_iban_mismatch")

    # ------------------------------------------------------------------
    # Graph / network signals
    # ------------------------------------------------------------------
    def recipient_hub_signal(self) -> pd.Series:
        col = "recipient_is_hub"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_recipient_hub")
        return pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).rename("sig_recipient_hub")

    def sender_out_in_degree_signal(self) -> pd.Series:
        """Very high out-degree vs in-degree: sender is spending much more than receiving."""
        col = "sender_out_in_degree_ratio"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_out_in_degree")
        ratio = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).clip(0, 20)
        return minmax_norm(ratio).rename("sig_out_in_degree")

    def unique_recipients_7d_signal(self) -> pd.Series:
        """Many unique recipients in last 7 days → fan-out attack."""
        col = "sender_unique_recips_last_7d"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_unique_recips_7d")
        v = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
        return minmax_norm(v.clip(0, 30)).rename("sig_unique_recips_7d")

    # ------------------------------------------------------------------
    # Message / communication signals
    # ------------------------------------------------------------------
    def message_trigger_signal(self) -> pd.Series:
        col = "msg_triggered_tx_flag"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_msg_trigger")
        return pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).rename("sig_msg_trigger")

    def suspicious_msg_proximity_signal(self) -> pd.Series:
        col24 = "suspicious_msg_prev_24h_sender"
        col72 = "suspicious_msg_prev_72h_sender"
        s24 = pd.to_numeric(self.df.get(col24, pd.Series(0, index=self.df.index)), errors="coerce").fillna(0.0)
        s72 = pd.to_numeric(self.df.get(col72, pd.Series(0, index=self.df.index)), errors="coerce").fillna(0.0)
        combined = (s24 * 0.7 + s72 * 0.3).clip(0, 5)
        return minmax_norm(combined).rename("sig_suspicious_msg_proximity")

    # ------------------------------------------------------------------
    # Salary overspend signals
    # ------------------------------------------------------------------
    def amount_vs_salary_signal(self) -> pd.Series:
        col = "sender_amount_vs_salary_ratio"
        if col not in self.df.columns:
            return pd.Series(0.0, index=self.df.index, name="sig_amount_vs_salary")
        ratio = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0).clip(0, 10)
        return minmax_norm(ratio).rename("sig_amount_vs_salary")


# =============================================================================
# Composite scorer
# =============================================================================

SIGNAL_WEIGHTS: Dict[str, float] = {
    # Amount anomaly
    "sig_amount_global_mad_z":          0.12,
    "sig_amount_vs_sender_mean":        0.10,
    "sig_amount_vs_pair_mean":          0.08,
    "sig_amount_absolute_percentile":   0.05,
    "sig_balance_drain":                0.06,
    # Velocity
    "sig_velocity_1h":                  0.10,
    "sig_velocity_24h":                 0.06,
    "sig_velocity_amt_24h":             0.05,
    "sig_gap_hours":                    0.04,
    # Novelty / behavioural
    "sig_new_recipient":                0.04,
    "sig_new_recipient_high_amount":    0.06,
    "sig_night_tx":                     0.03,
    "sig_night_new_high":               0.05,
    "sig_burst":                        0.04,
    # Identity
    "sig_iban_mismatch":                0.03,
    # Network
    "sig_recipient_hub":                0.04,
    "sig_out_in_degree":                0.03,
    "sig_unique_recips_7d":             0.05,
    # Communication
    "sig_msg_trigger":                  0.04,
    "sig_suspicious_msg_proximity":     0.05,
    # Salary
    "sig_amount_vs_salary":             0.04,
}

# Signals that, if present (>0.5), receive a multiplicative boost
AMPLIFIER_SIGNALS: List[str] = [
    "sig_new_recipient_high_amount",
    "sig_night_new_high",
    "sig_msg_trigger",
    "sig_burst",
]


class AnomalyScoringAgent:
    """
    Computes a composite anomaly score for every transaction row.
    Purely statistical, no LLM calls.
    """

    def __init__(self, dataset_dir: str, top_n: int = 1000):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = self.dataset_dir / "anomaly_outputs"
        self.top_n = top_n
        ensure_dir(self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        df = self._load_features()
        log.info(f"Loaded feature matrix: {len(df):,} rows × {df.shape[1]} columns")

        signals = self._build_signals(df)
        score_df = self._compute_composite(df, signals)

        # Persist
        score_df.to_csv(self.output_dir / "anomaly_scores.csv", index=False)

        top = score_df.sort_values("anomaly_score", ascending=False).head(self.top_n)
        top.to_csv(self.output_dir / "top_suspicious.csv", index=False)

        report = self._signal_report(score_df)
        save_json(report, self.output_dir / "signal_weights_report.json")

        log.info(f"Anomaly scoring complete. Output → {self.output_dir}")
        log.info(f"Score stats: mean={score_df['anomaly_score'].mean():.4f}  "
                 f"p90={score_df['anomaly_score'].quantile(0.90):.4f}  "
                 f"p99={score_df['anomaly_score'].quantile(0.99):.4f}")
        log.info(f"Flagged as suspicious (score > 0.50): "
                 f"{(score_df['anomaly_score'] > 0.50).sum():,}")

        return score_df

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------
    def _load_features(self) -> pd.DataFrame:
        feature_path = self.dataset_dir / "feature_outputs" / "features_master.csv"
        tx_path = self.dataset_dir / "transactions.csv"

        if feature_path.exists():
            log.info(f"Loading feature-engineered table from {feature_path}")
            df = pd.read_csv(feature_path, low_memory=False)
        elif tx_path.exists():
            log.warning(f"features_master.csv not found; falling back to {tx_path}")
            df = pd.read_csv(tx_path, low_memory=False)
            # Standardise column names
            col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
            df = df.rename(columns=col_map)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            if "timestamp" in df.columns:
                df["hour"] = df["timestamp"].dt.hour
                df["is_night"] = df["hour"].between(0, 5).astype(int)
        else:
            raise FileNotFoundError("Neither features_master.csv nor transactions.csv found.")

        return df

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _build_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sb = SignalBuilder(df)

        sigs = pd.DataFrame(index=df.index)
        for name, method in [
            ("sig_amount_global_mad_z",       sb.amount_global_mad_z_signal),
            ("sig_amount_vs_sender_mean",      sb.amount_vs_sender_mean_signal),
            ("sig_amount_vs_pair_mean",        sb.amount_vs_pair_mean_signal),
            ("sig_amount_absolute_percentile", sb.amount_absolute_percentile_signal),
            ("sig_balance_drain",              sb.balance_drain_signal),
            ("sig_velocity_1h",                sb.velocity_1h_signal),
            ("sig_velocity_24h",               sb.velocity_24h_signal),
            ("sig_velocity_amt_24h",           sb.velocity_amt_24h_signal),
            ("sig_gap_hours",                  sb.gap_hours_signal),
            ("sig_new_recipient",              sb.new_recipient_signal),
            ("sig_new_recipient_high_amount",  sb.new_recipient_high_amount_signal),
            ("sig_night_tx",                   sb.night_transaction_signal),
            ("sig_night_new_high",             sb.night_new_high_signal),
            ("sig_burst",                      sb.burst_signal),
            ("sig_iban_mismatch",              sb.iban_mismatch_signal),
            ("sig_recipient_hub",              sb.recipient_hub_signal),
            ("sig_out_in_degree",              sb.sender_out_in_degree_signal),
            ("sig_unique_recips_7d",           sb.unique_recipients_7d_signal),
            ("sig_msg_trigger",                sb.message_trigger_signal),
            ("sig_suspicious_msg_proximity",   sb.suspicious_msg_proximity_signal),
            ("sig_amount_vs_salary",           sb.amount_vs_salary_signal),
        ]:
            s = method()
            sigs[name] = s.values

        return sigs

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------
    def _compute_composite(self, df: pd.DataFrame, sigs: pd.DataFrame) -> pd.DataFrame:
        total_w = sum(SIGNAL_WEIGHTS.values())
        score = pd.Series(0.0, index=df.index)

        for sig_name, w in SIGNAL_WEIGHTS.items():
            if sig_name in sigs.columns:
                score += sigs[sig_name] * (w / total_w)

        # Amplifier boost: any major compound flag → push score higher
        for amp in AMPLIFIER_SIGNALS:
            if amp in sigs.columns:
                boost_mask = sigs[amp] > 0.5
                score = score + (boost_mask.astype(float) * 0.05)

        score = score.clip(0, 1)

        # Assemble output DataFrame
        tx_id_col = next((c for c in ["transaction_id", "TransactionID", "id"] if c in df.columns), None)

        out = pd.DataFrame(index=df.index)
        if tx_id_col:
            out["transaction_id"] = df[tx_id_col].astype(str)
        if "sender_id" in df.columns:
            out["sender_id"] = df["sender_id"].astype(str)
        if "recipient_id" in df.columns:
            out["recipient_id"] = df["recipient_id"].astype(str)
        if "amount" in df.columns:
            out["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        if "timestamp" in df.columns:
            out["timestamp"] = df["timestamp"].astype(str)
        # support both "transaction_type" (raw CSV) and "payment_type" (renamed by agent2)
        for _tx_type_col in ["transaction_type", "payment_type"]:
            if _tx_type_col in df.columns:
                out["transaction_type"] = df[_tx_type_col].astype(str)
                break

        out = pd.concat([out, sigs], axis=1)
        out["anomaly_score"] = score
        out["is_suspicious"] = (score > 0.50).astype(int)
        out["risk_tier"] = pd.cut(
            score,
            bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )

        return out

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def _signal_report(self, score_df: pd.DataFrame) -> Dict:
        sig_cols = [c for c in score_df.columns if c.startswith("sig_")]
        report = {
            "n_transactions": int(len(score_df)),
            "n_suspicious": int(score_df["is_suspicious"].sum()),
            "score_mean": float(score_df["anomaly_score"].mean()),
            "score_std": float(score_df["anomaly_score"].std()),
            "score_p50": float(score_df["anomaly_score"].quantile(0.50)),
            "score_p90": float(score_df["anomaly_score"].quantile(0.90)),
            "score_p99": float(score_df["anomaly_score"].quantile(0.99)),
            "signal_weights": SIGNAL_WEIGHTS,
            "signal_mean_values": {},
            "risk_tier_distribution": score_df["risk_tier"].value_counts().to_dict(),
        }

        for col in sig_cols:
            if col in score_df.columns:
                report["signal_mean_values"][col] = round(
                    float(pd.to_numeric(score_df[col], errors="coerce").mean()), 6
                )

        return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Statistical Anomaly Scoring Agent – Reply Mirror Challenge")
    parser.add_argument("--dataset", required=True, help="Dataset directory (e.g. datasets/level_1)")
    parser.add_argument("--top-n", type=int, default=1000, help="Number of top suspicious rows to export")
    args = parser.parse_args()

    agent = AnomalyScoringAgent(args.dataset, top_n=args.top_n)
    agent.run()


if __name__ == "__main__":
    main()
