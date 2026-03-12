#!/usr/bin/env python3
"""
advanced_eda_agent.py

Single-agent advanced exploratory analytics for the AI Agent Challenge datasets.

Expected dataset files (if present):
- transactions.csv
- users.json
- locations.json
- sms.json
- mails.json

Outputs under:
<dataset_dir>/eda_outputs/

Usage:
    python advanced_eda_agent.py --dataset datasets/level_1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("advanced_eda_agent")


# =============================================================================
# Helpers
# =============================================================================

def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def coalesce(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def resolve_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    for a in aliases:
        al = a.lower()
        for c in df.columns:
            if al in c.lower():
                return c
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def iqr_outlier_flags(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return pd.to_numeric(series, errors="coerce").apply(lambda x: False if pd.isna(x) else (x < lo or x > hi))


def mad_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (x - med) / mad


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1 - a)))


# =============================================================================
# Agent
# =============================================================================

@dataclass
class AdvancedEDAAgent:
    dataset_dir: str
    output_dir: Path = field(init=False)

    def __post_init__(self):
        self.dataset_dir = str(self.dataset_dir)
        self.output_dir = Path(self.dataset_dir) / "eda_outputs"
        ensure_dir(self.output_dir)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(self) -> None:
        data = self._load_all()

        tx = data["transactions"]
        users = data["users"]
        locations = data["locations"]
        sms = data["sms"]
        mails = data["mails"]

        summary = {
            "dataset_dir": self.dataset_dir,
            "n_transactions": int(len(tx)),
            "n_users": int(len(users)),
            "n_locations": int(sum(len(v) for v in locations.values())),
            "n_sms": int(len(sms)),
            "n_mails": int(len(mails)),
        }

        if tx.empty:
            log.warning("transactions.csv not found or empty; analysis limited.")
            save_json(summary, self.output_dir / "dataset_summary.json")
            return

        tx = self._enrich_transactions(tx, users)
        tx = self._temporal_features(tx)
        tx_stats = self._transaction_stats(tx)
        user_profiles = self._build_user_profiles(tx, users, locations)
        counterparty_profiles = self._build_counterparty_profiles(tx, users)
        salary_candidates = self._detect_salary_patterns(tx)
        temporal_patterns = self._temporal_patterns(tx)
        graph_nodes, graph_edges, G = self._graph_analysis(tx)

        tx_stats.to_csv(self.output_dir / "transaction_stats.csv", index=False)
        user_profiles.to_csv(self.output_dir / "user_profiles.csv", index=False)
        counterparty_profiles.to_csv(self.output_dir / "counterparty_profiles.csv", index=False)
        salary_candidates.to_csv(self.output_dir / "salary_candidates.csv", index=False)
        temporal_patterns.to_csv(self.output_dir / "temporal_patterns.csv", index=False)
        graph_nodes.to_csv(self.output_dir / "graph_metrics_nodes.csv", index=False)
        graph_edges.to_csv(self.output_dir / "graph_metrics_edges.csv", index=False)

        geo_summary = self._location_profiles(locations, users)
        geo_summary.to_csv(self.output_dir / "location_profiles.csv", index=False)

        msg_summary = self._message_summary(sms, mails)
        msg_summary.to_csv(self.output_dir / "message_summary.csv", index=False)

        summary.update({
            "n_unique_senders": int(tx["sender_id"].nunique()) if "sender_id" in tx.columns else 0,
            "n_unique_recipients": int(tx["recipient_id"].nunique()) if "recipient_id" in tx.columns else 0,
            "n_salary_candidates": int(len(salary_candidates)),
            "graph_nodes": int(G.number_of_nodes()),
            "graph_edges": int(G.number_of_edges()),
        })
        save_json(summary, self.output_dir / "dataset_summary.json")

        self._make_plots(tx, user_profiles, geo_summary, G)
        log.info(f"EDA completed. Outputs written to {self.output_dir}")

    # -------------------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------------------
    def _load_all(self) -> Dict[str, Any]:
        return {
            "transactions": self._load_transactions(Path(self.dataset_dir) / "transactions.csv"),
            "users": self._load_users(Path(self.dataset_dir) / "users.json"),
            "locations": self._load_locations(Path(self.dataset_dir) / "locations.json"),
            "sms": self._load_sms(Path(self.dataset_dir) / "sms.json"),
            "mails": self._load_mails(Path(self.dataset_dir) / "mails.json"),
        }

    def _load_transactions(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)

        aliases = {
            "transaction_id": ["transaction_id", "TransactionID", "id", "tx_id"],
            "timestamp": ["timestamp", "datetime", "date", "time", "created_at"],
            "sender_id": ["sender_id", "sender", "from_user", "from_id"],
            "recipient_id": ["recipient_id", "recipient", "to_user", "to_id"],
            "sender_iban": ["sender_iban", "from_iban", "iban_from"],
            "recipient_iban": ["recipient_iban", "to_iban", "iban_to"],
            "amount": ["amount", "value", "transaction_amount"],
            "currency": ["currency", "curr"],
            "payment_type": ["payment_type", "type", "transaction_type", "payment_method"],
            "merchant": ["merchant", "merchant_name", "shop", "counterparty_name"],
            "status": ["status", "state"],
        }

        rename = {}
        for tgt, als in aliases.items():
            src = resolve_col(df, als)
            if src:
                rename[src] = tgt

        df = df.rename(columns=rename)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        for c in ["transaction_id", "sender_id", "recipient_id", "sender_iban", "recipient_iban", "merchant", "payment_type"]:
            if c in df.columns:
                df[c] = df[c].astype(str)

        return df

    def _load_users(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()

        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = []

        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            if "data" in raw and isinstance(raw["data"], list):
                rows = raw["data"]
            else:
                for k, v in raw.items():
                    if isinstance(v, dict):
                        row = dict(v)
                        if "user_id" not in row and "id" not in row:
                            row["user_id"] = k
                        rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        aliases = {
            "user_id": ["user_id", "_user_id", "id", "citizen_id", "sender_id", "recipient_id"],
            "iban": ["iban", "IBAN", "account_iban"],
            "birth_year": ["birth_year", "BirthYear"],
            "job": ["job", "occupation", "profession"],
        }

        rename = {}
        for tgt, als in aliases.items():
            src = resolve_col(df, als)
            if src:
                rename[src] = tgt
        df = df.rename(columns=rename)

        if "user_id" in df.columns:
            df["user_id"] = df["user_id"].astype(str)
        if "iban" in df.columns:
            df["iban"] = df["iban"].astype(str)

        return df

    def _load_locations(self, path: Path) -> Dict[str, List[Dict]]:
        if not path.exists():
            return {}

        raw = json.loads(path.read_text(encoding="utf-8"))
        flat = []

        def collect(obj):
            if isinstance(obj, list):
                for it in obj:
                    collect(it)
            elif isinstance(obj, dict):
                if any(k in obj for k in ("lat", "Lat", "latitude")) and any(k in obj for k in ("lng", "Lng", "lon", "longitude")):
                    flat.append(obj)
                elif "data" in obj and isinstance(obj["data"], list):
                    collect(obj["data"])
                else:
                    for k, v in obj.items():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    row = dict(item)
                                    if not coalesce(row, ["BioTag", "biotag", "user_id", "sender_id", "recipient_id"]):
                                        row["BioTag"] = str(k)
                                    flat.append(row)
                        elif isinstance(v, dict):
                            collect(v)

        collect(raw)

        out = defaultdict(list)
        for row in flat:
            tag = str(coalesce(row, ["BioTag", "biotag", "user_id", "sender_id", "recipient_id"], ""))
            if tag:
                out[tag].append(row)
        return dict(out)

    def _load_sms(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []
        return pd.DataFrame(rows)

    def _load_mails(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Enrichment
    # -------------------------------------------------------------------------
    def _enrich_transactions(self, tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        df = tx.copy()

        if "amount" in df.columns:
            df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0))
            df["amount_outlier_iqr"] = iqr_outlier_flags(df["amount"])
            df["amount_mad_z"] = mad_zscore(df["amount"]).fillna(0.0).round(4)

        if not users.empty:
            user_cols = [c for c in ["user_id", "iban", "birth_year", "job"] if c in users.columns]
            users_small = users[user_cols].copy()

            if "sender_id" in df.columns and "user_id" in users_small.columns:
                tmp = users_small.add_prefix("sender_user_")
                df = df.merge(tmp, left_on="sender_id", right_on="sender_user_user_id", how="left")

            if "recipient_id" in df.columns and "user_id" in users_small.columns:
                tmp = users_small.add_prefix("recipient_user_")
                df = df.merge(tmp, left_on="recipient_id", right_on="recipient_user_user_id", how="left")

            if "sender_iban" in df.columns and "iban" in users_small.columns:
                tmp = users_small.add_prefix("sender_iban_user_")
                df = df.merge(tmp, left_on="sender_iban", right_on="sender_iban_user_iban", how="left")

            if "recipient_iban" in df.columns and "iban" in users_small.columns:
                tmp = users_small.add_prefix("recipient_iban_user_")
                df = df.merge(tmp, left_on="recipient_iban", right_on="recipient_iban_user_iban", how="left")

            if {"sender_id", "sender_iban", "sender_user_iban"}.issubset(df.columns):
                df["sender_iban_consistent"] = (df["sender_iban"] == df["sender_user_iban"]).fillna(False)

            if {"recipient_id", "recipient_iban", "recipient_user_iban"}.issubset(df.columns):
                df["recipient_iban_consistent"] = (df["recipient_iban"] == df["recipient_user_iban"]).fillna(False)

        return df

    def _temporal_features(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.copy()
        if "timestamp" not in df.columns:
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["hour"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek
        df["dom"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

        if "sender_id" in df.columns:
            df = df.sort_values(["sender_id", "timestamp"])
            df["sender_prev_ts"] = df.groupby("sender_id")["timestamp"].shift(1)
            df["sender_gap_hours"] = (df["timestamp"] - df["sender_prev_ts"]).dt.total_seconds() / 3600.0

        if "recipient_id" in df.columns:
            df["pair_key"] = df["sender_id"].astype(str) + "->" + df["recipient_id"].astype(str)
            df["pair_count"] = df.groupby("pair_key")["pair_key"].transform("count")

        return df

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------
    def _transaction_stats(self, tx: pd.DataFrame) -> pd.DataFrame:
        rows = []

        if "amount" in tx.columns:
            rows.append({
                "metric": "amount_mean",
                "value": float(tx["amount"].mean()),
            })
            rows.append({
                "metric": "amount_median",
                "value": float(tx["amount"].median()),
            })
            rows.append({
                "metric": "amount_p95",
                "value": float(tx["amount"].quantile(0.95)),
            })
            rows.append({
                "metric": "amount_p99",
                "value": float(tx["amount"].quantile(0.99)),
            })
            rows.append({
                "metric": "amount_outlier_iqr_frac",
                "value": float(tx.get("amount_outlier_iqr", pd.Series(False)).mean()),
            })

        if "hour" in tx.columns:
            rows.append({"metric": "night_tx_frac_00_06", "value": float(tx["hour"].between(0, 5).mean())})

        if "sender_id" in tx.columns:
            sender_counts = tx["sender_id"].value_counts()
            rows.append({"metric": "sender_count_mean", "value": float(sender_counts.mean())})
            rows.append({"metric": "sender_count_p95", "value": float(sender_counts.quantile(0.95))})

        if "recipient_id" in tx.columns:
            rec_counts = tx["recipient_id"].value_counts()
            rows.append({"metric": "recipient_count_mean", "value": float(rec_counts.mean())})
            rows.append({"metric": "recipient_count_p95", "value": float(rec_counts.quantile(0.95))})

        if "payment_type" in tx.columns:
            for k, v in tx["payment_type"].value_counts().head(20).items():
                rows.append({"metric": f"payment_type::{k}", "value": int(v)})

        return pd.DataFrame(rows)

    def _build_user_profiles(self, tx: pd.DataFrame, users: pd.DataFrame, locations: Dict[str, List[Dict]]) -> pd.DataFrame:
        if "sender_id" not in tx.columns:
            return pd.DataFrame()

        profiles = []

        for uid, sub in tx.groupby("sender_id"):
            amounts = pd.to_numeric(sub["amount"], errors="coerce") if "amount" in sub.columns else pd.Series(dtype=float)
            ts = sub["timestamp"] if "timestamp" in sub.columns else pd.Series(dtype="datetime64[ns]")

            row = {
                "user_id": uid,
                "n_tx_sent": int(len(sub)),
                "n_unique_recipients": int(sub["recipient_id"].nunique()) if "recipient_id" in sub.columns else 0,
                "amount_mean_sent": float(amounts.mean()) if len(amounts) else np.nan,
                "amount_median_sent": float(amounts.median()) if len(amounts) else np.nan,
                "amount_std_sent": float(amounts.std()) if len(amounts) > 1 else 0.0,
                "night_tx_frac": float(sub["hour"].between(0, 5).mean()) if "hour" in sub.columns else np.nan,
                "weekend_tx_frac": float(sub["is_weekend"].mean()) if "is_weekend" in sub.columns else np.nan,
                "burstiness_gap_h_median": float(sub["sender_gap_hours"].median()) if "sender_gap_hours" in sub.columns else np.nan,
                "top_recipient_concentration": float(sub["recipient_id"].value_counts(normalize=True).iloc[0]) if "recipient_id" in sub.columns and len(sub) else np.nan,
            }

            # salary-like inflows estimated later, but here keep behaviour summary
            user_row = {}
            if not users.empty and "user_id" in users.columns:
                hit = users[users["user_id"].astype(str) == str(uid)]
                if not hit.empty:
                    user_row["job"] = str(hit["job"].iloc[0]) if "job" in hit.columns else ""
                    by = safe_int(hit["birth_year"].iloc[0], 0) if "birth_year" in hit.columns else 0
                    age = pd.Timestamp.now().year - by if by else 0
                    user_row["age"] = age
                    user_row["iban"] = str(hit["iban"].iloc[0]) if "iban" in hit.columns else ""

            # location profile
            locs = locations.get(str(uid), [])
            if locs:
                row["has_location"] = 1
                row["n_location_pings"] = len(locs)
            else:
                row["has_location"] = 0
                row["n_location_pings"] = 0

            row.update(user_row)
            profiles.append(row)

        return pd.DataFrame(profiles)

    def _build_counterparty_profiles(self, tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        if "recipient_id" not in tx.columns:
            return pd.DataFrame()

        rows = []
        for rid, sub in tx.groupby("recipient_id"):
            amounts = pd.to_numeric(sub["amount"], errors="coerce") if "amount" in sub.columns else pd.Series(dtype=float)
            rows.append({
                "recipient_id": rid,
                "n_received": int(len(sub)),
                "n_unique_senders": int(sub["sender_id"].nunique()) if "sender_id" in sub.columns else 0,
                "amount_mean_in": float(amounts.mean()) if len(amounts) else np.nan,
                "amount_median_in": float(amounts.median()) if len(amounts) else np.nan,
                "hub_score_unique_senders": int(sub["sender_id"].nunique()) if "sender_id" in sub.columns else 0,
                "repeat_sender_frac": float((sub["sender_id"].value_counts() > 1).mean()) if "sender_id" in sub.columns and len(sub) else np.nan,
            })
        return pd.DataFrame(rows)

    def _detect_salary_patterns(self, tx: pd.DataFrame) -> pd.DataFrame:
        """
        Heuristic salary detector:
        incoming recurring transactions to recipient,
        similar amounts, repeated monthly-ish timing, same sender.
        """
        if not {"recipient_id", "sender_id", "amount", "timestamp"}.issubset(tx.columns):
            return pd.DataFrame()

        df = tx.copy().dropna(subset=["timestamp", "amount"])
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

        rows = []
        for (recipient_id, sender_id), sub in df.groupby(["recipient_id", "sender_id"]):
            if len(sub) < 2:
                continue

            months = sub["year_month"].nunique()
            amt_cv = float(sub["amount"].std() / max(sub["amount"].mean(), 1e-9)) if len(sub) > 1 else 999.0
            dom_std = float(sub["timestamp"].dt.day.std()) if len(sub) > 1 else 999.0
            median_amount = float(sub["amount"].median())

            salary_score = 0.0
            if months >= 2:
                salary_score += 0.35
            if amt_cv <= 0.15:
                salary_score += 0.35
            if dom_std <= 5:
                salary_score += 0.20
            if median_amount > 0:
                salary_score += 0.10

            if salary_score >= 0.5:
                rows.append({
                    "recipient_id": recipient_id,
                    "sender_id": sender_id,
                    "n_tx": int(len(sub)),
                    "n_months": int(months),
                    "median_amount": median_amount,
                    "amount_cv": round(amt_cv, 4),
                    "day_of_month_std": round(dom_std, 4),
                    "salary_like_score": round(salary_score, 4),
                })

        return pd.DataFrame(rows).sort_values("salary_like_score", ascending=False) if rows else pd.DataFrame()

    def _temporal_patterns(self, tx: pd.DataFrame) -> pd.DataFrame:
        rows = []

        if "hour" in tx.columns:
            by_hour = tx.groupby("hour").size().reset_index(name="count")
            by_hour["kind"] = "hour"
            rows.append(by_hour.rename(columns={"hour": "bucket"}))

        if "dow" in tx.columns:
            by_dow = tx.groupby("dow").size().reset_index(name="count")
            by_dow["kind"] = "dow"
            rows.append(by_dow.rename(columns={"dow": "bucket"}))

        if "month" in tx.columns:
            by_month = tx.groupby("month").size().reset_index(name="count")
            by_month["kind"] = "month"
            rows.append(by_month.rename(columns={"month": "bucket"}))

        if rows:
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame(columns=["bucket", "count", "kind"])

    def _location_profiles(self, locations: Dict[str, List[Dict]], users: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for tag, pings in locations.items():
            coords = []
            times = []

            for p in pings:
                lat = safe_float(coalesce(p, ["lat", "Lat", "latitude"], None), None)
                lng = safe_float(coalesce(p, ["lng", "Lng", "lon", "longitude"], None), None)
                if lat is not None and lng is not None:
                    coords.append((lat, lng))
                raw_t = coalesce(p, ["timestamp", "Timestamp", "datetime", "time"], None)
                if raw_t is not None:
                    try:
                        times.append(pd.to_datetime(raw_t))
                    except Exception:
                        pass

            if not coords:
                rows.append({"BioTag": tag, "n_pings": len(pings)})
                continue

            lats = np.array([x[0] for x in coords], dtype=float)
            lngs = np.array([x[1] for x in coords], dtype=float)
            c_lat, c_lng = float(lats.mean()), float(lngs.mean())
            d_cent = haversine(lats, lngs, c_lat, c_lng)

            active_hours = len(set(t.hour for t in times if pd.notna(t))) if times else 0

            rows.append({
                "BioTag": tag,
                "n_pings": len(coords),
                "lat_centroid": round(c_lat, 6),
                "lng_centroid": round(c_lng, 6),
                "mobility_radius_std_km": round(float(np.std(d_cent)) if len(d_cent) > 1 else 0.0, 4),
                "mobility_radius_p90_km": round(float(np.percentile(d_cent, 90)), 4),
                "active_hours": int(active_hours),
            })

        return pd.DataFrame(rows)

    def _message_summary(self, sms: pd.DataFrame, mails: pd.DataFrame) -> pd.DataFrame:
        suspicious_patterns = [
            r"verify",
            r"suspend",
            r"locked",
            r"unusual activity",
            r"urgent",
            r"click",
            r"security",
            r"confirm",
            r"password",
            r"delivery fee",
            r"customs",
        ]

        rows = []

        for source_name, df in [("sms", sms), ("mail", mails)]:
            if df is None or df.empty:
                continue

            text_col = None
            for candidate in ["text", "body", "content", "message", "subject", "sms", "mail"]:
                if candidate in df.columns:
                    text_col = candidate
                    break

            if text_col is None:
                rows.append({"source": source_name, "n_rows": int(len(df)), "suspicious_frac": np.nan})
                continue

            text = df[text_col].fillna("").astype(str).str.lower()
            suspicious = text.str.contains("|".join(suspicious_patterns), regex=True, na=False)
            rows.append({
                "source": source_name,
                "n_rows": int(len(df)),
                "suspicious_frac": float(suspicious.mean()),
            })

        return pd.DataFrame(rows)

    def _graph_analysis(self, tx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph]:
        G = nx.DiGraph()

        if not {"sender_id", "recipient_id"}.issubset(tx.columns):
            return pd.DataFrame(), pd.DataFrame(), G

        tmp = tx.copy()
        tmp["amount"] = pd.to_numeric(tmp["amount"], errors="coerce").fillna(0.0)

        edge_df = (
            tmp.groupby(["sender_id", "recipient_id"], as_index=False)
            .agg(
                n_tx=("amount", "size"),
                total_amount=("amount", "sum"),
                mean_amount=("amount", "mean"),
            )
        )

        for _, r in edge_df.iterrows():
            G.add_edge(
                str(r["sender_id"]),
                str(r["recipient_id"]),
                weight=float(r["total_amount"]),
                n_tx=int(r["n_tx"]),
                mean_amount=float(r["mean_amount"]),
            )

        node_rows = []
        for node in G.nodes():
            out_deg = G.out_degree(node)
            in_deg = G.in_degree(node)
            out_w = sum(G[node][nbr]["weight"] for nbr in G.successors(node))
            in_w = sum(G[pred][node]["weight"] for pred in G.predecessors(node))
            node_rows.append({
                "node": node,
                "out_degree": int(out_deg),
                "in_degree": int(in_deg),
                "weighted_out": float(out_w),
                "weighted_in": float(in_w),
                "total_degree": int(out_deg + in_deg),
            })

        edge_rows = []
        for u, v, d in G.edges(data=True):
            edge_rows.append({
                "sender_id": u,
                "recipient_id": v,
                "n_tx": int(d.get("n_tx", 0)),
                "total_amount": float(d.get("weight", 0.0)),
                "mean_amount": float(d.get("mean_amount", 0.0)),
            })

        return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), G

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    def _make_plots(self, tx: pd.DataFrame, user_profiles: pd.DataFrame, geo_summary: pd.DataFrame, G: nx.DiGraph) -> None:
        self._plot_amount_distribution(tx)
        self._plot_hourly(tx)
        self._plot_dow_hour_heatmap(tx)
        self._plot_top_users(tx)
        self._plot_geo(geo_summary)
        self._plot_graph(G)

    def _plot_amount_distribution(self, tx: pd.DataFrame):
        if "amount" not in tx.columns or tx["amount"].dropna().empty:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(tx["amount"].dropna(), bins=50)
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Amount")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(self.output_dir / "amount_distribution.png")
        plt.close()

    def _plot_hourly(self, tx: pd.DataFrame):
        if "hour" not in tx.columns:
            return
        plt.figure(figsize=(8, 5))
        tx.groupby("hour").size().plot(kind="bar")
        plt.title("Transactions by Hour")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "transactions_by_hour.png")
        plt.close()

    def _plot_dow_hour_heatmap(self, tx: pd.DataFrame):
        if not {"dow", "hour"}.issubset(tx.columns):
            return
        pivot = tx.pivot_table(index="dow", columns="hour", values="amount", aggfunc="size", fill_value=0)
        plt.figure(figsize=(10, 5))
        plt.imshow(pivot.values, aspect="auto")
        plt.title("Transactions Heatmap: Day-of-Week x Hour")
        plt.xlabel("Hour")
        plt.ylabel("Day of Week")
        plt.colorbar(label="Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "dow_hour_heatmap.png")
        plt.close()

    def _plot_top_users(self, tx: pd.DataFrame):
        if "sender_id" in tx.columns:
            plt.figure(figsize=(10, 5))
            tx["sender_id"].value_counts().head(15).plot(kind="bar")
            plt.title("Top Senders by Count")
            plt.xlabel("Sender ID")
            plt.ylabel("Transactions")
            plt.tight_layout()
            plt.savefig(self.output_dir / "top_senders.png")
            plt.close()

        if "recipient_id" in tx.columns:
            plt.figure(figsize=(10, 5))
            tx["recipient_id"].value_counts().head(15).plot(kind="bar")
            plt.title("Top Recipients by Count")
            plt.xlabel("Recipient ID")
            plt.ylabel("Transactions")
            plt.tight_layout()
            plt.savefig(self.output_dir / "top_recipients.png")
            plt.close()

    def _plot_geo(self, geo_summary: pd.DataFrame):
        if geo_summary is None or geo_summary.empty:
            return
        needed = {"lng_centroid", "lat_centroid"}
        if not needed.issubset(geo_summary.columns):
            return

        sub = geo_summary.dropna(subset=["lng_centroid", "lat_centroid"]).copy()
        if sub.empty:
            return

        plt.figure(figsize=(7, 6))
        plt.scatter(sub["lng_centroid"], sub["lat_centroid"], s=np.clip(sub["n_pings"] * 2, 10, 200))
        plt.title("Location Centroids")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(self.output_dir / "location_centroids.png")
        plt.close()

    def _plot_graph(self, G: nx.DiGraph):
        if G.number_of_nodes() == 0:
            return

        # limit to a manageable subgraph
        nodes_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:40]
        H = G.subgraph(nodes_sorted).copy()

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx(
            H,
            pos=pos,
            with_labels=False,
            node_size=[50 + 20 * H.degree(n) for n in H.nodes()],
            width=[0.5 + 0.3 * H[u][v].get("n_tx", 1) for u, v in H.edges()],
            arrows=True,
        )
        plt.title("Transfer Graph (Top Nodes)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "transfer_graph.png")
        plt.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced EDA Agent for AI Agent Challenge datasets")
    parser.add_argument("--dataset", required=True, help="Dataset directory path")
    args = parser.parse_args()

    agent = AdvancedEDAAgent(args.dataset)
    agent.run()


if __name__ == "__main__":
    main()