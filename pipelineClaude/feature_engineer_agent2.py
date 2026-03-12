#!/usr/bin/env python3
"""
feature_engineering_agent.py

Challenge-grade feature engineering agent for transaction-risk / fraud-style datasets.

Expected files if present:
- transactions.csv
- users.json
- locations.json
- sms.json
- mails.json

Outputs:
<dataset_dir>/feature_outputs/
    - features_master.csv
    - sender_profiles.csv
    - recipient_profiles.csv
    - pair_profiles.csv
    - salary_profiles.csv
    - feature_summary.json

Usage:
    python feature_engineering_agent.py --dataset datasets/level_1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_engineering_agent")


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def mad_z(series: pd.Series) -> pd.Series:
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


def robust_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den is None or pd.isna(den) or abs(den) < 1e-9:
        return default
    return float(num / den)


# =============================================================================
# Agent
# =============================================================================

class FeatureEngineeringAgent:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = str(dataset_dir)
        self.output_dir = Path(self.dataset_dir) / "feature_outputs"
        ensure_dir(self.output_dir)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(self) -> None:
        tx = self._load_transactions(Path(self.dataset_dir) / "transactions.csv")
        users = self._load_users(Path(self.dataset_dir) / "users.json")
        locations = self._load_locations(Path(self.dataset_dir) / "locations.json")
        sms = self._load_sms(Path(self.dataset_dir) / "sms.json")
        mails = self._load_mails(Path(self.dataset_dir) / "mails.json")

        if tx.empty:
            raise ValueError("transactions.csv missing or empty")

        tx = self._enrich_transactions(tx, users)
        tx = self._add_time_features(tx)

        sender_profiles = self._sender_profiles(tx, users, locations)
        recipient_profiles = self._recipient_profiles(tx, users)
        pair_profiles = self._pair_profiles(tx)
        salary_profiles = self._salary_profiles(tx)
        graph_profiles = self._graph_profiles(tx)
        location_profiles = self._location_profiles(locations, users)
        message_events = self._message_events(sms, mails)

        features = tx.copy()

        features = self._merge_profiles(
            features,
            sender_profiles,
            recipient_profiles,
            pair_profiles,
            salary_profiles,
            graph_profiles,
            location_profiles,
        )

        features = self._sequential_sender_features(features)
        features = self._sequential_pair_features(features)
        features = self._message_proximity_features(features, message_events)
        features = self._location_consistency_features(features, locations, users)
        features = self._final_cross_features(features)

        # Save
        sender_profiles.to_csv(self.output_dir / "sender_profiles.csv", index=False)
        recipient_profiles.to_csv(self.output_dir / "recipient_profiles.csv", index=False)
        pair_profiles.to_csv(self.output_dir / "pair_profiles.csv", index=False)
        salary_profiles.to_csv(self.output_dir / "salary_profiles.csv", index=False)
        features.to_csv(self.output_dir / "features_master.csv", index=False)

        summary = {
            "dataset_dir": self.dataset_dir,
            "n_transactions": int(len(features)),
            "n_senders": int(features["sender_id"].nunique()) if "sender_id" in features.columns else 0,
            "n_recipients": int(features["recipient_id"].nunique()) if "recipient_id" in features.columns else 0,
            "n_sender_profiles": int(len(sender_profiles)),
            "n_recipient_profiles": int(len(recipient_profiles)),
            "n_pair_profiles": int(len(pair_profiles)),
            "n_salary_profiles": int(len(salary_profiles)),
            "n_message_events": int(len(message_events)),
            "n_feature_columns": int(features.shape[1]),
        }
        save_json(summary, self.output_dir / "feature_summary.json")
        log.info(f"Feature engineering completed -> {self.output_dir}")

    # -------------------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------------------
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

        required = ["timestamp", "sender_id", "recipient_id", "amount"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"transactions.csv missing required column: {c}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["timestamp", "sender_id", "recipient_id", "amount"]).copy()

        for c in ["transaction_id", "sender_id", "recipient_id", "sender_iban", "recipient_iban", "merchant", "payment_type", "status"]:
            if c in df.columns:
                df[c] = df[c].astype(str)

        df = df.sort_values("timestamp").reset_index(drop=True)
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
    # Core enrichments
    # -------------------------------------------------------------------------
    def _enrich_transactions(self, tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        df = tx.copy()

        df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0))
        df["amount_global_mad_z"] = mad_z(df["amount"]).fillna(0.0)

        if not users.empty and "user_id" in users.columns:
            u = users.copy()

            if "sender_id" in df.columns:
                su = u.add_prefix("sender_user_")
                df = df.merge(su, left_on="sender_id", right_on="sender_user_user_id", how="left")

            if "recipient_id" in df.columns:
                ru = u.add_prefix("recipient_user_")
                df = df.merge(ru, left_on="recipient_id", right_on="recipient_user_user_id", how="left")

            if "sender_iban" in df.columns and "iban" in users.columns:
                si = u.add_prefix("sender_iban_user_")
                df = df.merge(si, left_on="sender_iban", right_on="sender_iban_user_iban", how="left")

            if "recipient_iban" in df.columns and "iban" in users.columns:
                ri = u.add_prefix("recipient_iban_user_")
                df = df.merge(ri, left_on="recipient_iban", right_on="recipient_iban_user_iban", how="left")

            if {"sender_iban", "sender_user_iban"}.issubset(df.columns):
                df["sender_iban_consistent"] = (df["sender_iban"] == df["sender_user_iban"]).fillna(False).astype(int)

            if {"recipient_iban", "recipient_user_iban"}.issubset(df.columns):
                df["recipient_iban_consistent"] = (df["recipient_iban"] == df["recipient_user_iban"]).fillna(False).astype(int)

        return df

    def _add_time_features(self, tx: pd.DataFrame) -> pd.DataFrame:
        df = tx.copy().sort_values("timestamp").reset_index(drop=True)

        df["hour"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek
        df["dom"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].between(0, 5).astype(int)
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

        df["pair_key"] = df["sender_id"].astype(str) + "->" + df["recipient_id"].astype(str)

        return df

    # -------------------------------------------------------------------------
    # Profile builders
    # -------------------------------------------------------------------------
    def _sender_profiles(self, tx: pd.DataFrame, users: pd.DataFrame, locations: Dict[str, List[Dict]]) -> pd.DataFrame:
        rows = []

        for uid, sub in tx.groupby("sender_id"):
            sub = sub.sort_values("timestamp")
            amounts = sub["amount"].astype(float)

            row = {
                "sender_id": uid,
                "sender_n_tx": int(len(sub)),
                "sender_n_unique_recipients": int(sub["recipient_id"].nunique()),
                "sender_amount_mean": float(amounts.mean()),
                "sender_amount_median": float(amounts.median()),
                "sender_amount_std": float(amounts.std()) if len(amounts) > 1 else 0.0,
                "sender_amount_p90": float(amounts.quantile(0.90)),
                "sender_amount_p95": float(amounts.quantile(0.95)),
                "sender_amount_max": float(amounts.max()),
                "sender_night_frac": float(sub["is_night"].mean()),
                "sender_weekend_frac": float(sub["is_weekend"].mean()),
                "sender_top_recipient_concentration": float(sub["recipient_id"].value_counts(normalize=True).iloc[0]),
                "sender_active_months": int(sub["year_month"].nunique()),
            }

            # inferred income
            inbound = tx[tx["recipient_id"].astype(str) == str(uid)].copy()
            if not inbound.empty:
                row["sender_n_inbound_tx"] = int(len(inbound))
                row["sender_inbound_amount_mean"] = float(inbound["amount"].mean())
                row["sender_inbound_amount_median"] = float(inbound["amount"].median())
            else:
                row["sender_n_inbound_tx"] = 0
                row["sender_inbound_amount_mean"] = 0.0
                row["sender_inbound_amount_median"] = 0.0

            # users
            if not users.empty and "user_id" in users.columns:
                hit = users[users["user_id"].astype(str) == str(uid)]
                if not hit.empty:
                    row["sender_job"] = str(hit["job"].iloc[0]) if "job" in hit.columns else ""
                    by = safe_int(hit["birth_year"].iloc[0], 0) if "birth_year" in hit.columns else 0
                    age = pd.Timestamp.now().year - by if by else 0
                    row["sender_age"] = age
                    row["sender_has_user"] = 1
                else:
                    row["sender_job"] = ""
                    row["sender_age"] = 0
                    row["sender_has_user"] = 0
            else:
                row["sender_job"] = ""
                row["sender_age"] = 0
                row["sender_has_user"] = 0

            locs = locations.get(str(uid), [])
            row["sender_has_location"] = 1 if locs else 0
            row["sender_location_pings"] = len(locs)

            rows.append(row)

        return pd.DataFrame(rows)

    def _recipient_profiles(self, tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for rid, sub in tx.groupby("recipient_id"):
            amounts = sub["amount"].astype(float)
            rows.append({
                "recipient_id": rid,
                "recipient_n_tx_in": int(len(sub)),
                "recipient_n_unique_senders": int(sub["sender_id"].nunique()),
                "recipient_amount_mean_in": float(amounts.mean()),
                "recipient_amount_median_in": float(amounts.median()),
                "recipient_amount_std_in": float(amounts.std()) if len(amounts) > 1 else 0.0,
                "recipient_amount_p95_in": float(amounts.quantile(0.95)),
                "recipient_night_frac_in": float(sub["is_night"].mean()),
                "recipient_weekend_frac_in": float(sub["is_weekend"].mean()),
                "recipient_active_months_in": int(sub["year_month"].nunique()),
            })

        return pd.DataFrame(rows)

    def _pair_profiles(self, tx: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (sid, rid), sub in tx.groupby(["sender_id", "recipient_id"]):
            sub = sub.sort_values("timestamp")
            amounts = sub["amount"].astype(float)

            if len(sub) >= 2:
                gaps = sub["timestamp"].diff().dt.total_seconds().dropna() / 3600.0
                median_gap_h = float(gaps.median()) if len(gaps) else np.nan
            else:
                median_gap_h = np.nan

            rows.append({
                "sender_id": sid,
                "recipient_id": rid,
                "pair_n_tx": int(len(sub)),
                "pair_total_amount": float(amounts.sum()),
                "pair_mean_amount": float(amounts.mean()),
                "pair_median_amount": float(amounts.median()),
                "pair_std_amount": float(amounts.std()) if len(amounts) > 1 else 0.0,
                "pair_active_months": int(sub["year_month"].nunique()),
                "pair_median_gap_h": median_gap_h,
                "pair_last_tx_ts": sub["timestamp"].max(),
            })

        return pd.DataFrame(rows)

    def _salary_profiles(self, tx: pd.DataFrame) -> pd.DataFrame:
        if not {"recipient_id", "sender_id", "amount", "timestamp"}.issubset(tx.columns):
            return pd.DataFrame(columns=["recipient_id", "salary_like_sender_id", "salary_like_score", "salary_median_amount"])

        rows = []
        df = tx.copy()
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

        for (rid, sid), sub in df.groupby(["recipient_id", "sender_id"]):
            if len(sub) < 2:
                continue

            months = sub["year_month"].nunique()
            amt_cv = float(sub["amount"].std() / max(sub["amount"].mean(), 1e-9)) if len(sub) > 1 else 999.0
            dom_std = float(sub["timestamp"].dt.day.std()) if len(sub) > 1 else 999.0
            median_amount = float(sub["amount"].median())

            score = 0.0
            if months >= 2:
                score += 0.35
            if amt_cv <= 0.15:
                score += 0.35
            if dom_std <= 5:
                score += 0.20
            if median_amount > 0:
                score += 0.10

            if score >= 0.5:
                rows.append({
                    "recipient_id": rid,
                    "salary_like_sender_id": sid,
                    "salary_like_score": round(score, 4),
                    "salary_median_amount": median_amount,
                    "salary_n_months": int(months),
                    "salary_amount_cv": round(amt_cv, 4),
                    "salary_dom_std": round(dom_std, 4),
                })

        if not rows:
            return pd.DataFrame(columns=["recipient_id", "salary_like_sender_id", "salary_like_score", "salary_median_amount"])

        sal = pd.DataFrame(rows).sort_values(["recipient_id", "salary_like_score"], ascending=[True, False])
        sal = sal.groupby("recipient_id", as_index=False).first()
        return sal

    def _graph_profiles(self, tx: pd.DataFrame) -> pd.DataFrame:
        G = nx.DiGraph()
        tmp = tx.groupby(["sender_id", "recipient_id"], as_index=False).agg(
            n_tx=("amount", "size"),
            total_amount=("amount", "sum"),
        )

        for _, r in tmp.iterrows():
            G.add_edge(str(r["sender_id"]), str(r["recipient_id"]), weight=float(r["total_amount"]), n_tx=int(r["n_tx"]))

        rows = []
        for node in G.nodes():
            rows.append({
                "node_id": node,
                "graph_out_degree": int(G.out_degree(node)),
                "graph_in_degree": int(G.in_degree(node)),
                "graph_total_degree": int(G.degree(node)),
                "graph_weighted_out": float(sum(G[node][nbr]["weight"] for nbr in G.successors(node))),
                "graph_weighted_in": float(sum(G[pred][node]["weight"] for pred in G.predecessors(node))),
            })

        return pd.DataFrame(rows)

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
                rows.append({"bio_tag": tag, "loc_n_pings": len(pings)})
                continue

            lats = np.array([x[0] for x in coords], dtype=float)
            lngs = np.array([x[1] for x in coords], dtype=float)
            c_lat, c_lng = float(lats.mean()), float(lngs.mean())
            d_cent = haversine(lats, lngs, c_lat, c_lng)

            rows.append({
                "bio_tag": tag,
                "loc_n_pings": len(coords),
                "loc_lat_centroid": c_lat,
                "loc_lng_centroid": c_lng,
                "loc_radius_std_km": float(np.std(d_cent)) if len(d_cent) > 1 else 0.0,
                "loc_radius_p90_km": float(np.percentile(d_cent, 90)),
                "loc_active_hours": len(set(t.hour for t in times if pd.notna(t))) if times else 0,
            })

        return pd.DataFrame(rows)

    def _message_events(self, sms: pd.DataFrame, mails: pd.DataFrame) -> pd.DataFrame:
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
            r"bank",
            r"fraud",
            r"card blocked",
        ]

        rows = []

        def parse_one(source_name: str, df: pd.DataFrame):
            if df is None or df.empty:
                return

            text_col = None
            for candidate in ["text", "body", "content", "message", "subject", "sms", "mail"]:
                if candidate in df.columns:
                    text_col = candidate
                    break

            ts_col = None
            for candidate in ["timestamp", "datetime", "date", "time", "created_at"]:
                if candidate in df.columns:
                    ts_col = candidate
                    break

            id_col = None
            for candidate in ["user_id", "recipient_id", "sender_id", "BioTag", "biotag"]:
                if candidate in df.columns:
                    id_col = candidate
                    break

            # text_col is mandatory; ts_col and id_col are optional (skip per-user linking if absent)
            if text_col is None:
                return

            # If we have no timestamp or id, still extract suspicious keyword signal at row level
            if ts_col is None or id_col is None:
                tmp = df.copy()
                tmp["text_norm"] = tmp[text_col].fillna("").astype(str).str.lower()
                tmp["is_suspicious_msg"] = tmp["text_norm"].str.contains(
                    "|".join(suspicious_patterns), regex=True, na=False
                ).astype(int)
                # assign a synthetic timestamp so we can still log them
                tmp["event_ts"] = pd.Timestamp("2087-01-01")
                tmp["entity_id"] = "GLOBAL"
                for _, r in tmp.iterrows():
                    rows.append({
                        "source": source_name,
                        "entity_id": "GLOBAL",
                        "event_ts": r["event_ts"],
                        "is_suspicious_msg": int(r["is_suspicious_msg"]),
                    })
                return

            tmp = df.copy()
            tmp["event_ts"] = pd.to_datetime(tmp[ts_col], errors="coerce")
            tmp["entity_id"] = tmp[id_col].astype(str)
            tmp["text_norm"] = tmp[text_col].fillna("").astype(str).str.lower()
            tmp = tmp.dropna(subset=["event_ts"])

            tmp["is_suspicious_msg"] = tmp["text_norm"].str.contains("|".join(suspicious_patterns), regex=True, na=False).astype(int)

            for _, r in tmp.iterrows():
                rows.append({
                    "source": source_name,
                    "entity_id": str(r["entity_id"]),
                    "event_ts": r["event_ts"],
                    "is_suspicious_msg": int(r["is_suspicious_msg"]),
                })

        parse_one("sms", sms)
        parse_one("mail", mails)

        if not rows:
            return pd.DataFrame(columns=["source", "entity_id", "event_ts", "is_suspicious_msg"])

        return pd.DataFrame(rows).sort_values("event_ts").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Merge profiles
    # -------------------------------------------------------------------------
    def _merge_profiles(
        self,
        df: pd.DataFrame,
        sender_profiles: pd.DataFrame,
        recipient_profiles: pd.DataFrame,
        pair_profiles: pd.DataFrame,
        salary_profiles: pd.DataFrame,
        graph_profiles: pd.DataFrame,
        location_profiles: pd.DataFrame,
    ) -> pd.DataFrame:
        out = df.copy()

        if not sender_profiles.empty:
            out = out.merge(sender_profiles, on="sender_id", how="left")

        if not recipient_profiles.empty:
            out = out.merge(recipient_profiles, on="recipient_id", how="left")

        if not pair_profiles.empty:
            out = out.merge(pair_profiles.drop(columns=["pair_last_tx_ts"], errors="ignore"), on=["sender_id", "recipient_id"], how="left")

        if not salary_profiles.empty:
            out = out.merge(salary_profiles, left_on="sender_id", right_on="recipient_id", how="left", suffixes=("", "_salary_sender"))
            out = out.rename(columns={
                "salary_like_sender_id": "sender_salary_like_inflow_from",
                "salary_like_score": "sender_salary_like_score",
                "salary_median_amount": "sender_salary_median_amount",
                "salary_n_months": "sender_salary_n_months",
                "salary_amount_cv": "sender_salary_amount_cv",
                "salary_dom_std": "sender_salary_dom_std",
            })
            out = out.drop(columns=["recipient_id_salary_sender"], errors="ignore")

        if not graph_profiles.empty:
            gp_sender = graph_profiles.add_prefix("sender_")
            gp_sender = gp_sender.rename(columns={"sender_node_id": "sender_id"})
            out = out.merge(gp_sender, on="sender_id", how="left")

            gp_rec = graph_profiles.add_prefix("recipient_")
            gp_rec = gp_rec.rename(columns={"recipient_node_id": "recipient_id"})
            out = out.merge(gp_rec, on="recipient_id", how="left")

        if not location_profiles.empty:
            lp = location_profiles.rename(columns={"bio_tag": "sender_id"})
            out = out.merge(lp.add_prefix("sender_loc_").rename(columns={"sender_loc_sender_id": "sender_id"}), on="sender_id", how="left")

            lp2 = location_profiles.rename(columns={"bio_tag": "recipient_id"})
            out = out.merge(lp2.add_prefix("recipient_loc_").rename(columns={"recipient_loc_recipient_id": "recipient_id"}), on="recipient_id", how="left")

        return out

    # -------------------------------------------------------------------------
    # Sequential features
    # -------------------------------------------------------------------------
    def _sequential_sender_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["sender_id", "timestamp"]).copy()

        out["sender_prev_ts"] = out.groupby("sender_id")["timestamp"].shift(1)
        out["sender_gap_hours"] = (out["timestamp"] - out["sender_prev_ts"]).dt.total_seconds() / 3600.0

        out["sender_tx_count_before"] = out.groupby("sender_id").cumcount()

        # first time recipient for sender
        out["sender_seen_pair_before"] = out.groupby(["sender_id", "recipient_id"]).cumcount()
        out["is_new_recipient_for_sender"] = (out["sender_seen_pair_before"] == 0).astype(int)

        # expanding baselines
        grp = out.groupby("sender_id")["amount"]
        out["sender_expanding_mean_before"] = grp.transform(lambda s: s.shift(1).expanding().mean())
        out["sender_expanding_std_before"] = grp.transform(lambda s: s.shift(1).expanding().std())
        out["sender_expanding_median_before"] = grp.transform(lambda s: s.shift(1).expanding().median())

        out["amount_vs_sender_mean"] = out["amount"] - out["sender_expanding_mean_before"]
        out["amount_over_sender_mean_ratio"] = out.apply(
            lambda r: robust_ratio(r["amount"], r["sender_expanding_mean_before"], default=1.0), axis=1
        )
        out["amount_sender_mad_like_z"] = out.groupby("sender_id")["amount"].transform(lambda s: mad_z(s)).fillna(0.0)

        # rolling windows by sender — corrected implementation with proper index tracking
        # Reset index first so integer indices are contiguous and assignment is unambiguous
        out = out.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
        out["sender_tx_last_1h"] = 0
        out["sender_tx_last_24h"] = 0
        out["sender_tx_last_7d"] = 0
        out["sender_amt_last_24h"] = 0.0
        out["sender_unique_recips_last_7d"] = 0

        _one_h   = np.timedelta64(3_600,      "s")
        _one_d   = np.timedelta64(86_400,     "s")
        _seven_d = np.timedelta64(7 * 86_400, "s")

        for _sid, _grp_idx in out.groupby("sender_id").groups.items():
            _grp = out.loc[_grp_idx].sort_values("timestamp").copy()
            _n = len(_grp)
            if _n == 0:
                continue

            # Save the (reset) integer indices in timestamp order for write-back
            _orig_idx = _grp.index.tolist()
            _ts  = _grp["timestamp"].values
            _amt = _grp["amount"].values.astype(float)
            _rid = _grp["recipient_id"].values

            _cnt_1h  = np.zeros(_n, dtype=np.int32)
            _cnt_24h = np.zeros(_n, dtype=np.int32)
            _cnt_7d  = np.zeros(_n, dtype=np.int32)
            _sum_24h = np.zeros(_n, dtype=np.float64)
            _uniq_7d = np.zeros(_n, dtype=np.int32)

            for _i in range(_n):
                _t    = _ts[_i]
                _seen = set()
                for _j in range(_i - 1, -1, -1):   # walk backwards (sorted by ts)
                    _dt = _t - _ts[_j]
                    if _dt > _seven_d:
                        break                        # nothing older can fit
                    _cnt_7d[_i] += 1
                    _seen.add(str(_rid[_j]))
                    if _dt <= _one_d:
                        _cnt_24h[_i] += 1
                        _sum_24h[_i] += _amt[_j]
                        if _dt <= _one_h:
                            _cnt_1h[_i] += 1
                _uniq_7d[_i] = len(_seen)

            out.loc[_orig_idx, "sender_tx_last_1h"]           = _cnt_1h
            out.loc[_orig_idx, "sender_tx_last_24h"]          = _cnt_24h
            out.loc[_orig_idx, "sender_tx_last_7d"]           = _cnt_7d
            out.loc[_orig_idx, "sender_amt_last_24h"]         = _sum_24h
            out.loc[_orig_idx, "sender_unique_recips_last_7d"] = _uniq_7d

        return out.sort_index()

    def _sequential_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["sender_id", "recipient_id", "timestamp"]).copy()

        out["pair_prev_ts"] = out.groupby(["sender_id", "recipient_id"])["timestamp"].shift(1)
        out["pair_gap_hours"] = (out["timestamp"] - out["pair_prev_ts"]).dt.total_seconds() / 3600.0
        out["pair_tx_count_before"] = out.groupby(["sender_id", "recipient_id"]).cumcount()

        grp = out.groupby(["sender_id", "recipient_id"])["amount"]
        out["pair_expanding_mean_before"] = grp.transform(lambda s: s.shift(1).expanding().mean())
        out["pair_expanding_std_before"] = grp.transform(lambda s: s.shift(1).expanding().std())
        out["amount_over_pair_mean_ratio"] = out.apply(
            lambda r: robust_ratio(r["amount"], r["pair_expanding_mean_before"], default=1.0), axis=1
        )

        return out.sort_index()

    def _message_proximity_features(self, df: pd.DataFrame, msg_events: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["suspicious_msg_prev_24h_sender"] = 0
        out["suspicious_msg_prev_72h_sender"] = 0
        out["any_msg_prev_24h_sender"] = 0

        if msg_events.empty:
            return out

        msg_events = msg_events.sort_values("event_ts").copy()

        events_by_entity = {
            eid: sub.copy()
            for eid, sub in msg_events.groupby("entity_id")
        }

        values_24_susp = []
        values_72_susp = []
        values_24_any = []

        for _, row in out.iterrows():
            sender = str(row["sender_id"])
            ts = row["timestamp"]

            sub = events_by_entity.get(sender)
            if sub is None or sub.empty:
                values_24_susp.append(0)
                values_72_susp.append(0)
                values_24_any.append(0)
                continue

            prev24 = sub[(sub["event_ts"] < ts) & (sub["event_ts"] >= ts - pd.Timedelta(hours=24))]
            prev72 = sub[(sub["event_ts"] < ts) & (sub["event_ts"] >= ts - pd.Timedelta(hours=72))]

            values_24_susp.append(int(prev24["is_suspicious_msg"].sum()))
            values_72_susp.append(int(prev72["is_suspicious_msg"].sum()))
            values_24_any.append(int(len(prev24)))

        out["suspicious_msg_prev_24h_sender"] = values_24_susp
        out["suspicious_msg_prev_72h_sender"] = values_72_susp
        out["any_msg_prev_24h_sender"] = values_24_any

        return out

    def _location_consistency_features(self, df: pd.DataFrame, locations: Dict[str, List[Dict]], users: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        loc_map = {}
        for tag, pings in locations.items():
            coords = []
            for p in pings:
                lat = safe_float(coalesce(p, ["lat", "Lat", "latitude"], None), None)
                lng = safe_float(coalesce(p, ["lng", "Lng", "lon", "longitude"], None), None)
                if lat is not None and lng is not None:
                    coords.append((lat, lng))
            if coords:
                lats = np.array([x[0] for x in coords], dtype=float)
                lngs = np.array([x[1] for x in coords], dtype=float)
                loc_map[str(tag)] = {
                    "centroid_lat": float(lats.mean()),
                    "centroid_lng": float(lngs.mean()),
                    "radius_p90": float(np.percentile(haversine(lats, lngs, lats.mean(), lngs.mean()), 90)),
                }

        # home from users, if present
        user_home = {}
        if not users.empty:
            for _, r in users.iterrows():
                uid = str(r["user_id"]) if "user_id" in users.columns else None
                if uid is None:
                    continue
                res = r.get("residence", None)
                if isinstance(res, dict):
                    lat = safe_float(coalesce(res, ["lat", "latitude"], None), None)
                    lng = safe_float(coalesce(res, ["lng", "lon", "longitude"], None), None)
                    if lat is not None and lng is not None:
                        user_home[uid] = (lat, lng)

        out["sender_has_loc_profile"] = out["sender_id"].astype(str).map(lambda x: 1 if x in loc_map else 0)
        out["sender_loc_radius_p90"] = out["sender_id"].astype(str).map(lambda x: loc_map.get(x, {}).get("radius_p90", np.nan))

        # no exact tx location exists, so this block mostly exposes profile completeness / plausibility hooks
        out["sender_has_home"] = out["sender_id"].astype(str).map(lambda x: 1 if x in user_home else 0)

        return out

    # -------------------------------------------------------------------------
    # Final cross features
    # -------------------------------------------------------------------------
    def _final_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # amount anomaly vs sender baseline
        out["amount_gt_sender_p95"] = (
            (out["amount"] > out["sender_amount_p95"].fillna(np.inf))
        ).astype(int)

        out["amount_gt_3x_sender_mean"] = (
            out["amount_over_sender_mean_ratio"].fillna(0) > 3.0
        ).astype(int)

        out["amount_gt_3x_pair_mean"] = (
            out["amount_over_pair_mean_ratio"].fillna(0) > 3.0
        ).astype(int)

        # novelty-risk compound
        out["new_recipient_high_amount_flag"] = (
            (out["is_new_recipient_for_sender"] == 1)
            & (out["amount_gt_sender_p95"] == 1)
        ).astype(int)

        # burst-risk compound
        out["burst_1h_flag"] = (out["sender_tx_last_1h"].fillna(0) >= 3).astype(int)
        out["burst_24h_flag"] = (out["sender_tx_last_24h"].fillna(0) >= 5).astype(int)

        # night + novelty + amount
        out["night_new_high_flag"] = (
            (out["is_night"] == 1)
            & (out["is_new_recipient_for_sender"] == 1)
            & (out["amount_gt_sender_p95"] == 1)
        ).astype(int)

        # hub recipient
        out["recipient_is_hub"] = (
            out["recipient_n_unique_senders"].fillna(0) >= out["recipient_n_unique_senders"].fillna(0).quantile(0.90)
            if "recipient_n_unique_senders" in out.columns else 0
        ).astype(int) if "recipient_n_unique_senders" in out.columns else 0

        # spending vs salary
        out["sender_amount_vs_salary_ratio"] = out.apply(
            lambda r: robust_ratio(r["amount"], r.get("sender_salary_median_amount", np.nan), default=0.0),
            axis=1,
        )

        out["amount_gt_50pct_salary"] = (out["sender_amount_vs_salary_ratio"] > 0.5).astype(int)
        out["amount_gt_salary"] = (out["sender_amount_vs_salary_ratio"] > 1.0).astype(int)

        # msg-triggered behaviour
        out["msg_triggered_tx_flag"] = (
            (out["suspicious_msg_prev_24h_sender"].fillna(0) > 0)
            & (out["amount_gt_sender_p95"] == 1)
        ).astype(int)

        # graph asymmetry
        out["sender_out_in_degree_ratio"] = out.apply(
            lambda r: robust_ratio(r.get("sender_graph_out_degree", 0), r.get("sender_graph_in_degree", 0), default=0.0),
            axis=1,
        )

        out["recipient_in_out_degree_ratio"] = out.apply(
            lambda r: robust_ratio(r.get("recipient_graph_in_degree", 0), r.get("recipient_graph_out_degree", 0), default=0.0),
            axis=1,
        )

        # pair concentration
        out["pair_share_of_sender_volume"] = out.apply(
            lambda r: robust_ratio(r.get("pair_total_amount", 0.0), r.get("sender_amount_mean", 0.0) * max(r.get("sender_n_tx", 0), 1), default=0.0),
            axis=1,
        )

        # cleanup timestamps if you want CSV cleaner
        for c in ["sender_prev_ts", "pair_prev_ts"]:
            if c in out.columns:
                out[c] = out[c].astype(str)

        return out


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Challenge-grade feature engineering agent")
    parser.add_argument("--dataset", required=True, help="Dataset directory path")
    args = parser.parse_args()

    agent = FeatureEngineeringAgent(args.dataset)
    agent.run()


if __name__ == "__main__":
    main()