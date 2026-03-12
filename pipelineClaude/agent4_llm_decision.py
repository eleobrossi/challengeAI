#!/usr/bin/env python3
"""
agent4_llm_decision.py

LLM-based final decision agent for the Reply Mirror AI Agent Challenge.

Reads the statistical anomaly scores produced by anomaly_score_agent3.py and
enriches the top-suspicious candidates with raw transaction context, then calls
an LLM via OpenRouter to make final fraud / legitimate decisions.

All LLM calls are tracked in Langfuse under the shared pipeline session_id so
that token usage and costs appear as a single session in the challenge dashboard.

Input:
    <dataset_dir>/anomaly_outputs/anomaly_scores.csv   (from agent3)
    <dataset_dir>/transactions.csv                     (raw, for context)

Output:
    <dataset_dir>/submission.txt   – one fraudulent transaction_id per line (ASCII)
    <dataset_dir>/llm_decisions/
        decision_detail.csv        – per-transaction LLM decision + explanation
        decision_summary.json      – counts / cost summary

Usage:
    python agent4_llm_decision.py --dataset datasets/level_1
    python agent4_llm_decision.py --dataset datasets/level_1 --session-id MY_TEAM-01JXXX
    python agent4_llm_decision.py --dataset datasets/level_1 --threshold 0.30 --batch-size 15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent4_llm_decision")

# ---------------------------------------------------------------------------
# Langfuse + LangChain imports
# ---------------------------------------------------------------------------
try:
    import ulid
    from langfuse import Langfuse, observe
    from langfuse.langchain import CallbackHandler
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    _LANGFUSE_AVAILABLE = True
except ImportError as _e:
    log.warning(f"Langfuse/LangChain not installed ({_e}). LLM calls disabled.")
    _LANGFUSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "team")
    if _LANGFUSE_AVAILABLE:
        return f"{team}-{ulid.new().str}"
    import uuid
    return f"{team}-{uuid.uuid4().hex[:20].upper()}"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a world-class financial fraud detection expert at MirrorPay, 2087.

Your task: analyze a batch of transactions that were flagged as suspicious by a
statistical anomaly detector, and determine which ones are truly fraudulent.

The Mirror Hackers use evolving tactics:
- Targeting new merchants / recipients
- Shifting to unusual hours (midnight – 5 AM)
- High-velocity bursts of transactions
- Amounts that drain the sender's balance
- Transactions preceded by phishing SMS / emails

Scoring asymmetry:
- A missed fraud (false negative) causes direct financial loss — HIGH cost
- A blocked legitimate transaction (false positive) causes reputational/economic damage — MEDIUM cost
- Therefore: lean toward flagging when evidence is ambiguous but significant

For each batch you receive a JSON array. Each element has:
  transaction_id, sender_id, recipient_id, transaction_type, amount,
  timestamp, anomaly_score (0–1), risk_tier (LOW/MEDIUM/HIGH/CRITICAL),
  and the top statistical signals that drove the score.

Respond ONLY with a valid JSON object like:
{
  "fraudulent_ids": ["id1", "id2", ...],
  "reasoning": "brief 1-2 sentence explanation"
}

Rules:
- Include in fraudulent_ids ONLY transaction_ids from the input batch.
- Do NOT include transactions you are confident are legitimate (low score, routine pattern).
- CRITICAL / HIGH risk_tier items should be flagged unless there is a very clear
  legitimate explanation (e.g., known recurring salary, consistent behaviour).
"""


def _format_batch_prompt(rows: List[Dict]) -> str:
    """Format a batch of transaction dicts into an LLM-readable JSON prompt."""
    items = []
    for r in rows:
        sig_keys = [k for k in r.keys() if k.startswith("sig_") and float(r.get(k, 0) or 0) > 0.3]
        top_sigs = sorted(sig_keys, key=lambda k: float(r.get(k, 0) or 0), reverse=True)[:5]
        sig_summary = {k.replace("sig_", ""): round(float(r.get(k, 0) or 0), 3) for k in top_sigs}

        items.append({
            "transaction_id": str(r.get("transaction_id", "")),
            "sender_id": str(r.get("sender_id", "")),
            "recipient_id": str(r.get("recipient_id", "")),
            "transaction_type": str(r.get("transaction_type", "")),
            "amount": round(float(r.get("amount", 0) or 0), 2),
            "timestamp": str(r.get("timestamp", "")),
            "anomaly_score": round(float(r.get("anomaly_score", 0) or 0), 4),
            "risk_tier": str(r.get("risk_tier", "MEDIUM")),
            "top_signals": sig_summary,
        })
    return json.dumps(items, indent=2)


def _parse_llm_response(text: str) -> Tuple[List[str], str]:
    """
    Extract fraudulent_ids list from LLM response text.
    Handles both clean JSON and JSON embedded in markdown code blocks.
    Returns (fraudulent_ids, reasoning).
    """
    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).strip()
    clean = clean.rstrip("`").strip()

    # Try to find first JSON object
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        log.warning("No JSON object found in LLM response.")
        return [], ""

    try:
        obj = json.loads(m.group())
        ids = obj.get("fraudulent_ids", [])
        reason = obj.get("reasoning", "")
        return [str(i) for i in ids], str(reason)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error in LLM response: {e}\nRaw: {text[:200]}")
        return [], ""


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class LLMDecisionAgent:
    """
    Uses an LLM (via OpenRouter) + Langfuse tracing to make final fraud decisions
    on the top-suspicious transactions from the statistical anomaly scorer.
    """

    def __init__(
        self,
        dataset_dir: str,
        session_id: Optional[str] = None,
        batch_size: int = 15,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir  = self.dataset_dir / "llm_decisions"
        self.batch_size  = batch_size
        ensure_dir(self.output_dir)

        # Session ID — can be provided by the pipeline orchestrator or generated here
        self.session_id = session_id or generate_session_id()
        log.info(f"LLM agent session_id: {self.session_id}")

        # Init Langfuse client
        if _LANGFUSE_AVAILABLE:
            self.langfuse_client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
            )
            self.model = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                model=os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"),
                temperature=0.1,
                max_tokens=1024,
            )
        else:
            self.langfuse_client = None
            self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, threshold: float = 0.30) -> List[str]:
        """
        Run the LLM decision pipeline.

        Args:
            threshold: minimum anomaly_score to pass to the LLM (0-1).
                       Transactions below this threshold are considered legitimate.

        Returns:
            List of fraudulent transaction_ids.
        """
        scores_df = self._load_scores()
        raw_tx    = self._load_raw_transactions()

        if scores_df.empty:
            log.error("No anomaly scores found. Run anomaly_score_agent3 first.")
            return []

        # Merge raw transaction context onto scores
        if not raw_tx.empty and "transaction_id" in scores_df.columns and "transaction_id" in raw_tx.columns:
            extra_cols = [c for c in ["description", "location", "payment_method", "sender_iban", "recipient_iban", "balance_after"]
                          if c in raw_tx.columns and c not in scores_df.columns]
            if extra_cols:
                scores_df = scores_df.merge(
                    raw_tx[["transaction_id"] + extra_cols],
                    on="transaction_id",
                    how="left",
                )

        # Filter to candidates above threshold
        candidates = scores_df[
            scores_df["anomaly_score"].fillna(0) >= threshold
        ].copy()

        # Always include CRITICAL tier regardless of threshold
        critical = scores_df[scores_df.get("risk_tier", pd.Series(dtype=str)) == "CRITICAL"].copy()
        candidates = pd.concat([candidates, critical]).drop_duplicates(subset=["transaction_id"])

        log.info(f"Candidates for LLM review: {len(candidates):,} "
                 f"(threshold={threshold}, total={len(scores_df):,})")

        if candidates.empty:
            log.warning("No candidates to review. Lowering threshold to 0.20.")
            candidates = scores_df.nlargest(max(20, len(scores_df) // 5), "anomaly_score")

        # Process in batches
        all_fraudulent: List[str] = []
        decision_rows: List[Dict] = []
        rows = candidates.to_dict(orient="records")
        batches = [rows[i: i + self.batch_size] for i in range(0, len(rows), self.batch_size)]

        log.info(f"Processing {len(batches)} batch(es) of up to {self.batch_size} transactions.")

        for batch_num, batch in enumerate(batches, 1):
            log.info(f"  Batch {batch_num}/{len(batches)} ({len(batch)} txns)…")
            try:
                fraud_ids, reasoning = self._call_llm(batch)
            except Exception as exc:
                log.error(f"  LLM call failed for batch {batch_num}: {exc}")
                # Fallback: adaptive percentile-based heuristic
                fraud_ids = self._heuristic_fallback(batch)
                reasoning = "fallback-heuristic (LLM unavailable)"

            all_fraudulent.extend(fraud_ids)

            # Record decisions per transaction
            fraud_set = set(fraud_ids)
            for r in batch:
                tid = str(r.get("transaction_id", ""))
                decision_rows.append({
                    "transaction_id": tid,
                    "anomaly_score":  round(float(r.get("anomaly_score", 0) or 0), 4),
                    "risk_tier":      str(r.get("risk_tier", "")),
                    "llm_flagged":    int(tid in fraud_set),
                    "batch":          batch_num,
                    "reasoning":      reasoning,
                })

        # Dedup (should not happen but just in case)
        all_fraudulent = list(dict.fromkeys(all_fraudulent))

        # Save outputs
        self._write_submission(all_fraudulent)
        pd.DataFrame(decision_rows).to_csv(self.output_dir / "decision_detail.csv", index=False)
        summary = {
            "session_id":         self.session_id,
            "n_candidates":       int(len(candidates)),
            "n_flagged_by_llm":   len(all_fraudulent),
            "n_total_in_dataset": int(len(scores_df)),
            "threshold_used":     threshold,
            "flagged_fraction":   round(len(all_fraudulent) / max(len(scores_df), 1), 4),
        }
        save_json(summary, self.output_dir / "decision_summary.json")

        if self.langfuse_client:
            self.langfuse_client.flush()

        log.info(f"LLM decision complete. Flagged {len(all_fraudulent)} transactions.")
        log.info(f"Submission written to {self.dataset_dir / 'submission.txt'}")
        return all_fraudulent

    # ------------------------------------------------------------------
    # LLM call (with Langfuse tracing via @observe + CallbackHandler)
    # ------------------------------------------------------------------

    def _call_llm(self, batch: List[Dict]) -> Tuple[List[str], str]:
        """Call the LLM for a single batch and return (fraud_ids, reasoning)."""
        if not _LANGFUSE_AVAILABLE or self.model is None:
            # Pure heuristic fallback when LLM/Langfuse not available
            return self._heuristic_fallback(batch), "heuristic-fallback"

        return self._call_llm_traced(batch)

    @observe()
    def _call_llm_traced(self, batch: List[Dict]) -> Tuple[List[str], str]:
        """@observe-decorated LLM call so Langfuse captures every generation."""
        # Tag this trace with the pipeline session so all batches aggregate together
        # Langfuse 3.x: update_current_trace is a method on the Langfuse instance
        self.langfuse_client.update_current_trace(
            session_id=self.session_id,
            name="fraud_decision_batch",
            tags=["fraud-detection", "batch-decision"],
        )

        langfuse_handler = CallbackHandler()

        batch_prompt = _format_batch_prompt(batch)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze these {len(batch)} transactions:\n\n{batch_prompt}"),
        ]

        response = self.model.invoke(
            messages,
            config={"callbacks": [langfuse_handler]},
        )

        return _parse_llm_response(response.content)

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_fallback(batch: List[Dict]) -> List[str]:
        """
        Rule-based fallback when LLM is unavailable.
        Flags transactions in the top 30% by anomaly_score within this batch,
        always including CRITICAL tier and any score >= 0.50.
        """
        if not batch:
            return []

        scores = [float(r.get("anomaly_score", 0) or 0) for r in batch]
        p70 = float(np.percentile(scores, 70)) if len(scores) > 3 else 0.0
        # Adaptive threshold: at least p70, but no lower than 0.30
        adaptive_threshold = max(p70, 0.30)

        flagged = []
        for r in batch:
            score = float(r.get("anomaly_score", 0) or 0)
            tier  = str(r.get("risk_tier", ""))
            if score >= adaptive_threshold or tier in ("CRITICAL", "HIGH"):
                flagged.append(str(r["transaction_id"]))
        return flagged

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_scores(self) -> pd.DataFrame:
        path = self.dataset_dir / "anomaly_outputs" / "anomaly_scores.csv"
        if not path.exists():
            log.error(f"anomaly_scores.csv not found at {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False)
        if "anomaly_score" in df.columns:
            df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0)
        return df

    def _load_raw_transactions(self) -> pd.DataFrame:
        path = self.dataset_dir / "transactions.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Output writer
    # ------------------------------------------------------------------

    def _write_submission(self, fraud_ids: List[str]) -> None:
        output_path = self.dataset_dir / "submission.txt"
        with open(output_path, "w", encoding="ascii", errors="replace") as f:
            for tid in fraud_ids:
                f.write(tid + "\n")
        log.info(f"submission.txt written: {len(fraud_ids)} lines → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Decision Agent – Reply Mirror AI Agent Challenge"
    )
    parser.add_argument("--dataset",    required=True,  help="Dataset directory (e.g. datasets/level_1)")
    parser.add_argument("--session-id", default=None,   help="Langfuse session ID (generated if omitted)")
    parser.add_argument("--threshold",  type=float, default=0.30,
                        help="Min anomaly_score to pass to LLM (default: 0.30)")
    parser.add_argument("--batch-size", type=int,   default=15,
                        help="Transactions per LLM batch (default: 15)")
    args = parser.parse_args()

    agent = LLMDecisionAgent(
        dataset_dir=args.dataset,
        session_id=args.session_id,
        batch_size=args.batch_size,
    )
    fraud_ids = agent.run(threshold=args.threshold)

    print(f"\n{'='*60}")
    print(f"Session ID : {agent.session_id}")
    print(f"Flagged    : {len(fraud_ids)} fraudulent transactions")
    print(f"Output     : {Path(args.dataset) / 'submission.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
