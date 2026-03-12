#!/usr/bin/env python3
"""
agent4_llm_decision.py

LLM-based final decision agent for the Reply Mirror AI Agent Challenge.

Langfuse integration follows the EXACT pattern from the challenge tutorial:
  - module-level langfuse_client  (not per-instance)
  - module-level @observe() function (not a class method)
  - CallbackHandler() created INSIDE the @observe() function
  - langfuse_client.update_current_trace(session_id=...) inside @observe()
  - langfuse_client.flush() after all calls

This ensures traces appear in the Langfuse dashboard and costs are aggregated
under a single session_id across all agent batches.

Input:
    <dataset_dir>/anomaly_outputs/anomaly_scores.csv   (from agent3)
    <dataset_dir>/transactions.csv                     (raw, for context)

Output:
    <dataset_dir>/submission.txt        – one fraudulent transaction_id per line
    <dataset_dir>/llm_decisions/
        decision_detail.csv             – per-transaction decision + reasoning
        decision_summary.json           – counts / session summary

Usage:
    python agent4_llm_decision.py --dataset datasets/level_1
    python agent4_llm_decision.py --dataset datasets/level_1 --session-id MYTEAM-01JX...
    python agent4_llm_decision.py --dataset datasets/level_1 --threshold 0.28 --batch-size 15
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

# =============================================================================
# Langfuse + LangChain — tutorial-exact module-level setup
# =============================================================================

_LANGFUSE_AVAILABLE = False
langfuse_client     = None   # module-level, exactly like the tutorial
_llm_model          = None   # module-level, exactly like the tutorial

try:
    import ulid
    # IMPORTANT: use get_client() so that langfuse_client is the SAME singleton
    # used internally by @observe().  Creating a new Langfuse(...) would give a
    # different object, and update_current_trace() would not reach the active trace.
    from langfuse import observe, get_client
    from langfuse.langchain import CallbackHandler
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    # get_client() returns (or lazily creates) the global default client.
    # It auto-reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST from the environment,
    # which load_dotenv() already populated at the top of this file.
    langfuse_client = get_client()

    _llm_model = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.getenv("OPENROUTER_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        max_tokens=1024,
    )

    _LANGFUSE_AVAILABLE = True
    log.info("✓ Langfuse client (get_client) and LLM model initialised")

except Exception as _e:
    log.warning(f"Langfuse/LangChain not available ({_e}). Heuristic fallback will be used.")


def generate_session_id() -> str:
    """Generate a unique session ID: {TEAM_NAME}-{ULID} — tutorial format."""
    team = os.getenv("TEAM_NAME", "team")
    if _LANGFUSE_AVAILABLE:
        return f"{team}-{ulid.new().str}"
    import uuid
    return f"{team}-{uuid.uuid4().hex[:20].upper()}"


# =============================================================================
# Tutorial-exact @observe() function  ← THIS is the key pattern
# =============================================================================

@observe()   # <-- module-level @observe(), NOT a class method
def _invoke_llm_traced(session_id: str, user_prompt: str) -> str:
    """
    Single LLM call, traced by Langfuse.

    Follows the tutorial pattern verbatim:
      1. langfuse_client.update_current_trace(session_id=...) tags the trace
      2. CallbackHandler() created INSIDE @observe() so it auto-attaches
      3. model.invoke(..., config={"callbacks": [handler]}) triggers capture
    """
    # Step 1 – tag this trace with the pipeline session_id (tutorial line)
    langfuse_client.update_current_trace(session_id=session_id)

    # Step 2 – CallbackHandler inside @observe() → auto-attaches to current trace
    langfuse_handler = CallbackHandler()

    # Step 3 – invoke LangChain with the handler (tokens / cost captured automatically)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = _llm_model.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    return response.content


# =============================================================================
# Prompt templates
# =============================================================================

SYSTEM_PROMPT = """You are a world-class financial fraud detection expert at MirrorPay, year 2087.

Your task: analyze transactions flagged as suspicious by a statistical anomaly
detector and determine which ones are truly fraudulent (Mirror Hacker activity).

Common Mirror Hacker tactics:
- Targeting new merchants / recipients the user has never paid before
- Unusual hours (midnight – 5 AM local time)
- High-velocity transaction bursts within minutes
- Amounts that nearly drain the sender's balance
- Transactions preceded by phishing SMS / emails

Scoring asymmetry (important):
- A missed fraud (false negative) → direct financial loss    [HIGH cost]
- A blocked legitimate transaction (false positive) → reputational damage [MEDIUM cost]
So: lean toward flagging when the evidence is ambiguous but significant.

Each transaction in the batch has:
  transaction_id, sender_id, recipient_id, transaction_type, amount,
  timestamp, anomaly_score (0–1), risk_tier (LOW/MEDIUM/HIGH/CRITICAL),
  and top_signals (the statistical features that drove the score highest).

Respond ONLY with a valid JSON object — no markdown, no extra text:
{
  "fraudulent_ids": ["id1", "id2", ...],
  "reasoning": "brief 1-2 sentence explanation"
}

Rules:
- Include ONLY transaction_ids from the input batch.
- CRITICAL / HIGH tier items should be flagged unless there is a clear
  legitimate explanation (known recurring salary, consistent daily pattern).
- Do NOT flag transactions that look like routine recurring payments.
"""


def _build_user_prompt(batch: List[Dict]) -> str:
    """Format a batch into a compact JSON prompt for the LLM."""
    items = []
    for r in batch:
        sig_keys = sorted(
            [k for k in r if k.startswith("sig_") and float(r.get(k) or 0) > 0.25],
            key=lambda k: float(r.get(k) or 0),
            reverse=True,
        )[:5]
        items.append({
            "transaction_id":   str(r.get("transaction_id", "")),
            "sender_id":        str(r.get("sender_id", "")),
            "recipient_id":     str(r.get("recipient_id", "")),
            "transaction_type": str(r.get("transaction_type", "")),
            "amount":           round(float(r.get("amount") or 0), 2),
            "timestamp":        str(r.get("timestamp", "")),
            "anomaly_score":    round(float(r.get("anomaly_score") or 0), 4),
            "risk_tier":        str(r.get("risk_tier", "MEDIUM")),
            "top_signals":      {k.replace("sig_", ""): round(float(r.get(k) or 0), 3) for k in sig_keys},
        })
    return f"Analyze these {len(batch)} transactions:\n\n{json.dumps(items, indent=2)}"


def _parse_llm_response(text: str) -> Tuple[List[str], str]:
    """Extract fraudulent_ids from LLM response (handles markdown fences)."""
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return [], ""
    try:
        obj = json.loads(m.group())
        ids  = [str(i) for i in obj.get("fraudulent_ids", [])]
        reason = str(obj.get("reasoning", ""))
        return ids, reason
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e} | raw: {text[:200]}")
        return [], ""


# =============================================================================
# Heuristic fallback (when LLM unavailable)
# =============================================================================

def _heuristic_fallback(batch: List[Dict]) -> List[str]:
    """
    Adaptive percentile-based fallback when LLM is unavailable.
    Flags the top ~30 % by anomaly_score within the batch, always including
    CRITICAL / HIGH tier items.
    """
    if not batch:
        return []
    scores = [float(r.get("anomaly_score") or 0) for r in batch]
    p70 = float(np.percentile(scores, 70)) if len(scores) > 3 else 0.0
    threshold = max(p70, 0.30)
    return [
        str(r["transaction_id"])
        for r in batch
        if float(r.get("anomaly_score") or 0) >= threshold
        or str(r.get("risk_tier", "")) in ("CRITICAL", "HIGH")
    ]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main agent class
# =============================================================================

class LLMDecisionAgent:
    """
    Orchestrates LLM fraud decisions across all batches, sharing one session_id
    so every Langfuse trace is grouped under a single session.
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
        self.session_id  = session_id or generate_session_id()
        ensure_dir(self.output_dir)
        log.info(f"LLM agent session_id: {self.session_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, threshold: float = 0.30) -> List[str]:
        """
        Run the LLM decision pipeline and write submission.txt.

        Args:
            threshold: min anomaly_score to pass to LLM (0–1)

        Returns:
            List of flagged (fraudulent) transaction_ids
        """
        scores_df = self._load_scores()
        raw_tx    = self._load_raw_transactions()

        if scores_df.empty:
            log.error("No anomaly_scores.csv found. Run anomaly_score_agent3 first.")
            return []

        # Enrich with extra raw columns if missing
        if not raw_tx.empty and "transaction_id" in scores_df.columns:
            extra = [c for c in ["description", "location", "payment_method",
                                 "sender_iban", "recipient_iban", "balance_after"]
                     if c in raw_tx.columns and c not in scores_df.columns]
            if extra:
                scores_df = scores_df.merge(
                    raw_tx[["transaction_id"] + extra], on="transaction_id", how="left"
                )

        # Candidate selection
        candidates = scores_df[scores_df["anomaly_score"].fillna(0) >= threshold].copy()
        # Always include CRITICAL tier regardless of threshold
        if "risk_tier" in scores_df.columns:
            crits = scores_df[scores_df["risk_tier"] == "CRITICAL"]
            candidates = pd.concat([candidates, crits]).drop_duplicates("transaction_id")

        if candidates.empty:
            log.warning(f"No candidates at threshold={threshold}. Using top-20% fallback.")
            n = max(20, len(scores_df) // 5)
            candidates = scores_df.nlargest(n, "anomaly_score")

        log.info(f"Candidates for LLM review: {len(candidates):,}  "
                 f"(threshold={threshold}, total={len(scores_df):,})")

        # Batch processing
        rows    = candidates.to_dict(orient="records")
        batches = [rows[i: i + self.batch_size] for i in range(0, len(rows), self.batch_size)]
        log.info(f"Processing {len(batches)} batch(es) of ≤{self.batch_size} transactions")

        all_fraudulent: List[str] = []
        decision_rows:  List[Dict] = []

        for batch_num, batch in enumerate(batches, 1):
            log.info(f"  Batch {batch_num}/{len(batches)}  ({len(batch)} txns)…")

            if _LANGFUSE_AVAILABLE and _llm_model is not None:
                try:
                    prompt      = _build_user_prompt(batch)
                    raw_resp    = _invoke_llm_traced(self.session_id, prompt)
                    fraud_ids, reasoning = _parse_llm_response(raw_resp)
                    log.info(f"    → LLM flagged {len(fraud_ids)} fraudulent")
                except Exception as exc:
                    log.error(f"    LLM call failed: {exc}")
                    fraud_ids = _heuristic_fallback(batch)
                    reasoning = "fallback-heuristic (LLM error)"
                    log.info(f"    → Heuristic fallback: {len(fraud_ids)} flagged")
            else:
                fraud_ids = _heuristic_fallback(batch)
                reasoning = "heuristic-only (LLM/Langfuse not available)"
                log.info(f"    → Heuristic: {len(fraud_ids)} flagged")

            all_fraudulent.extend(fraud_ids)

            fraud_set = set(fraud_ids)
            for r in batch:
                tid = str(r.get("transaction_id", ""))
                decision_rows.append({
                    "transaction_id": tid,
                    "anomaly_score":  round(float(r.get("anomaly_score") or 0), 4),
                    "risk_tier":      str(r.get("risk_tier", "")),
                    "llm_flagged":    int(tid in fraud_set),
                    "batch":          batch_num,
                    "reasoning":      reasoning,
                })

        all_fraudulent = list(dict.fromkeys(all_fraudulent))  # dedup, preserve order

        # Flush traces — tutorial step: always flush after all calls
        if _LANGFUSE_AVAILABLE and langfuse_client is not None:
            langfuse_client.flush()
            log.info("✓ Langfuse traces flushed")

        # Save outputs
        self._write_submission(all_fraudulent)
        pd.DataFrame(decision_rows).to_csv(self.output_dir / "decision_detail.csv", index=False)
        save_json({
            "session_id":         self.session_id,
            "n_candidates":       int(len(candidates)),
            "n_flagged_by_llm":   len(all_fraudulent),
            "n_total_in_dataset": int(len(scores_df)),
            "threshold_used":     threshold,
            "flagged_fraction":   round(len(all_fraudulent) / max(len(scores_df), 1), 4),
            "llm_available":      _LANGFUSE_AVAILABLE,
        }, self.output_dir / "decision_summary.json")

        log.info(f"Decision complete. Flagged {len(all_fraudulent)} / {len(scores_df)} transactions.")
        return all_fraudulent

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_scores(self) -> pd.DataFrame:
        path = self.dataset_dir / "anomaly_outputs" / "anomaly_scores.csv"
        if not path.exists():
            log.error(f"Not found: {path}")
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

    def _write_submission(self, fraud_ids: List[str]) -> None:
        out = self.dataset_dir / "submission.txt"
        with open(out, "w", encoding="ascii", errors="replace") as f:
            for tid in fraud_ids:
                f.write(tid + "\n")
        log.info(f"submission.txt → {out}  ({len(fraud_ids)} lines)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Decision Agent – Reply Mirror Challenge")
    parser.add_argument("--dataset",    required=True,            help="Dataset directory")
    parser.add_argument("--session-id", default=None,             help="Force a Langfuse session ID")
    parser.add_argument("--threshold",  type=float, default=0.30, help="Min anomaly_score for LLM (default 0.30)")
    parser.add_argument("--batch-size", type=int,   default=15,   help="Transactions per LLM batch (default 15)")
    args = parser.parse_args()

    agent = LLMDecisionAgent(
        dataset_dir=args.dataset,
        session_id=args.session_id,
        batch_size=args.batch_size,
    )
    fraud_ids = agent.run(threshold=args.threshold)

    print(f"\n{'='*60}")
    print(f"Session ID  : {agent.session_id}")
    print(f"Flagged     : {len(fraud_ids)} fraudulent transactions")
    print(f"Submission  : {Path(args.dataset) / 'submission.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
