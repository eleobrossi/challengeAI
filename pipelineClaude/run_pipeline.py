#!/usr/bin/env python3
"""
run_pipeline.py

Master orchestrator for the Reply Mirror AI Agent Challenge.

Runs all agents in sequence under a single Langfuse session ID so that every
LLM call's token usage and cost is aggregated in the challenge dashboard.

Pipeline:
    Agent 1 (base_statistics_agent1)  – Advanced EDA & statistical profiling
    Agent 2 (feature_engineer_agent2) – Feature engineering (per-tx feature matrix)
    Agent 3 (anomaly_score_agent3)    – Pure-statistical composite anomaly scoring
    Agent 4 (agent4_llm_decision)     – LLM final decision + Langfuse tracking

Output:
    <dataset_dir>/submission.txt      – Required ASCII output for challenge submission
    Langfuse traces grouped under a single session_id

Usage:
    python run_pipeline.py --dataset datasets/level_1
    python run_pipeline.py --dataset datasets/level_1 --top-n 300 --threshold 0.28
    python run_pipeline.py --dataset datasets/level_1 --skip-eda --skip-features
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# Session ID generation
# The Langfuse client lives in agent4_llm_decision (module-level, tutorial pattern).
# The pipeline only needs to generate the shared session_id.
# ---------------------------------------------------------------------------
try:
    import ulid
    _ULID_AVAILABLE = True
except ImportError:
    _ULID_AVAILABLE = False


def generate_session_id() -> str:
    """Generate {TEAM_NAME}-{ULID} session ID — tutorial format."""
    team = os.getenv("TEAM_NAME", "team")
    if _ULID_AVAILABLE:
        return f"{team}-{ulid.new().str}"
    import uuid
    return f"{team}-{uuid.uuid4().hex[:20].upper()}"


# ---------------------------------------------------------------------------
# Agent imports (same directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from base_statistics_agent1  import AdvancedEDAAgent
from feature_engineer_agent2 import FeatureEngineeringAgent
from anomaly_score_agent3    import AnomalyScoringAgent
from agent4_llm_decision     import LLMDecisionAgent


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)


def _run_with_timing(name: str, fn):
    _banner(f"Starting {name}")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        log.info(f"  ✓ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f"  ✗ {name} FAILED after {elapsed:.1f}s: {exc}")
        raise


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_dir: str,
    top_n: int = 500,
    threshold: float = 0.30,
    batch_size: int = 15,
    skip_eda: bool = False,
    skip_features: bool = False,
    skip_scoring: bool = False,
    session_id: str | None = None,
) -> str:
    """
    Execute the full fraud-detection pipeline.

    Args:
        dataset_dir:   Path to the dataset directory (must contain transactions.csv).
        top_n:         Number of top suspicious transactions kept by agent3.
        threshold:     Min anomaly_score passed to the LLM in agent4.
        batch_size:    LLM batch size for agent4.
        skip_eda:      Skip agent1 (useful when outputs already exist).
        skip_features: Skip agent2 (use existing feature_outputs/).
        skip_scoring:  Skip agent3 (use existing anomaly_outputs/).
        session_id:    Force a specific Langfuse session ID.

    Returns:
        The session_id used for Langfuse tracking.
    """

    # One session ID shared across the whole pipeline
    # agent4 will use it internally and tag every Langfuse trace with it
    sid = session_id or generate_session_id()

    # Expose via env so agent4's module-level client can pick it up if needed
    os.environ["PIPELINE_SESSION_ID"] = sid

    # Reuse the module-level langfuse_client and flag already imported by agent4
    from agent4_llm_decision import langfuse_client as _lf_client
    from agent4_llm_decision import _LANGFUSE_AVAILABLE

    _banner("REPLY MIRROR – Fraud Detection Pipeline")
    log.info(f"  Dataset    : {dataset_dir}")
    log.info(f"  Session ID : {sid}")
    log.info(f"  Top-N      : {top_n}")
    log.info(f"  LLM thresh : {threshold}")
    log.info(f"  Langfuse   : {'enabled' if _LANGFUSE_AVAILABLE else 'disabled'}")
    print()

    # -----------------------------------------------------------------------
    # Agent 1 – Advanced EDA
    # -----------------------------------------------------------------------
    if not skip_eda:
        def _run_eda():
            agent = AdvancedEDAAgent(dataset_dir)
            agent.run()

        _run_with_timing("Agent 1 – Advanced EDA", _run_eda)
    else:
        log.info("Agent 1 – EDA skipped (--skip-eda)")

    # -----------------------------------------------------------------------
    # Agent 2 – Feature Engineering
    # -----------------------------------------------------------------------
    if not skip_features:
        def _run_features():
            agent = FeatureEngineeringAgent(dataset_dir)
            agent.run()

        _run_with_timing("Agent 2 – Feature Engineering", _run_features)
    else:
        log.info("Agent 2 – Features skipped (--skip-features)")

    # -----------------------------------------------------------------------
    # Agent 3 – Statistical Anomaly Scoring
    # -----------------------------------------------------------------------
    if not skip_scoring:
        def _run_scoring():
            agent = AnomalyScoringAgent(dataset_dir, top_n=top_n)
            agent.run()

        _run_with_timing("Agent 3 – Anomaly Scoring", _run_scoring)
    else:
        log.info("Agent 3 – Scoring skipped (--skip-scoring)")

    # -----------------------------------------------------------------------
    # Agent 4 – LLM Final Decision (with Langfuse)
    # -----------------------------------------------------------------------
    def _run_llm_decision():
        agent = LLMDecisionAgent(
            dataset_dir=dataset_dir,
            session_id=sid,
            batch_size=batch_size,
        )
        return agent.run(threshold=threshold)

    fraud_ids = _run_with_timing("Agent 4 – LLM Decision", _run_llm_decision)

    # -----------------------------------------------------------------------
    # Final flush — agent4 already calls flush(), this is a safety net
    # -----------------------------------------------------------------------
    if _LANGFUSE_AVAILABLE and _lf_client is not None:
        try:
            _lf_client.flush()
        except Exception:
            pass

    submission_path = Path(dataset_dir) / "submission.txt"
    _banner("Pipeline Complete")
    log.info(f"  Session ID     : {sid}")
    log.info(f"  Fraudulent txs : {len(fraud_ids) if fraud_ids else 0}")
    log.info(f"  Submission     : {submission_path}")
    if _LANGFUSE_AVAILABLE:
        log.info(f"  Langfuse host  : {os.getenv('LANGFUSE_HOST', 'N/A')}")
    print()

    # Print Langfuse trace info if available
    if _LANGFUSE_AVAILABLE and _lf_client is not None:
        _print_trace_info(_lf_client, sid)

    return sid


# ---------------------------------------------------------------------------
# Langfuse trace viewer (from tutorial)
# ---------------------------------------------------------------------------

def _print_trace_info(langfuse_client, session_id: str) -> None:
    """Fetch and pretty-print aggregated trace info for the session."""
    try:
        from datetime import datetime
        from collections import defaultdict

        traces, page = [], 1
        while True:
            resp = langfuse_client.api.trace.list(session_id=session_id, limit=100, page=page)
            if not resp.data:
                break
            traces.extend(resp.data)
            if len(resp.data) < 100:
                break
            page += 1

        if not traces:
            # Langfuse propagation can take a few seconds — retry up to 3 times
            import time as _time
            for _retry in range(3):
                _time.sleep(5)
                log.info(f"  Waiting for Langfuse propagation… (attempt {_retry + 2}/4)")
                page, traces = 1, []
                while True:
                    resp2 = langfuse_client.api.trace.list(session_id=session_id, limit=100, page=page)
                    if not resp2.data:
                        break
                    traces.extend(resp2.data)
                    if len(resp2.data) < 100:
                        break
                    page += 1
                if traces:
                    break
            if not traces:
                log.info(f"Session {session_id} not yet indexed — check the dashboard in ~1 min.")
                return

        observations = []
        for trace in traces:
            detail = langfuse_client.api.trace.get(trace.id)
            if detail and hasattr(detail, "observations"):
                observations.extend(detail.observations)

        if not observations:
            return

        counts  = defaultdict(int)
        costs   = defaultdict(float)
        total_t = 0.0

        for obs in observations:
            if hasattr(obs, "type") and obs.type == "GENERATION":
                model = getattr(obs, "model", "unknown") or "unknown"
                counts[model] += 1
                if hasattr(obs, "calculated_total_cost") and obs.calculated_total_cost:
                    costs[model] += obs.calculated_total_cost
                if hasattr(obs, "start_time") and hasattr(obs, "end_time"):
                    if obs.start_time and obs.end_time:
                        total_t += (obs.end_time - obs.start_time).total_seconds()

        log.info("── Langfuse Session Summary ──────────────────────────")
        for model, cnt in counts.items():
            log.info(f"  Generations  [{model}]: {cnt}")
        total_cost = sum(costs.values())
        if total_cost > 0:
            for model, cost in costs.items():
                log.info(f"  Cost         [{model}]: ${cost:.6f}")
            log.info(f"  Total cost   : ${total_cost:.6f}")
        log.info(f"  Total time   : {total_t:.2f}s")
        log.info("──────────────────────────────────────────────────────")

    except Exception as e:
        log.debug(f"Could not fetch trace info: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reply Mirror – Full Fraud Detection Pipeline"
    )
    parser.add_argument("--dataset",       required=True,             help="Dataset directory (e.g. datasets/level_1)")
    parser.add_argument("--top-n",         type=int,   default=500,   help="Top-N anomalies kept by agent3 (default: 500)")
    parser.add_argument("--threshold",     type=float, default=0.30,  help="Min anomaly_score for LLM review (default: 0.30)")
    parser.add_argument("--batch-size",    type=int,   default=15,    help="Transactions per LLM batch (default: 15)")
    parser.add_argument("--session-id",    default=None,              help="Force a specific Langfuse session ID")
    parser.add_argument("--skip-eda",      action="store_true",       help="Skip agent1 (EDA)")
    parser.add_argument("--skip-features", action="store_true",       help="Skip agent2 (feature engineering)")
    parser.add_argument("--skip-scoring",  action="store_true",       help="Skip agent3 (anomaly scoring)")
    args = parser.parse_args()

    run_pipeline(
        dataset_dir   = args.dataset,
        top_n         = args.top_n,
        threshold     = args.threshold,
        batch_size    = args.batch_size,
        skip_eda      = args.skip_eda,
        skip_features = args.skip_features,
        skip_scoring  = args.skip_scoring,
        session_id    = args.session_id,
    )


if __name__ == "__main__":
    main()
