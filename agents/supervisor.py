"""
agents/supervisor.py

Controls routing through the LangGraph DAG.

need_more_info behaviour:
  Previously hard-exited, dropping all arXiv results when SS was rate-limited.
  Now: if we have fetched_docs but the analyst flagged need_more_info (typically
  because SS returned nothing and arXiv-only coverage is thin), we still proceed
  to summarize rather than throwing the work away. The summarizer will produce a
  partial review and the frontend can surface a warning via sources_used / ss_failed.
"""

import logging
from orchestration.state import LitReviewState

logger = logging.getLogger(__name__)

# Minimum papers to attempt a review when SS failed — below this we truly give up
_MIN_DOCS_FOR_PARTIAL = 3


def supervisor_agent(state: LitReviewState) -> LitReviewState:
    fetched  = state.get("fetched_docs", [])
    decision = state.get("analysis_decision", "")
    logs     = state.get("logs", [])

    # ── First pass — nothing fetched yet → go research ────────────────────────
    if not fetched:
        logger.info("[supervisor] No docs yet → research")
        logs.append("[supervisor] Bootstrapping → research")
        state["next_step"] = "research"
        state["logs"]      = logs
        return state

    # ── Analyst flagged need_more_info ────────────────────────────────────────
    if decision == "need_more_info":
        n = len(fetched)
        if n >= _MIN_DOCS_FOR_PARTIAL:
            # We have enough to produce something useful — don't discard it.
            # This is the common SS-429 case: arXiv returned papers but SS failed.
            logger.warning(
                "[supervisor] need_more_info but %d docs available — "
                "proceeding to summarize (partial review)", n
            )
            logs.append(
                f"[supervisor] SS rate-limited; partial review from {n} arXiv papers"
            )
            state["next_step"] = "summarize"
        else:
            # Genuinely too few papers — end gracefully
            logger.warning(
                "[supervisor] need_more_info with only %d docs — ending", n
            )
            logs.append(f"[supervisor] Insufficient data ({n} docs) → end")
            state["next_step"] = "end"

        state["logs"] = logs
        return state

    # ── Analyst said ready → summarize ────────────────────────────────────────
    if decision == "ready":
        logger.info("[supervisor] Analysis ready → summarize")
        logs.append("[supervisor] Analysis ready → summarize")
        state["next_step"] = "summarize"
        state["logs"]      = logs
        return state

    # ── Default: docs present but no decision yet → run analysis ──────────────
    logger.info("[supervisor] Docs fetched → analysis")
    logs.append("[supervisor] Docs fetched → analysis")
    state["next_step"] = "analysis"
    state["logs"]      = logs
    return state