"""Background task scheduler using APScheduler + in-process AsyncIOScheduler."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ConversationReference

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None

# Set by init_scheduler() — shared with tools/teams_messaging.py
_adapter = None
_app_id: str = ""


async def init_scheduler(adapter, app_id: str) -> AsyncIOScheduler:
    """Initialize the scheduler, load persisted tasks, and start.

    Called once from app.py on_startup.
    """
    global _scheduler, _adapter, _app_id
    _adapter = adapter
    _app_id = app_id

    # Also configure the Teams messaging module with adapter access
    import tools.teams_messaging as teams_msg

    teams_msg._adapter = adapter
    teams_msg._app_id = app_id

    _scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 300,
        }
    )

    from task_store import get_task_store

    store = await get_task_store()
    tasks = await store.list_tasks(status="active")
    for task in tasks:
        _register_job(task)

    # Daily in-boundary knowledge-base compile (support pilot). Runs in-process
    # on samurai-bot (inside the Assured Workloads boundary) via regional Vertex
    # Gemini. Gated by KB_PIPELINE_ENABLED so it stays dormant until enabled.
    from kb.run import engineering_pipeline_enabled, pipeline_enabled

    if pipeline_enabled():
        _scheduler.add_job(
            _run_kb_pipeline,
            CronTrigger.from_crontab(os.environ.get("KB_PIPELINE_CRON", "0 8 * * *")),
            id="kb_support_pipeline",
            replace_existing=True,
        )
        logger.info("KB support pipeline scheduled (daily, in-boundary Gemini).")

    # Daily engineering-knowledge sync: refresh the virtualdojo system map from
    # the repo (only when something merged to main). Defaults to 9am. Gated by
    # KB_ENG_PIPELINE_ENABLED so it stays dormant until enabled.
    if engineering_pipeline_enabled():
        _scheduler.add_job(
            _run_kb_engineering_pipeline,
            CronTrigger.from_crontab(os.environ.get("KB_ENG_PIPELINE_CRON", "0 9 * * *")),
            id="kb_engineering_pipeline",
            replace_existing=True,
        )
        logger.info("KB engineering pipeline scheduled (daily 9am, in-boundary Gemini).")

    # Skills catalog sync: pull approved skills from virtualdojo-skills into the
    # in-boundary bucket SamurAI serves (support/skills/synced/). Read-only inward,
    # runs in-process (never a GitHub runner). Gated by SKILLS_SYNC_ENABLED.
    from kb.sync_skills import sync_enabled as skills_sync_enabled

    if skills_sync_enabled():
        _scheduler.add_job(
            _run_skill_sync,
            CronTrigger.from_crontab(os.environ.get("SKILLS_SYNC_CRON", "*/30 * * * *")),
            id="skills_sync",
            replace_existing=True,
        )
        logger.info("Skills catalog sync scheduled (in-boundary, read-only pull).")

    # Skills capture: distill reusable skills from SamurAI's own conversation log,
    # in-boundary on Vertex Gemini, sanitize, and file labeled skill-draft issues for
    # review. Gated by SKILLS_DISTILL_ENABLED.
    from kb.distill_skills import distill_enabled as skills_distill_enabled

    if skills_distill_enabled():
        _scheduler.add_job(
            _run_skill_distill,
            CronTrigger.from_crontab(os.environ.get("SKILLS_DISTILL_CRON", "30 8 * * *")),
            id="skills_distill",
            replace_existing=True,
        )
        logger.info("Skills capture/distill scheduled (in-boundary Gemini, review-gated).")

    # Prompt self-tuning loop: propose→evaluate→promote edits to the mutable
    # learned_hints.md, gated by an objective eval set. Adaptive cadence is
    # self-managed inside run_tuning_cycle. Gated by KB_TUNE_ENABLED.
    from selftune.loop import tune_enabled

    if tune_enabled():
        _scheduler.add_job(
            _run_tuning_cycle,
            CronTrigger.from_crontab(os.environ.get("KB_TUNE_CRON", "0 7 * * *")),
            id="selftune_cycle",
            replace_existing=True,
        )
        logger.info("Prompt self-tuning scheduled (in-boundary, eval-gated).")

    # DH Tech Issue Tracker triage: pre-compute fact-grounded diagnoses for
    # new/changed rows so they're ready when a team member engages. Read-only
    # (never acts). Runs twice per business day (10:00 and 17:00 Eastern) to keep
    # AI spend down — down from the old every-10-min business-hours cadence. The
    # trigger is timezone-aware (TRACKER_TRIAGE_TZ, default America/New_York) so it
    # tracks EST/EDT automatically instead of drifting an hour across DST.
    # Gated by TRACKER_TRIAGE_ENABLED.
    from tracker_triage import triage_enabled

    if triage_enabled():
        _scheduler.add_job(
            _run_tracker_triage,
            CronTrigger.from_crontab(
                os.environ.get("TRACKER_TRIAGE_CRON", "0 10,17 * * 1-5"),
                timezone=os.environ.get("TRACKER_TRIAGE_TZ", "America/New_York"),
            ),
            id="tracker_triage",
            replace_existing=True,
        )
        logger.info(
            "Tracker triage scheduled (twice daily 10:00/17:00 America/New_York, read-only)."
        )

    _scheduler.start()
    logger.info("Scheduler started with %d active tasks", len(tasks))
    return _scheduler


# Serialize the heavyweight background pipelines. They all call Gemini, and the
# 2026-07 log review showed the KB compile (*/5), the engineering sync, the
# tracker-triage batch (*/10), and proactive tasks landing on the same minutes —
# saturating the model quota (429 RESOURCE_EXHAUSTED storms) and pinning the
# single warm instance long enough for live Teams requests to hit Cloud Run's
# 600s timeout. One pipeline at a time is plenty; they're all periodic caches.
_BG_PIPELINE_LOCK = asyncio.Lock()


async def _run_kb_pipeline() -> None:
    """Run the in-boundary KB pipeline off the event loop (it does blocking I/O)."""
    from kb.run import run_support_pipeline

    try:
        async with _BG_PIPELINE_LOCK:
            await asyncio.to_thread(run_support_pipeline)
    except Exception as e:  # never let a pipeline run crash the scheduler
        # exc_info=True so the FULL traceback lands in Cloud Logging — without
        # it we only get "<Type>: <msg>" and the real failure stays hidden.
        logger.error(
            "[kb.run] support pipeline failed: %s: %s",
            type(e).__name__, e, exc_info=True,
        )


async def _run_kb_engineering_pipeline() -> None:
    """Run the in-boundary engineering-knowledge sync off the event loop."""
    from kb.run import run_engineering_pipeline

    try:
        async with _BG_PIPELINE_LOCK:
            await asyncio.to_thread(run_engineering_pipeline)
    except Exception as e:  # never let a pipeline run crash the scheduler
        logger.error(
            "[kb.run] engineering pipeline failed: %s: %s",
            type(e).__name__, e, exc_info=True,
        )


async def _run_skill_sync() -> None:
    """Pull approved skills into the bucket, off the event loop (blocking GCS/GitHub I/O)."""
    from kb.sync_skills import run_skill_sync

    try:
        async with _BG_PIPELINE_LOCK:
            await asyncio.to_thread(run_skill_sync)
    except Exception as e:  # never let a sync run crash the scheduler
        logger.error(
            "[skills.sync] catalog sync failed: %s: %s",
            type(e).__name__, e, exc_info=True,
        )


async def _run_skill_distill() -> None:
    """Distill skills from the conversation log off the event loop (blocking Gemini/GCS I/O)."""
    from kb.distill_skills import run_skill_distill

    try:
        async with _BG_PIPELINE_LOCK:
            await asyncio.to_thread(run_skill_distill)
    except Exception as e:  # never let a distill run crash the scheduler
        logger.error(
            "[skills.distill] capture run failed: %s: %s",
            type(e).__name__, e, exc_info=True,
        )


async def _run_tuning_cycle() -> None:
    """Run the prompt self-tuning cycle off the event loop."""
    from selftune.loop import run_tuning_cycle

    try:
        async with _BG_PIPELINE_LOCK:
            await asyncio.to_thread(run_tuning_cycle)
    except Exception as e:  # never let a tuning run crash the scheduler
        logger.error(
            "[selftune] cycle failed: %s: %s",
            type(e).__name__, e, exc_info=True,
        )


async def _run_tracker_triage() -> None:
    """Diagnose new/changed DH Tech Issue Tracker rows; park the results."""
    from tracker_triage import run_triage_batch

    try:
        async with _BG_PIPELINE_LOCK:
            await run_triage_batch()
    except Exception as e:  # never let a triage run crash the scheduler
        logger.error(
            "[tracker.triage] batch failed: %s: %s",
            type(e).__name__, e, exc_info=True,
        )


def _register_job(task: dict) -> None:
    """Register a single task as an APScheduler job."""
    if not _scheduler:
        return

    job_id = f"task_{task['id']}"

    if task["task_type"] == "recurring" and task.get("cron_expression"):
        try:
            trigger = CronTrigger.from_crontab(task["cron_expression"])
        except ValueError as e:
            logger.error("Invalid cron for task %s: %s", task["id"], e)
            return
    elif task["task_type"] == "one_shot" and task.get("run_at"):
        try:
            run_date = datetime.fromisoformat(task["run_at"])
        except ValueError as e:
            logger.error("Invalid run_at for task %s: %s", task["id"], e)
            return
        # A restart rebuilds jobs from the store, which can resurface one-shots
        # whose run time is long past (observed: a job "missed by 82 days" firing
        # a misfire warning after a deploy). Don't register ancient one-shots.
        now = datetime.now(run_date.tzinfo) if run_date.tzinfo else datetime.now()
        if (now - run_date).total_seconds() > 3600:
            logger.warning(
                "Skipping stale one-shot task %s (run_at=%s is >1h past)",
                task["id"], task["run_at"],
            )
            return
        trigger = DateTrigger(run_date=run_date)
    else:
        logger.warning("Task %s has no valid trigger config, skipping", task["id"])
        return

    _scheduler.add_job(
        _execute_task,
        trigger=trigger,
        id=job_id,
        args=[task["id"]],
        replace_existing=True,
    )
    logger.info("Registered job %s for task %s", job_id, task["id"])


def _reschedule_one_shot(task_id: str, run_date: datetime) -> None:
    """Re-register a one-shot task with a new DateTrigger for retry."""
    if not _scheduler:
        return
    job_id = f"task_{task_id}"
    _scheduler.add_job(
        _execute_task,
        trigger=DateTrigger(run_date=run_date),
        id=job_id,
        args=[task_id],
        replace_existing=True,
    )
    logger.info("Rescheduled job %s for %s", job_id, run_date.isoformat())


async def _resolve_conversation_ref(store, task: dict) -> str | None:
    """Look up a conversation ref, following the bg_task_ parent chain if needed.

    When a background task spawns sub-tasks, those sub-tasks get a synthetic
    ``bg_task_<parent_id>`` conversation ID that has no saved ref.  This helper
    walks the parent chain until it finds a real (non-synthetic) ref, then
    caches the result so future lookups succeed directly.
    """
    ref_json = await store.get_conversation_ref(task["conversation_id"])
    if ref_json:
        return ref_json

    # Follow bg_task_ → parent task → parent's conversation_id chain
    conv_id = task["conversation_id"]
    visited: set[str] = set()
    while conv_id.startswith("bg_task_") and conv_id not in visited:
        visited.add(conv_id)
        parent_task_id = conv_id.removeprefix("bg_task_")
        parent_task = await store.get_task(parent_task_id)
        if not parent_task:
            break
        ref_json = await store.get_conversation_ref(parent_task["conversation_id"])
        if ref_json:
            # Cache so future lookups succeed directly
            await store.save_conversation_ref(
                conversation_id=task["conversation_id"],
                user_id=task["user_id"],
                ref_json=ref_json,
            )
            return ref_json
        conv_id = parent_task["conversation_id"]

    return None


_BG_RECURSION_LIMIT = 75
_BG_RECURSION_LIMIT_RETRY = 150


def _is_recursion_error(exc: BaseException) -> bool:
    return (
        type(exc).__name__ == "GraphRecursionError"
        or "recursion limit" in str(exc).lower()
    )


async def _run_agent_with_recursion_recovery(
    *,
    run_agent,
    task: dict,
    bg_conversation_id: str,
) -> str:
    """Run the agent for a background task, retrying once with a higher
    recursion limit if the first attempt hits GraphRecursionError.

    If the retry also fails, return a graceful user-facing message instead
    of re-raising — so the user gets a coherent reply rather than a silent
    auto-pause.
    """
    kwargs = dict(
        user_message=task["prompt"],
        conversation_id=bg_conversation_id,
        user_id=task["user_id"],
        user_name=task["user_name"],
        user_timezone=task["user_timezone"],
        user_email=task["user_email"],
        is_background_task=True,
    )
    try:
        return await run_agent(recursion_limit=_BG_RECURSION_LIMIT, **kwargs)
    except Exception as e:
        if not _is_recursion_error(e):
            raise
        logger.warning(
            "[scheduler] recursion retry kicked in for task %s (limit %d -> %d): %s",
            task["id"],
            _BG_RECURSION_LIMIT,
            _BG_RECURSION_LIMIT_RETRY,
            e,
        )

    try:
        return await run_agent(recursion_limit=_BG_RECURSION_LIMIT_RETRY, **kwargs)
    except Exception as e:
        if not _is_recursion_error(e):
            raise
        logger.error(
            "[scheduler] recursion retry exhausted for task %s at limit %d: %s",
            task["id"],
            _BG_RECURSION_LIMIT_RETRY,
            e,
        )
        return (
            f"I worked on **{task['prompt'][:120]}** but it turned out more "
            "involved than I expected and I couldn't finish in the steps I had. "
            "Want me to narrow the scope, or break it into a few smaller tasks?"
        )


async def _execute_task(task_id: str) -> None:
    """Execute a background task: run the agent, send results to Teams."""
    from task_store import get_task_store
    from agent import run_agent

    store = await get_task_store()
    task = await store.get_task(task_id)
    if not task or task["status"] != "active":
        return

    # Acquire execution lock — prevents duplicate runs across instances
    if not await store.try_lock(task_id):
        logger.info("Task %s already locked by another instance, skipping", task_id)
        return

    logger.info("Executing task %s: %s", task_id, task["prompt"][:80])

    try:
        # Use a dedicated thread_id so background history stays separate
        bg_conversation_id = f"bg_task_{task_id}"

        # Propagate the original conversation ref to the bg conversation ID
        # so any sub-tasks created during execution can deliver results
        ref_json = await _resolve_conversation_ref(store, task)
        if ref_json:
            await store.save_conversation_ref(
                conversation_id=bg_conversation_id,
                user_id=task["user_id"],
                ref_json=ref_json,
            )

        response = await _run_agent_with_recursion_recovery(
            run_agent=run_agent,
            task=task,
            bg_conversation_id=bg_conversation_id,
        )

        await _send_task_result(task, response)
        await store.record_run(task_id, success=True)
        logger.info("Task %s completed successfully (run #%d)", task_id, task["run_count"] + 1)

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e, exc_info=True)
        updated = await store.record_run(
            task_id, success=False, error_message=str(e)
        )

        if updated and updated["status"] == "failed":
            await _send_failure_notification(task, str(e))
            # Remove from scheduler since it's auto-paused
            await pause_task(task_id)
        elif (
            updated
            and task["task_type"] == "one_shot"
            and updated["status"] == "active"
        ):
            # One-shot tasks lose their DateTrigger after firing once.
            # Reschedule with a 60-second delay so the retry has a chance.
            retry_at = datetime.now(timezone.utc) + timedelta(seconds=60)
            logger.info(
                "Rescheduling one-shot task %s for retry at %s",
                task_id,
                retry_at.isoformat(),
            )
            _reschedule_one_shot(task_id, retry_at)
    finally:
        await store.unlock(task_id)


async def _send_task_result(task: dict, response: str) -> None:
    """Send the agent's response to the original Teams conversation."""
    from task_store import get_task_store

    store = await get_task_store()
    ref_json = await _resolve_conversation_ref(store, task)
    if not ref_json:
        logger.error(
            "No conversation ref for task %s, conv %s",
            task["id"],
            task["conversation_id"],
        )
        return

    conv_ref = ConversationReference().deserialize(json.loads(ref_json))

    # Keep the message clean — just deliver the content
    message_text = response

    async def _notify(turn_context: TurnContext):
        await turn_context.send_activity(
            Activity(type="message", text=message_text)
        )

    try:
        await _adapter.continue_conversation(conv_ref, _notify, _app_id)
    except Exception as e:
        logger.error("Proactive message failed for task %s: %s", task["id"], e)


async def _send_failure_notification(task: dict, error: str) -> None:
    """Notify the user that a task has been auto-paused due to repeated failures."""
    from task_store import get_task_store

    store = await get_task_store()
    ref_json = await _resolve_conversation_ref(store, task)
    if not ref_json:
        return

    conv_ref = ConversationReference().deserialize(json.loads(ref_json))

    error_count = task.get("error_count", 0) + 1
    message = (
        f"**Task Auto-Paused** `{task['id']}`\n\n"
        f"Task: _{task['prompt'][:80]}_\n"
        f"Failed {error_count} consecutive times.\n"
        f"Last error: `{error[:200]}`\n\n"
        f"Say **resume task {task['id']}** to retry, or "
        f"**cancel task {task['id']}** to remove it."
    )

    async def _notify(turn_context: TurnContext):
        await turn_context.send_activity(Activity(type="message", text=message))

    try:
        await _adapter.continue_conversation(conv_ref, _notify, _app_id)
    except Exception as e:
        logger.error(
            "Failure notification failed for task %s: %s", task["id"], e
        )


# ── Public API for agent tools ─────────────────────────────────────────


async def schedule_task(task: dict) -> None:
    """Register a newly created task with the scheduler."""
    _register_job(task)


async def pause_task(task_id: str) -> None:
    """Pause a scheduled task (remove from APScheduler, keep in DB)."""
    job_id = f"task_{task_id}"
    if _scheduler and _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)


async def resume_task(task: dict) -> None:
    """Resume a paused task (re-register with APScheduler)."""
    _register_job(task)


async def cancel_task(task_id: str) -> None:
    """Fully remove a task from the scheduler."""
    await pause_task(task_id)


async def shutdown_scheduler() -> None:
    """Gracefully shut down the scheduler. Called from app.py on_cleanup."""
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")
