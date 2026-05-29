"""Tests for scheduler.py — APScheduler-based background task execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import scheduler as scheduler_module


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_task(
    task_id="abc123",
    task_type="recurring",
    cron_expression="0 9 * * *",
    run_at=None,
    status="active",
    prompt="Check deployment status",
    user_id="user-1",
    user_name="Test User",
    user_email="test@example.com",
    user_timezone="America/Chicago",
    conversation_id="conv-1",
    run_count=0,
    error_count=0,
    max_failures=3,
):
    return {
        "id": task_id,
        "task_type": task_type,
        "cron_expression": cron_expression,
        "run_at": run_at,
        "status": status,
        "prompt": prompt,
        "user_id": user_id,
        "user_name": user_name,
        "user_email": user_email,
        "user_timezone": user_timezone,
        "conversation_id": conversation_id,
        "run_count": run_count,
        "error_count": error_count,
        "max_failures": max_failures,
    }


@pytest.fixture(autouse=True)
def _reset_scheduler_globals():
    """Ensure every test starts with a clean scheduler module state."""
    original_scheduler = scheduler_module._scheduler
    original_adapter = scheduler_module._adapter
    original_app_id = scheduler_module._app_id
    yield
    scheduler_module._scheduler = original_scheduler
    scheduler_module._adapter = original_adapter
    scheduler_module._app_id = original_app_id


@pytest.fixture
def mock_scheduler():
    """Provide a MagicMock standing in for AsyncIOScheduler."""
    sched = MagicMock()
    sched.add_job = MagicMock()
    sched.get_job = MagicMock(return_value=MagicMock())
    sched.remove_job = MagicMock()
    sched.shutdown = MagicMock()
    scheduler_module._scheduler = sched
    return sched


@pytest.fixture
def mock_adapter():
    """Provide a MagicMock standing in for BotFrameworkAdapter."""
    adapter = MagicMock()
    adapter.continue_conversation = AsyncMock()
    scheduler_module._adapter = adapter
    scheduler_module._app_id = "test-app-id"
    return adapter


@pytest.fixture
def mock_store():
    """Provide an AsyncMock standing in for TaskStore."""
    store = AsyncMock()
    store.get_task = AsyncMock()
    store.record_run = AsyncMock()
    store.get_conversation_ref = AsyncMock()
    return store


# ── _register_job tests ──────────────────────────────────────────────────


class TestRegisterJob:
    """Tests for _register_job()."""

    def test_registers_recurring_job_with_cron(self, mock_scheduler):
        task = _make_task(task_type="recurring", cron_expression="0 9 * * *")
        scheduler_module._register_job(task)

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert call_kwargs.kwargs["id"] == "task_abc123"
        assert call_kwargs.kwargs["replace_existing"] is True
        assert call_kwargs.kwargs["args"] == ["abc123"]

    def test_registers_one_shot_job_with_run_at(self, mock_scheduler):
        task = _make_task(
            task_type="one_shot",
            cron_expression=None,
            run_at="2026-06-15T14:00:00",
        )
        scheduler_module._register_job(task)

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert call_kwargs.kwargs["id"] == "task_abc123"

    def test_invalid_cron_does_not_raise(self, mock_scheduler):
        task = _make_task(task_type="recurring", cron_expression="bad cron")
        # Should log an error but not raise
        scheduler_module._register_job(task)
        mock_scheduler.add_job.assert_not_called()

    def test_invalid_run_at_does_not_raise(self, mock_scheduler):
        task = _make_task(
            task_type="one_shot",
            cron_expression=None,
            run_at="not-a-date",
        )
        scheduler_module._register_job(task)
        mock_scheduler.add_job.assert_not_called()

    def test_missing_trigger_config_skips(self, mock_scheduler):
        task = _make_task(task_type="recurring", cron_expression=None)
        scheduler_module._register_job(task)
        mock_scheduler.add_job.assert_not_called()

    def test_noop_when_scheduler_is_none(self):
        scheduler_module._scheduler = None
        task = _make_task()
        # Should silently return without error
        scheduler_module._register_job(task)


# ── _execute_task tests ──────────────────────────────────────────────────


class TestExecuteTask:
    """Tests for _execute_task()."""

    @pytest.mark.asyncio
    async def test_success_calls_run_agent_and_records(
        self, mock_adapter, mock_store
    ):
        task = _make_task()
        mock_store.get_task.return_value = task
        mock_store.record_run.return_value = task
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                return_value="Deployment healthy",
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("abc123")

        mock_run_agent.assert_called_once_with(
            user_message="Check deployment status",
            conversation_id="bg_task_abc123",
            user_id="user-1",
            user_name="Test User",
            user_timezone="America/Chicago",
            user_email="test@example.com",
            recursion_limit=75,
            is_background_task=True,
        )
        mock_store.record_run.assert_called_once_with("abc123", success=True)

    @pytest.mark.asyncio
    async def test_success_sends_proactive_message(
        self, mock_adapter, mock_store
    ):
        task = _make_task()
        mock_store.get_task.return_value = task
        mock_store.record_run.return_value = task
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                return_value="All good",
            ),
        ):
            await scheduler_module._execute_task("abc123")

        mock_adapter.continue_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_records_error(self, mock_adapter, mock_store):
        task = _make_task()
        mock_store.get_task.return_value = task
        # Not auto-paused yet (status stays active)
        mock_store.record_run.return_value = {**task, "status": "active", "error_count": 1}
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API timeout"),
            ),
        ):
            await scheduler_module._execute_task("abc123")

        mock_store.record_run.assert_called_once_with(
            "abc123", success=False, error_message="API timeout"
        )

    @pytest.mark.asyncio
    async def test_failure_auto_pause_sends_notification(
        self, mock_scheduler, mock_adapter, mock_store
    ):
        task = _make_task(error_count=2)
        mock_store.get_task.return_value = task
        # Simulate auto-pause: status becomes "failed"
        mock_store.record_run.return_value = {**task, "status": "failed", "error_count": 3}
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with patch(
            "task_store.get_task_store",
            new_callable=AsyncMock,
            return_value=mock_store,
        ), patch(
            "agent.run_agent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            await scheduler_module._execute_task("abc123")

        # Failure notification sends a proactive message
        # continue_conversation is called for the failure notification
        mock_adapter.continue_conversation.assert_called()
        # The job should also be removed from the scheduler (via pause_task)
        mock_scheduler.remove_job.assert_called_with("task_abc123")

    @pytest.mark.asyncio
    async def test_skips_inactive_task(self, mock_store):
        task = _make_task(status="paused")
        mock_store.get_task.return_value = task

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("abc123")

        mock_run_agent.assert_not_called()
        mock_store.record_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_missing_task(self, mock_store):
        mock_store.get_task.return_value = None

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("nonexistent")

        mock_run_agent.assert_not_called()


# ── One-shot retry tests ────────────────────────────────────────────────


class TestOneShotRetry:
    """Tests for one-shot task retry on failure."""

    @pytest.mark.asyncio
    async def test_one_shot_failure_reschedules(
        self, mock_scheduler, mock_adapter, mock_store
    ):
        """A failed one-shot task that hasn't exhausted retries gets rescheduled."""
        task = _make_task(
            task_type="one_shot",
            cron_expression=None,
            run_at="2026-04-10T16:00:00Z",
            max_failures=2,
        )
        mock_store.get_task.return_value = task
        # error_count=1 < max_failures=2, so status stays "active"
        mock_store.record_run.return_value = {
            **task,
            "status": "active",
            "error_count": 1,
        }
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Database unavailable"),
            ),
        ):
            await scheduler_module._execute_task("abc123")

        # Should reschedule via add_job with a DateTrigger
        reschedule_call = None
        for call in mock_scheduler.add_job.call_args_list:
            if call.kwargs.get("id") == "task_abc123":
                reschedule_call = call
        assert reschedule_call is not None, "Expected task to be rescheduled"
        # The trigger should be a DateTrigger (for the retry)
        from apscheduler.triggers.date import DateTrigger

        assert isinstance(reschedule_call.kwargs["trigger"], DateTrigger)

    @pytest.mark.asyncio
    async def test_one_shot_exhausted_retries_not_rescheduled(
        self, mock_scheduler, mock_adapter, mock_store
    ):
        """A one-shot task that hits max_failures does NOT get rescheduled."""
        task = _make_task(
            task_type="one_shot",
            cron_expression=None,
            run_at="2026-04-10T16:00:00Z",
            error_count=1,
            max_failures=2,
        )
        mock_store.get_task.return_value = task
        # error_count reaches max_failures, status becomes "failed"
        mock_store.record_run.return_value = {
            **task,
            "status": "failed",
            "error_count": 2,
        }
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Database still unavailable"),
            ),
        ):
            await scheduler_module._execute_task("abc123")

        # Should send failure notification, not reschedule
        mock_adapter.continue_conversation.assert_called()
        # The scheduler remove_job should be called (pause), not add_job for reschedule
        mock_scheduler.remove_job.assert_called_with("task_abc123")

    @pytest.mark.asyncio
    async def test_recurring_failure_does_not_reschedule(
        self, mock_scheduler, mock_adapter, mock_store
    ):
        """Recurring tasks rely on their cron trigger — no rescheduling needed."""
        task = _make_task(
            task_type="recurring",
            cron_expression="0 9 * * *",
        )
        mock_store.get_task.return_value = task
        mock_store.record_run.return_value = {
            **task,
            "status": "active",
            "error_count": 1,
        }
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        initial_add_job_count = mock_scheduler.add_job.call_count

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("temporary error"),
            ),
        ):
            await scheduler_module._execute_task("abc123")

        # No new add_job calls (no reschedule for recurring tasks)
        assert mock_scheduler.add_job.call_count == initial_add_job_count


# ── Public API tests ─────────────────────────────────────────────────────


class TestPublicAPI:
    """Tests for schedule_task, pause_task, resume_task, cancel_task."""

    @pytest.mark.asyncio
    async def test_schedule_task_registers_job(self, mock_scheduler):
        task = _make_task()
        await scheduler_module.schedule_task(task)
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_task_removes_job(self, mock_scheduler):
        mock_scheduler.get_job.return_value = MagicMock()
        await scheduler_module.pause_task("abc123")
        mock_scheduler.remove_job.assert_called_once_with("task_abc123")

    @pytest.mark.asyncio
    async def test_pause_task_noop_when_no_job(self, mock_scheduler):
        mock_scheduler.get_job.return_value = None
        await scheduler_module.pause_task("abc123")
        mock_scheduler.remove_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_pause_task_noop_when_no_scheduler(self):
        scheduler_module._scheduler = None
        # Should not raise
        await scheduler_module.pause_task("abc123")

    @pytest.mark.asyncio
    async def test_resume_task_registers_job(self, mock_scheduler):
        task = _make_task()
        await scheduler_module.resume_task(task)
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_task_removes_job(self, mock_scheduler):
        mock_scheduler.get_job.return_value = MagicMock()
        await scheduler_module.cancel_task("abc123")
        mock_scheduler.remove_job.assert_called_once_with("task_abc123")


# ── shutdown_scheduler tests ─────────────────────────────────────────────


class TestShutdownScheduler:
    """Tests for shutdown_scheduler()."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_scheduler_shutdown(self, mock_scheduler):
        await scheduler_module.shutdown_scheduler()
        mock_scheduler.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_scheduler(self):
        scheduler_module._scheduler = None
        # Should not raise
        await scheduler_module.shutdown_scheduler()


# ── _resolve_conversation_ref tests ─────────────────────────────────────


class TestResolveConversationRef:
    """Tests for _resolve_conversation_ref() — resolving refs through bg_task_ chains."""

    @pytest.mark.asyncio
    async def test_direct_ref_found(self, mock_store):
        """When the task's conversation_id has a ref, return it directly."""
        task = _make_task(conversation_id="conv-1")
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        ref = await scheduler_module._resolve_conversation_ref(mock_store, task)

        assert ref == '{"conversationId": "conv-1"}'
        mock_store.get_conversation_ref.assert_called_once_with("conv-1")
        mock_store.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_for_non_bg_task_missing_ref(self, mock_store):
        """When a regular conversation_id has no ref, return None (no chain to follow)."""
        task = _make_task(conversation_id="conv-missing")
        mock_store.get_conversation_ref.return_value = None

        ref = await scheduler_module._resolve_conversation_ref(mock_store, task)

        assert ref is None

    @pytest.mark.asyncio
    async def test_follows_bg_task_chain_one_level(self, mock_store):
        """A sub-task with bg_task_<parent_id> resolves via the parent's ref."""
        sub_task = _make_task(
            task_id="child1",
            conversation_id="bg_task_parent1",
            user_id="user-1",
        )
        parent_task = _make_task(task_id="parent1", conversation_id="conv-real")

        # First call: sub_task's conversation_id has no ref
        # Second call: parent's conversation_id has the real ref
        mock_store.get_conversation_ref.side_effect = [
            None,                                # bg_task_parent1 → miss
            '{"conversationId": "conv-real"}',   # conv-real → hit
        ]
        mock_store.get_task.return_value = parent_task

        ref = await scheduler_module._resolve_conversation_ref(mock_store, sub_task)

        assert ref == '{"conversationId": "conv-real"}'
        # Should cache the resolved ref under the sub-task's conversation_id
        mock_store.save_conversation_ref.assert_called_once_with(
            conversation_id="bg_task_parent1",
            user_id="user-1",
            ref_json='{"conversationId": "conv-real"}',
        )

    @pytest.mark.asyncio
    async def test_follows_bg_task_chain_two_levels(self, mock_store):
        """A deeply nested sub-task resolves through multiple bg_task_ levels."""
        grandchild = _make_task(
            task_id="gc1",
            conversation_id="bg_task_child1",
            user_id="user-1",
        )
        child_task = _make_task(task_id="child1", conversation_id="bg_task_parent1")
        parent_task = _make_task(task_id="parent1", conversation_id="conv-original")

        mock_store.get_conversation_ref.side_effect = [
            None,                                    # bg_task_child1 → miss
            None,                                    # bg_task_parent1 → miss
            '{"conversationId": "conv-original"}',   # conv-original → hit
        ]
        mock_store.get_task.side_effect = [child_task, parent_task]

        ref = await scheduler_module._resolve_conversation_ref(mock_store, grandchild)

        assert ref == '{"conversationId": "conv-original"}'
        mock_store.save_conversation_ref.assert_called_once_with(
            conversation_id="bg_task_child1",
            user_id="user-1",
            ref_json='{"conversationId": "conv-original"}',
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_parent_task_missing(self, mock_store):
        """If the parent task was deleted, return None gracefully."""
        sub_task = _make_task(
            task_id="orphan1",
            conversation_id="bg_task_deleted_parent",
        )
        mock_store.get_conversation_ref.return_value = None
        mock_store.get_task.return_value = None  # parent doesn't exist

        ref = await scheduler_module._resolve_conversation_ref(mock_store, sub_task)

        assert ref is None
        mock_store.save_conversation_ref.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_infinite_loop_on_circular_chain(self, mock_store):
        """Cycle detection: bg_task_A → task A with conv bg_task_A doesn't loop forever."""
        task_a = _make_task(task_id="aaa", conversation_id="bg_task_aaa")

        mock_store.get_conversation_ref.return_value = None
        mock_store.get_task.return_value = task_a  # points back to itself

        ref = await scheduler_module._resolve_conversation_ref(mock_store, task_a)

        assert ref is None

    @pytest.mark.asyncio
    async def test_execute_task_propagates_ref_to_bg_conversation(
        self, mock_adapter, mock_store
    ):
        """_execute_task copies the parent ref to bg_task_<id> before running the agent."""
        task = _make_task(conversation_id="conv-1")
        mock_store.get_task.return_value = task
        mock_store.record_run.return_value = task
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                return_value="Done",
            ),
        ):
            await scheduler_module._execute_task("abc123")

        # The ref should be saved under the bg_task_ conversation ID
        mock_store.save_conversation_ref.assert_any_call(
            conversation_id="bg_task_abc123",
            user_id="user-1",
            ref_json='{"conversationId": "conv-1"}',
        )

    @pytest.mark.asyncio
    async def test_execute_task_still_runs_when_ref_missing(
        self, mock_adapter, mock_store
    ):
        """Task execution proceeds even if no conversation ref is found."""
        task = _make_task(conversation_id="bg_task_gone")
        mock_store.get_task.side_effect = [
            task,   # get_task(task_id) in _execute_task
            None,   # get_task(parent_id) in _resolve_conversation_ref
            None,   # get_task(parent_id) in _resolve_conversation_ref (_send_task_result)
        ]
        mock_store.record_run.return_value = task
        mock_store.get_conversation_ref.return_value = None

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                return_value="Result",
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("abc123")

        # Agent should still have been called even though ref is missing
        mock_run_agent.assert_called_once()
        # Run should still be recorded as successful
        mock_store.record_run.assert_called_once_with("abc123", success=True)

    @pytest.mark.asyncio
    async def test_send_task_result_resolves_bg_task_ref(
        self, mock_adapter, mock_store
    ):
        """_send_task_result resolves a bg_task_ ref through the parent chain."""
        sub_task = _make_task(
            task_id="sub1",
            conversation_id="bg_task_parent1",
        )
        parent_task = _make_task(task_id="parent1", conversation_id="conv-real")

        mock_store.get_conversation_ref.side_effect = [
            None,                                # bg_task_parent1 → miss
            '{"conversationId": "conv-real"}',   # conv-real → hit
        ]
        mock_store.get_task.return_value = parent_task

        with patch(
            "task_store.get_task_store",
            new_callable=AsyncMock,
            return_value=mock_store,
        ):
            await scheduler_module._send_task_result(sub_task, "Task complete")

        # Should have sent the proactive message using the resolved ref
        mock_adapter.continue_conversation.assert_called_once()
