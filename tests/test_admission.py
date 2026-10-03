"""Admission: what is shed and why (`should_shed`), placement, the overflow gate, and that counts always come back.

The cap on concurrent requests (12 + K) is LiteLLM's own middleware now, so it is not tested here (it is checked live)."""
import asyncio
import datetime
import unittest
from unittest import mock

from fastapi import HTTPException
from prometheus_client import REGISTRY

from tests._env import admission, fleet_state, place

HOOK = admission.admission_handler
T0 = datetime.datetime(2026, 1, 1, 12, 0, 0)


def request(**extra):
    data = {"model": "qwen-coding-local",
            "messages": [{"role": "system", "content": "You are a coding agent."},
                         {"role": "user", "content": "hi"}]}
    data.update(extra)
    return data


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def admit(data):
    return await HOOK.async_pre_call_hook(None, None, data, "acompletion")


class AdmissionCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for target, value in (("PREFIXES", admission._SeenPrefixes()),
                              ("ADMISSION_CHECKS", {"timeout_queue", "kv_pressure", "batch_pressure"}),
                              ("PLACEMENT_POLICY", "litellm"),
                              ("ensure_metrics_poller_started", lambda: None),
                              ("print_fleet_state", lambda *a, **k: None)):
            patcher = mock.patch.object(admission, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        admission.INFLIGHT.ids.clear()
        self.addCleanup(admission.INFLIGHT.ids.clear)
        fleet = fleet_state.global_fleet_state
        saved = (fleet.replicas, fleet.ttft_p50_s, fleet.ttft_p99_s)
        self.addCleanup(lambda: (setattr(fleet, "replicas", saved[0]), setattr(fleet, "ttft_p50_s", saved[1]),
                                 setattr(fleet, "ttft_p99_s", saved[2])))
        self.set_fleet()

    def set_fleet(self, waiting=0, kv=0.0, p50=0.0, p99=0.0):
        fleet = fleet_state.global_fleet_state
        fleet.replicas = [fleet_state.ReplicaState(url="u0", id="0", waiting=waiting, kv_usage=kv),
                          fleet_state.ReplicaState(url="u1", id="1", waiting=0, kv_usage=0.0)]
        fleet.ttft_p50_s, fleet.ttft_p99_s = p50, p99

    async def assertShed(self, data, reason, klass="interactive"):
        before = sample("orch_requests_shed_total", reason=reason, status_code="503", priority_class=klass)
        with self.assertRaises(HTTPException) as caught:
            await admit(data)
        exc = caught.exception
        self.assertEqual(exc.status_code, 503)                 # capacity is 503, never 429
        self.assertEqual(exc.headers["shed-reason"], reason)
        self.assertEqual(exc.headers["source"], "admission")
        self.assertEqual(exc.headers["retry-after"], "2")
        after = sample("orch_requests_shed_total", reason=reason, status_code="503", priority_class=klass)
        self.assertEqual(after - before, 1)
        return exc


class Admit(AdmissionCase):
    async def test_healthy_fleet_admits_and_tags_the_request(self):
        data = await admit(request())
        meta = data["metadata"]
        self.assertTrue(meta["fleet_interactive"])
        self.assertEqual(meta["fleet_priority"], 5)
        self.assertEqual(data["extra_body"]["priority"], 5)    # the engine orders its own queue with it

    async def test_other_call_types_pass_untouched(self):
        data = request()
        result = await HOOK.async_pre_call_hook(None, None, data, "embedding")
        self.assertIs(result, data)
        self.assertNotIn("metadata", data)

    async def test_extra_body_keys_are_kept(self):
        data = await admit(request(extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
        self.assertEqual(data["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(data["extra_body"]["priority"], 5)

    async def test_priority_parsing(self):
        for value, batch in ((None, False), (5, False), ("5", False), (8, True), ("8", True), ("junk", False)):
            data = await admit(request(priority=value) if value is not None else request())
            self.assertEqual(data["metadata"]["fleet_interactive"], not batch, value)

    async def test_priority_in_metadata_is_honoured(self):
        data = await admit(request(metadata={"priority": 9}))
        self.assertFalse(data["metadata"]["fleet_interactive"])


class Sheds(AdmissionCase):
    async def test_timeout_queue_when_engine_queue_would_eat_half_the_deadline(self):
        self.set_fleet(waiting=10, p50=2.0)                    # 20 s estimated wait > 20 s deadline / 2
        await self.assertShed(request(), "timeout_queue")

    async def test_the_same_load_does_not_shed_batch_because_its_deadline_is_longer(self):
        self.set_fleet(waiting=10, p50=2.0)                    # 20 s < 120 s / 2
        data = await admit(request(priority=8))
        self.assertFalse(data["metadata"]["fleet_interactive"])

    async def test_no_wait_estimate_without_ttft_samples(self):
        self.set_fleet(waiting=1000, p50=0.0)                  # p50 unknown: no basis to shed
        await admit(request())

    async def test_the_estimate_rule_is_off_unless_listed(self):
        self.set_fleet(waiting=10, p50=2.0)
        with mock.patch.object(admission, "ADMISSION_CHECKS", {"kv_pressure", "batch_pressure"}):
            await admit(request())

    async def test_kv_pressure_sheds_a_new_prefix_but_not_a_cached_one(self):
        self.set_fleet(kv=0.97)
        await self.assertShed(request(), "kv_pressure")
        admission.PREFIXES.add(admission._prefix_key(request()))
        await admit(request())                                  # cached prefixes cost no new KV

    async def test_batch_pressure_sheds_batch_first(self):
        self.set_fleet(p50=2.0, p99=12.0)                      # p99 > 4 x p50 and above the 10 s floor
        await self.assertShed(request(priority=8), "batch_pressure", klass="batch")
        self.assertTrue((await admit(request()))["metadata"]["fleet_interactive"])

    async def test_batch_is_not_shed_for_a_wide_but_harmless_tail(self):
        self.set_fleet(p50=1.0, p99=8.0)                       # 8x the median but far from the SLO: the normal shape here
        self.assertFalse((await admit(request(priority=8)))["metadata"]["fleet_interactive"])


class BatchShare(AdmissionCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(admission, "BATCH_ADMIT_BELOW", 3)
        patcher.start()
        self.addCleanup(patcher.stop)
        admission.INFLIGHT.ids.clear()
        self.addCleanup(admission.INFLIGHT.ids.clear)

    async def test_batch_enters_only_while_fewer_than_the_threshold_are_in_flight_and_interactive_always_does(self):
        await admit(request(priority=8))
        await admit(request())
        await admit(request(priority=8))                         # 2 in flight < 3: batch still enters
        await self.assertShed(request(priority=8), "batch_share", klass="batch")      # 3 in flight (any class)
        for _ in range(5):                                       # the places above the threshold are for interactive calls
            self.assertTrue((await admit(request()))["metadata"]["fleet_interactive"])

    async def test_interactive_requests_count_toward_the_threshold(self):
        for _ in range(3):
            await admit(request())
        await self.assertShed(request(priority=8), "batch_share", klass="batch")

    async def test_a_finished_request_gives_its_place_back_once(self):
        first = await admit(request())
        await admit(request())
        await admit(request())
        event = {"litellm_params": {"metadata": first["metadata"]}}
        await HOOK.async_log_success_event(event, None, T0, T0)
        await HOOK.async_log_failure_event(event, None, T0, T0)                       # twice: harmless
        await admit(request(priority=8))                                              # 2 in flight again: batch enters
        await self.assertShed(request(priority=8), "batch_share", klass="batch")

    async def test_zero_means_no_reservation(self):
        with mock.patch.object(admission, "BATCH_ADMIT_BELOW", 0):
            for _ in range(20):
                await admit(request(priority=8))


class ShouldShed(unittest.TestCase):
    """`should_shed(req, snap)` -> (shed?, code, reason, retry_after_seconds): the pure admission decision."""

    class Snap:
        def __init__(self, kv=0.0, p50=0.0, p99=0.0, wait=0.0):
            self.kv_usage_max, self.ttft_p50_s, self.ttft_p99_s, self.estimated_queue_wait_s = kv, p50, p99, wait

        @property
        def very_bad_tail_latency(self):
            return self.ttft_p50_s > 0 and self.ttft_p99_s > 4 * self.ttft_p50_s and self.ttft_p99_s > 10

    ALL = {"timeout_queue", "kv_pressure", "batch_pressure"}

    def decide(self, req=None, snap=None, **kw):
        return admission.should_shed(req or admission.ShedRequest(), snap or self.Snap(), checks=kw.pop("checks", self.ALL), **kw)

    def test_a_healthy_fleet_does_not_shed(self):
        self.assertEqual(self.decide(), (False, None, None, None))

    def test_each_reason_has_a_503_and_a_retry_after(self):
        cases = [
            (admission.ShedRequest(deadline_s=20), self.Snap(p50=2.0, wait=11.0), {}, "timeout_queue"),
            (admission.ShedRequest(), self.Snap(kv=0.96), {}, "kv_pressure"),
            (admission.ShedRequest(is_batch=True), self.Snap(p50=2.0, p99=12.0), {}, "batch_pressure"),
            (admission.ShedRequest(is_batch=True), self.Snap(), {"in_flight": 10}, "batch_share"),
        ]
        for req, snap, kw, reason in cases:
            self.assertEqual(self.decide(req, snap, **kw), (True, 503, reason, 2), reason)

    def test_a_cached_prefix_is_exempt_from_kv_pressure_and_interactive_from_the_batch_rules(self):
        self.assertFalse(self.decide(admission.ShedRequest(cached_prefix=True), self.Snap(kv=0.99))[0])
        self.assertFalse(self.decide(admission.ShedRequest(), self.Snap(p50=2.0, p99=12.0), in_flight=14)[0])

    def test_a_check_that_is_not_enabled_does_not_run(self):
        self.assertFalse(self.decide(admission.ShedRequest(), self.Snap(kv=0.99), checks=set())[0])

    def test_the_cheap_stateless_checks_come_before_the_batch_share(self):
        self.assertEqual(self.decide(admission.ShedRequest(is_batch=True), self.Snap(kv=0.99), in_flight=12)[2], "kv_pressure")


class EventsRecordLatency(AdmissionCase):
    def _event(self, data):
        return {"litellm_params": {"metadata": data["metadata"]}, "stream": False}

    async def test_success_event_records_the_class_latency(self):
        data = await admit(request())
        before = sample("orch_request_latency_seconds_count", priority_class="interactive")
        await HOOK.async_log_success_event(self._event(data), None, T0, T0 + datetime.timedelta(seconds=2))
        self.assertEqual(sample("orch_request_latency_seconds_count", priority_class="interactive") - before, 1)

    async def test_ttft_is_recorded_only_for_streaming_requests(self):
        data = await admit(request())
        before = sample("orch_request_ttft_seconds_count", priority_class="interactive")
        await HOOK.async_log_success_event(self._event(data), None, T0, T0 + datetime.timedelta(seconds=2))
        self.assertEqual(sample("orch_request_ttft_seconds_count", priority_class="interactive"), before)
        data = await admit(request())
        event = self._event(data)
        event.update(stream=True, completion_start_time=T0 + datetime.timedelta(seconds=1))
        await HOOK.async_log_success_event(event, None, T0, T0 + datetime.timedelta(seconds=2))
        self.assertEqual(sample("orch_request_ttft_seconds_count", priority_class="interactive") - before, 1)


class PlacementIntegration(AdmissionCase):
    def setUp(self):
        super().setUp()
        import random
        self.placement = place.Placement(workers=("w0", "w1"), rng=random.Random(3))
        patcher = mock.patch.object(admission, "PLACEMENT", self.placement)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_default_policy_leaves_the_model_to_litellm(self):
        data = await admit(request())
        self.assertEqual(data["model"], "qwen-coding-local")
        self.assertEqual(self.placement.inflight, {"w0": 0, "w1": 0})

    async def test_affload_rewrites_the_model_to_the_worker_alias_and_counts_it(self):
        with mock.patch.object(admission, "PLACEMENT_POLICY", "affload"):
            data = await admit(request())
        self.assertIn(data["model"], ("qwen-coding-w0", "qwen-coding-w1"))
        self.assertEqual(sum(self.placement.inflight.values()), 1)
        self.assertIn(data["metadata"]["placement_worker"], ("w0", "w1"))

    async def test_success_and_failure_events_release_the_placement_count(self):
        with mock.patch.object(admission, "PLACEMENT_POLICY", "affload"):
            first, second = await admit(request(messages=[{"role": "user", "content": "a"}])), await admit(request(messages=[{"role": "user", "content": "b"}]))
        self.assertEqual(sorted(self.placement.inflight.values()), [1, 1])            # spread over both workers
        await HOOK.async_log_success_event({"litellm_params": {"metadata": first["metadata"]}}, None, T0, T0)
        await HOOK.async_log_failure_event({"litellm_params": {"metadata": first["metadata"]}}, None, T0, T0)   # twice: harmless
        self.assertEqual(sum(self.placement.inflight.values()), 1)
        await HOOK.async_post_call_failure_hook({"metadata": second["metadata"]}, Exception("x"), None)
        self.assertEqual(sum(self.placement.inflight.values()), 0)

    async def test_no_worker_answering_is_a_503_no_healthy_worker_and_leaks_nothing(self):
        fleet_state.global_fleet_state.workers_up = set()
        self.addCleanup(lambda: setattr(fleet_state.global_fleet_state, "workers_up", None))
        with mock.patch.object(admission, "PLACEMENT_POLICY", "affload"):
            await self.assertShed(request(), "no_healthy_worker")
        self.assertEqual(sum(self.placement.inflight.values()), 0)
        self.assertEqual(len(admission.INFLIGHT.ids), 0)

    async def test_the_engine_queue_depth_reaches_the_placement_and_breaks_the_tie(self):
        self.set_fleet()
        fleet_state.global_fleet_state.replicas = [
            fleet_state.ReplicaState(url="http://sglang-worker-0:30000", id="0", waiting=4),
            fleet_state.ReplicaState(url="http://sglang-worker-1:30001", id="1", waiting=0)]
        with mock.patch.object(admission, "PLACEMENT_POLICY", "affload"):
            data = await admit(request())
        self.assertEqual(data["metadata"]["placement_worker"], "w1")

    async def test_a_shed_request_is_never_placed(self):
        self.set_fleet(kv=0.97)
        with mock.patch.object(admission, "PLACEMENT_POLICY", "affload"):
            await self.assertShed(request(), "kv_pressure")
        self.assertEqual(sum(self.placement.inflight.values()), 0)


class OverflowGate(unittest.TestCase):
    def test_only_a_capacity_refusal_may_leave(self):
        for status in (429, 500, 400, 403, 408, 413, 200):
            self.assertEqual(admission.overflow_decision(status), "stay", status)
        for status in (503, 529):
            self.assertEqual(admission.overflow_decision(status, "kv_pressure", "admission"), "leave")
        self.assertEqual(admission.overflow_decision(None), "stay")

    def test_slice_oom_and_the_guard_never_leave_even_with_a_503(self):
        self.assertEqual(admission.overflow_decision(503, "slice_oom"), "stay")
        self.assertEqual(admission.overflow_decision(503, None, "inspect"), "stay")     # fail-closed guard

    def test_every_failed_call_is_counted_and_nothing_is_forwarded(self):
        before = sample("orch_overflow_decisions_total", decision="leave", status="503", forwarded="no")
        admission.admission_handler._gate_overflow(HTTPException(status_code=503, headers={"shed-reason": "kv_pressure", "source": "admission"}))
        admission.admission_handler._gate_overflow(HTTPException(status_code=429))
        admission.admission_handler._gate_overflow(RuntimeError("no status"))            # ignored
        self.assertEqual(sample("orch_overflow_decisions_total", decision="leave", status="503", forwarded="no") - before, 1)
        self.assertGreaterEqual(sample("orch_overflow_decisions_total", decision="stay", status="429", forwarded="no"), 1)


class PromptEstimate(unittest.TestCase):
    def test_four_characters_per_token_for_strings_and_text_parts(self):
        messages = [{"role": "user", "content": "x" * 400},
                    {"role": "user", "content": [{"type": "text", "text": "y" * 400}, {"type": "image_url", "image_url": {"url": "data:..."}}]}]
        self.assertEqual(admission._estimate_prompt_tokens(messages), 200)
        self.assertEqual(admission._estimate_prompt_tokens(None), 0)


class PrefixKey(unittest.TestCase):
    def test_same_system_prompt_same_key_and_different_prompt_different_key(self):
        a = admission._prefix_key(request())
        self.assertEqual(a, admission._prefix_key(request()))
        other = request()
        other["messages"][0]["content"] = "another system prompt"
        self.assertNotEqual(a, admission._prefix_key(other))

    def test_user_turns_do_not_change_the_key(self):
        a = request()
        b = request()
        b["messages"].append({"role": "user", "content": "a much longer conversation"})
        self.assertEqual(admission._prefix_key(a), admission._prefix_key(b))

    def test_tools_change_the_key_and_no_system_no_tools_means_no_key(self):
        self.assertNotEqual(admission._prefix_key(request()), admission._prefix_key(request(tools=[{"type": "function"}])))
        self.assertIsNone(admission._prefix_key({"messages": [{"role": "user", "content": "x"}]}))


if __name__ == "__main__":
    unittest.main()
