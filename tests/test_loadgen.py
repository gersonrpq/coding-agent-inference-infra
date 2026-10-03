"""The load generator's analysis: it decides what we conclude from the experiment, so it is tested too."""
import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from tests._env import ROOT

spec = importlib.util.spec_from_file_location("loadgen", os.path.join(ROOT, "script", "loadgen.py"))
loadgen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loadgen)


def row(t, session=0, turn=1, call=1, status=200, ttft=1.0, cls="interactive", prompt=1000, cached=0, shed_reason=None):
    return {"t": t, "session": session, "cls": cls, "turn": turn, "call": call, "attempt": 1, "status": status,
            "ttft": ttft if status == 200 else None, "total": (ttft or 0) + 5 if status == 200 else 0.1,
            "prompt_tokens": prompt if status == 200 else None, "completion_tokens": 100 if status == 200 else None,
            "cached_tokens": cached if status == 200 else None, "error": None, "shed_reason": shed_reason}


def summarize(rows, **over):
    args = SimpleNamespace(warmup=0, duration=600, slo_ttft=20.0, max_shed=0.01, sessions=4, profile="paper", out=None)
    for k, v in over.items():
        setattr(args, k, v)
    with tempfile.TemporaryDirectory() as tmp:
        args.out = tmp
        run = loadgen.Run.__new__(loadgen.Run)
        run.a = args
        run.t0 = 1000.0
        run.counters = {"compactions": 0, "turns": 0}
        run.rec = SimpleNamespace(snapshot=lambda: rows)
        return run.summarize()


class Pure(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        values = list(range(1, 101))
        self.assertEqual((loadgen.pct(values, 50), loadgen.pct(values, 99), loadgen.pct(values, 100)), (50, 99, 100))
        self.assertIsNone(loadgen.pct([], 50))

    def test_call_position(self):
        self.assertEqual(loadgen.position({"turn": 1, "call": 1}), "session_start")
        self.assertEqual(loadgen.position({"turn": 3, "call": 1}), "turn_start")
        self.assertEqual(loadgen.position({"turn": 3, "call": 2}), "within_turn")

    def test_lognormal_stays_inside_its_bounds(self):
        import random
        rnd = random.Random(1)
        values = [loadgen.lognorm(rnd, 247, 0.9, 16, 1000) for _ in range(2000)]
        self.assertTrue(16 <= min(values) and max(values) <= 1000)


class ShedClassification(unittest.TestCase):
    def test_admission_header_wins(self):
        self.assertEqual(loadgen.classify_shed(503, "anything", "kv_pressure"), "kv_pressure")

    def test_engine_refusals_are_recognised_by_message(self):
        self.assertEqual(loadgen.classify_shed(503, "The request queue is full.", None), "engine_queue_full")
        self.assertEqual(loadgen.classify_shed(503, "The request is aborted by a higher priority request.", None), "engine_evicted")
        self.assertEqual(loadgen.classify_shed(503, "boom", None), "upstream_503")
        self.assertEqual(loadgen.classify_shed(503, "Worker at capacity: 14 in-flight, 0 queued requests. Retry later.", None), "queue_full")

    def test_timeouts_are_recognised_by_status_or_message(self):
        self.assertEqual(loadgen.classify_shed(408, "Request timed out", None), "timeout")
        self.assertEqual(loadgen.classify_shed(500, "litellm.Timeout: ...", None), "timeout")
        self.assertIsNone(loadgen.classify_shed(400, "bad request", None))


class Verdict(unittest.TestCase):
    def test_pass_when_p99_inside_slo_and_no_sheds(self):
        s = summarize([row(t, ttft=2.0) for t in range(100)])
        self.assertTrue(s["verdict"]["pass"])
        self.assertAlmostEqual(s["classes"]["interactive"]["ttft_within_slo"], 1.0)

    def test_fail_on_a_slow_tail(self):
        rows = [row(t, ttft=2.0) for t in range(95)] + [row(100 + t, ttft=25.0) for t in range(5)]
        s = summarize(rows)
        self.assertFalse(s["verdict"]["ttft_p99_ok"])
        self.assertFalse(s["verdict"]["pass"])

    def test_fail_on_too_many_sheds_and_reasons_are_counted(self):
        rows = [row(t) for t in range(90)] + [row(100 + t, status=503, shed_reason="timeout_queue") for t in range(10)]
        s = summarize(rows)
        c = s["classes"]["interactive"]
        self.assertEqual(c["shed"], 10)
        self.assertEqual(c["shed_by_reason"], {"timeout_queue": 10})
        self.assertAlmostEqual(c["shed_rate"], 0.1)
        self.assertFalse(s["verdict"]["shed_ok"])

    def test_batch_does_not_count_against_the_interactive_verdict(self):
        rows = [row(t, ttft=2.0) for t in range(100)] + [row(t, cls="batch", status=503, shed_reason="batch_pressure") for t in range(50)]
        s = summarize(rows)
        self.assertTrue(s["verdict"]["pass"])
        self.assertEqual(s["classes"]["batch"]["shed"], 50)

    def test_warmup_is_excluded(self):
        rows = [row(t, ttft=60.0) for t in range(50)] + [row(100 + t, ttft=1.0) for t in range(100)]
        s = summarize(rows, warmup=100)
        self.assertEqual(s["classes"]["interactive"]["attempts"], 100)
        self.assertTrue(s["verdict"]["pass"])

    def test_a_call_retried_after_a_shed_counts_as_served_not_abandoned(self):
        rows = [row(10, status=503, shed_reason="kv_pressure"), dict(row(13, ttft=1.0), attempt=2)]
        c = summarize(rows)["classes"]["interactive"]
        self.assertEqual((c["calls"], c["calls_abandoned"], c["shed"]), (1, 0, 1))


class ShedWait(unittest.TestCase):
    def test_how_long_the_refusal_took_is_reported(self):
        fast = [dict(row(t, status=503, shed_reason="engine_queue_full"), total=0.2) for t in range(8)]
        slow = [dict(row(10 + t, status=503, shed_reason="timeout"), total=20.0) for t in range(2)]
        c = summarize(fast + slow + [row(50, ttft=1.0)])["classes"]["interactive"]
        self.assertEqual(c["shed"], 10)
        self.assertEqual(c["shed_late_over_5s"], 2)
        self.assertEqual(c["shed_wait_p50"], 0.2)
        self.assertEqual(c["shed_wait_p95"], 20.0)


class Stickiness(unittest.TestCase):
    def test_share_of_consecutive_calls_of_a_session_on_the_same_worker(self):
        rows = [dict(row(1, session=0), worker="w0"), dict(row(2, session=0), worker="w0"), dict(row(3, session=0), worker="w1"),
                dict(row(1, session=1), worker="w1"), dict(row(2, session=1), worker="w0")]
        c = summarize(rows)["classes"]["interactive"]
        self.assertAlmostEqual(c["worker_stickiness"], 1 / 3, places=3)          # 1 of the 3 follow-up calls stayed

    def test_unknown_when_the_worker_is_not_reported(self):
        self.assertIsNone(summarize([row(1), row(2)])["classes"]["interactive"]["worker_stickiness"])


class Positions(unittest.TestCase):
    def test_ttft_and_cache_share_are_split_by_position(self):
        rows = [row(1, turn=1, call=1, ttft=4.0, prompt=1000, cached=0),
                row(2, turn=2, call=1, ttft=9.0, prompt=1000, cached=500),
                row(3, turn=2, call=2, ttft=0.5, prompt=1000, cached=950)]
        pos = summarize(rows)["classes"]["interactive"]["by_position"]
        self.assertEqual(pos["session_start"]["ttft_p99"], 4.0)
        self.assertEqual(pos["turn_start"]["ttft_p99"], 9.0)
        self.assertEqual(pos["turn_start"]["cached_share"], 0.5)
        self.assertEqual(pos["within_turn"]["cached_share"], 0.95)

    def test_a_median_that_keeps_rising_is_not_a_steady_state(self):
        rising = [row(t, ttft=1.0) for t in range(0, 300)] + [row(t, ttft=5.0) for t in range(300, 600)]
        flat = [row(t, ttft=1.0) for t in range(600)]
        self.assertFalse(summarize(rising)["verdict"]["steady_state"])
        self.assertTrue(summarize(flat)["verdict"]["steady_state"])


if __name__ == "__main__":
    unittest.main()
