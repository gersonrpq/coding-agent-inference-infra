"""Placement: the policies of `pick`, the load counters and the session memory."""
import random
import unittest
from unittest import mock

from tests._env import place

Req = place.PlacementRequest


def fresh(**kw):
    return place.Placement(workers=("w0", "w1"), rng=random.Random(7), **kw)


class SessionKey(unittest.TestCase):
    def test_the_first_user_message_identifies_the_conversation(self):
        a = [{"role": "system", "content": "S"}, {"role": "user", "content": "task one"}]
        b = a + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        self.assertEqual(place.session_key(a), place.session_key(b))
        self.assertNotEqual(place.session_key(a), place.session_key([{"role": "user", "content": "task two"}]))

    def test_text_parts_and_plain_strings_give_the_same_key(self):
        plain = [{"role": "user", "content": "hello world"}]
        parts = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}]
        self.assertEqual(place.session_key(plain), place.session_key(parts))

    def test_no_user_message_means_no_key(self):
        self.assertIsNone(place.session_key([{"role": "system", "content": "S"}]))
        self.assertIsNone(place.session_key(None))


class LoadCounters(unittest.TestCase):
    def test_start_and_finish_count_and_finish_is_idempotent(self):
        p = fresh()
        p.start("r1", "w0")
        p.start("r2", "w0")
        self.assertEqual(p.inflight, {"w0": 2, "w1": 0})
        p.finish("r1")
        p.finish("r1")
        p.finish(None)
        p.finish("never-started")
        self.assertEqual(p.inflight, {"w0": 1, "w1": 0})

    def test_least_loaded_follows_the_load_and_the_weights(self):
        p = fresh(weights={"w1": 2.0})                 # w1 is twice as big
        for i in range(2):
            p.start(f"a{i}", "w0")
        p.start("b", "w1")
        self.assertEqual(p.least_loaded(), "w1")        # 1/2 < 2/1
        for i in range(4):
            p.start(f"c{i}", "w1")
        self.assertEqual(p.least_loaded(), "w0")        # 2/1 < 5/2

    def test_ties_are_broken_at_random_not_always_the_same_worker(self):
        p = fresh()
        self.assertEqual({p.least_loaded() for _ in range(40)}, {"w0", "w1"})


class Policies(unittest.TestCase):
    def test_litellm_policy_leaves_the_choice_to_the_router(self):
        self.assertIsNone(fresh().pick(Req("s"), "litellm"))

    def test_unknown_policy_is_an_error(self):
        for policy in ("nonsense", "random", "affinity", "least_loaded"):    # the experiment-only policies were removed
            with self.assertRaises(ValueError):
                fresh().pick(Req("s"), policy)

    def test_the_pick_function_delegates(self):
        p = fresh()
        p.start("r", "w0")
        self.assertEqual(place.pick(Req("s"), p, policy="affload"), "w1")

    def test_affload_keeps_the_session_on_its_worker_when_load_is_similar(self):
        p = fresh(slack=3)
        home = p.pick(Req("s"), "affload")
        p.start("a", home)
        p.start("b", home)                               # gap of 2 < slack
        self.assertEqual(p.pick(Req("s"), "affload"), home)

    def test_affload_moves_when_the_worker_is_busier_by_the_slack(self):
        p = fresh(slack=3)
        home = p.pick(Req("s"), "affload")
        other = "w1" if home == "w0" else "w0"
        for i in range(3):
            p.start(f"r{i}", home)                       # gap of 3 >= slack
        self.assertEqual(p.pick(Req("s"), "affload"), other)
        self.assertEqual(p._previous("s"), other)        # and the session now lives there

    def test_affload_sends_batch_to_the_least_loaded_and_does_not_pin_it(self):
        p = fresh(slack=3)
        for i in range(2):
            p.start(f"r{i}", "w0")
        self.assertEqual(p.pick(Req("b", priority=8, is_batch=True), "affload"), "w1")
        self.assertIsNone(p._previous("b"))

    def test_batch_ignores_the_affinity_of_a_known_session(self):
        p = fresh(slack=3)
        home = p.pick(Req("s"), "affload")
        other = "w1" if home == "w0" else "w0"
        p.start("r", home)                               # home is busier by 1: an interactive call would stay
        self.assertEqual(p.pick(Req("s"), "affload"), home)
        self.assertEqual(p.pick(Req("s", priority=8, is_batch=True), "affload"), other)

    def test_a_new_session_goes_to_the_least_loaded_worker(self):
        p = fresh()
        for i in range(2):
            p.start(f"r{i}", "w0")
        self.assertEqual(p.pick(Req("fresh"), "affload"), "w1")

    def test_sessions_expire(self):
        p = fresh()
        p.pick(Req("s"), "affload")
        with mock.patch.object(place.time, "monotonic", return_value=place.time.monotonic() + place.SESSION_TTL_S + 1):
            self.assertIsNone(p._previous("s"))

    def test_session_outcomes_are_counted(self):
        before = {r: place.SESSIONS.labels(r)._value.get() for r in ("new", "kept", "moved")}
        p = fresh(slack=1)
        home = p.pick(Req("s"), "affload")
        p.pick(Req("s"), "affload")
        p.start("x", home)
        p.pick(Req("s"), "affload")                      # gap 1 >= slack 1 -> moved
        after = {r: place.SESSIONS.labels(r)._value.get() for r in ("new", "kept", "moved")}
        self.assertEqual([after[r] - before[r] for r in ("new", "kept", "moved")], [1, 1, 1])

    def test_a_move_is_a_hop_and_staying_is_not(self):
        key = lambda s, d: place.HOPS.labels(s, d, place.HOP_BACKEND)._value.get()
        tok = lambda s, d: place.HOP_TOKENS.labels(s, d, place.HOP_BACKEND)._value.get()
        p = fresh(slack=2)
        home = p.pick(Req("s", prompt_tokens=40000), "affload")
        other = "w1" if home == "w0" else "w0"
        h0, t0 = key(home, other), tok(home, other)
        p.pick(Req("s", prompt_tokens=41000), "affload")                       # stays: no hop
        self.assertEqual(key(home, other), h0)
        for i in range(2):
            p.start(f"r{i}", home)                                              # gap 2 >= slack 2
        p.pick(Req("s", prompt_tokens=42000), "affload")                       # moves: hop home -> other
        self.assertEqual((key(home, other) - h0, tok(home, other) - t0), (1, 42000))

    def test_a_new_session_and_batch_are_never_hops(self):
        before = sum(c._value.get() for c in place.HOPS._metrics.values())
        p = fresh()
        p.pick(Req("fresh", prompt_tokens=1000), "affload")
        p.pick(Req("b", priority=8, is_batch=True, prompt_tokens=1000), "affload")
        self.assertEqual(sum(c._value.get() for c in place.HOPS._metrics.values()), before)


class QueueDepthScorer(unittest.TestCase):
    def test_equal_loads_are_broken_by_the_shorter_engine_queue_not_by_chance(self):
        for seed in range(20):
            p = place.Placement(workers=("w0", "w1"), rng=random.Random(seed))
            self.assertEqual(p.least_loaded({"w0": 3, "w1": 0}), "w1")
            self.assertEqual(p.least_loaded({"w0": 0, "w1": 5}), "w0")

    def test_a_clearly_lighter_worker_still_wins_over_a_shorter_queue(self):
        p = fresh()
        p.start("a", "w0")
        p.start("b", "w0")
        self.assertEqual(p.least_loaded({"w0": 0, "w1": 9}), "w1")          # load first, queue depth only breaks ties

    def test_without_telemetry_it_is_the_old_random_tie_break(self):
        self.assertEqual({fresh().least_loaded(None) for _ in range(1)} <= {"w0", "w1"}, True)
        p = fresh()
        self.assertEqual({p.least_loaded({}) for _ in range(40)}, {"w0", "w1"})

    def test_pick_uses_the_queue_depth_of_the_request(self):
        for seed in range(10):
            p = place.Placement(workers=("w0", "w1"), rng=random.Random(seed))
            self.assertEqual(p.pick(Req("s", queue_depth={"w0": 4, "w1": 0}), "affload"), "w1")


class WorkerDownAndShed(unittest.TestCase):
    def test_a_worker_that_does_not_answer_is_never_picked(self):
        p = fresh()
        p.observe({"w0"}, 0.0, now=0)
        for i in range(30):
            self.assertEqual(p.pick(Req(f"s{i}"), "affload"), "w0")

    def test_no_worker_answering_is_a_shed_with_a_reason_and_a_code(self):
        p = fresh()
        p.observe(set(), 0.0, now=0)
        result = p.pick(Req("s"), "affload")
        self.assertIsInstance(result, place.Shed)
        self.assertEqual((result.reason, result.code), ("no_healthy_worker", 503))
        self.assertIsNone(p.pick(Req("s"), "litellm"))                       # the router policy never sheds here

    def test_unknown_telemetry_means_everybody_is_assumed_up(self):
        p = fresh()
        p.observe(None, 0.0, now=0)
        self.assertEqual(p.available(), ["w0", "w1"])

    def test_a_session_whose_worker_went_down_moves_and_is_a_hop(self):
        p = fresh()
        home = p.pick(Req("s"), "affload")
        other = "w1" if home == "w0" else "w0"
        p.observe({other}, 0.0, now=0)
        before = place.HOPS.labels(home, other, place.HOP_BACKEND)._value.get()
        self.assertEqual(p.pick(Req("s", prompt_tokens=10), "affload"), other)
        # its worker is gone, so the session is treated as new there: no hop is invented from a dead worker
        self.assertEqual(place.HOPS.labels(home, other, place.HOP_BACKEND)._value.get(), before)


class Ramp(unittest.TestCase):
    def make(self, **kw):
        p = place.Placement(workers=("w0", "w1"), rng=random.Random(1), **kw)
        p.ramp_seconds, p.ramp_floor, p.ramp_hold_p99_s = 60.0, 0.25, 10.0
        return p

    def comes_back(self, p, at=0.0):
        p.observe({"w0"}, 0.0, now=at - 1)       # w1 is down
        p.observe({"w0", "w1"}, 0.0, now=at)     # w1 answers again: the ramp starts

    def test_a_returning_worker_starts_at_the_floor_and_reaches_full_weight_in_the_ramp_time(self):
        p = self.make()
        self.comes_back(p)
        self.assertAlmostEqual(p.ramp_factor("w1"), 0.25)
        self.assertEqual(p.ramp_factor("w0"), 1.0)
        p.observe({"w0", "w1"}, 0.0, now=30)
        self.assertAlmostEqual(p.ramp_factor("w1"), 0.625)                     # halfway
        p.observe({"w0", "w1"}, 0.0, now=60)
        self.assertEqual(p.ramp_factor("w1"), 1.0)                             # done: no ramp state left
        self.assertNotIn("w1", p._progress)

    def test_while_ramping_the_worker_looks_more_loaded_so_it_gets_less_traffic(self):
        p = self.make()
        self.comes_back(p)
        for i in range(2):
            p.start(f"a{i}", "w0")
            p.start(f"b{i}", "w1")
        self.assertGreater(p.load("w1"), p.load("w0"))                         # same requests, a quarter of the weight
        self.assertEqual(p.least_loaded(), "w0")

    def test_the_ramp_waits_while_the_fleet_p99_is_above_the_hold(self):
        p = self.make()
        self.comes_back(p)
        p.observe({"w0", "w1"}, 25.0, now=30)                                  # p99 25 s > 10 s: the clock does not run
        self.assertAlmostEqual(p.ramp_factor("w1"), 0.25)
        p.observe({"w0", "w1"}, 4.0, now=60)                                   # healthy again: 30 s count
        self.assertAlmostEqual(p.ramp_factor("w1"), 0.625)

    def test_a_worker_that_drops_again_loses_its_ramp_and_restarts_it_on_return(self):
        p = self.make()
        self.comes_back(p)
        p.observe({"w0", "w1"}, 0.0, now=30)
        p.observe({"w0"}, 0.0, now=31)
        self.assertNotIn("w1", p._progress)
        p.observe({"w0", "w1"}, 0.0, now=40)
        self.assertAlmostEqual(p.ramp_factor("w1"), 0.25)

    def test_the_first_observation_never_starts_a_ramp(self):
        p = self.make()
        p.observe({"w0", "w1"}, 0.0, now=0)
        self.assertEqual((p.ramp_factor("w0"), p.ramp_factor("w1")), (1.0, 1.0))

    def test_ramp_counter_counts_returns(self):
        p = self.make()
        before = place.RAMPS.labels("w1")._value.get()
        self.comes_back(p)
        self.assertEqual(place.RAMPS.labels("w1")._value.get() - before, 1)


if __name__ == "__main__":
    unittest.main()
