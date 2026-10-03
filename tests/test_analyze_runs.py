"""The comparison used in the experiments: cache share by worker continuity."""
import importlib.util
import os
import unittest

from tests._env import ROOT

spec = importlib.util.spec_from_file_location("analyze_runs", os.path.join(ROOT, "script", "analyze_runs.py"))
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)


def rec(t, session, call, worker, prompt, cached):
    return {"t": t, "session": session, "call": call, "worker": worker, "prompt_tokens": prompt, "cached_tokens": cached,
            "status": 200, "error": None, "ttft": 1.0}


class Continuity(unittest.TestCase):
    def test_cached_share_is_split_by_whether_the_worker_changed(self):
        rs = [rec(1, 0, 1, "w0", 1000, 0),
              rec(2, 0, 2, "w0", 1000, 800),            # stayed: 800 of 1000
              rec(3, 0, 3, "w1", 1000, 200),            # moved:  200 of 1000
              rec(1, 1, 1, "w1", 1000, 0),
              rec(2, 1, 2, "w1", 1000, 600)]            # stayed: 600 of 1000
        same, other, _, _ = az.cached_by_continuity(rs)
        self.assertEqual((same, other), (70.0, 20.0))

    def test_the_first_call_of_a_turn_is_not_counted(self):
        rs = [rec(1, 0, 1, "w0", 1000, 0), rec(2, 0, 1, "w1", 1000, 0)]      # call == 1 both times
        same, other, _, _ = az.cached_by_continuity(rs)
        self.assertEqual((same, other), (None, None))

    def test_shed_and_unknown_worker_calls_are_ignored(self):
        shed = dict(rec(2, 0, 2, "w0", 1000, 900), status=503)
        unknown = dict(rec(3, 0, 3, None, 1000, 900))
        same, other, _, _ = az.cached_by_continuity([rec(1, 0, 1, "w0", 1000, 0), shed, unknown])
        self.assertEqual((same, other), (None, None))

    def test_percentile_helper(self):
        self.assertEqual(az.pct(list(range(1, 101)), 99), 99)
        self.assertIsNone(az.pct([], 50))


if __name__ == "__main__":
    unittest.main()
