"""Fleet state: the snapshot admission decides on."""
import unittest
from unittest import mock

from tests._env import fleet_state as fs

INF = float("inf")


class Quantile(unittest.TestCase):
    def test_empty_or_zero_histograms_give_zero(self):
        self.assertEqual(fs._histogram_quantile({}, 0.5), 0.0)
        self.assertEqual(fs._histogram_quantile({1.0: 0, INF: 0}, 0.5), 0.0)

    def test_interpolates_inside_a_bucket(self):
        buckets = {0.5: 30, 1.0: 40, INF: 40}
        self.assertAlmostEqual(fs._histogram_quantile(buckets, 0.5), 20 / 30 * 0.5)
        self.assertAlmostEqual(fs._histogram_quantile(buckets, 0.99), 0.5 + 0.5 * (39.6 - 30) / 10)

    def test_a_tail_in_the_infinite_bucket_is_capped_at_the_last_finite_bound(self):
        self.assertEqual(fs._histogram_quantile({1.0: 10, 5.0: 10, INF: 20}, 0.99), 5.0)


class Properties(unittest.TestCase):
    def fleet(self, p50=0.0, p99=0.0, **replicas):
        f = fs.FleetState(replicas=[fs.ReplicaState("u0", "0", waiting=replicas.get("w0", 0), kv_usage=replicas.get("k0", 0.0)),
                                    fs.ReplicaState("u1", "1", waiting=replicas.get("w1", 0), kv_usage=replicas.get("k1", 0.0))])
        f.ttft_p50_s, f.ttft_p99_s = p50, p99
        return f

    def test_kv_is_the_worst_replica_and_waiting_is_the_sum(self):
        f = self.fleet(k0=0.4, k1=0.9, w0=3, w1=4)
        self.assertEqual(f.kv_usage_max, 0.9)
        self.assertEqual(f.waiting_total, 7)

    def test_estimated_wait_is_waiting_times_p50_and_zero_without_p50(self):
        self.assertAlmostEqual(self.fleet(p50=2.0, w0=3).estimated_queue_wait_s, 6.0)
        self.assertEqual(self.fleet(p50=0.0, w0=50).estimated_queue_wait_s, 0.0)

    def test_tail_is_very_bad_above_four_times_the_median_and_above_the_floor(self):
        floor = fs.cfg.VERY_BAD_TAIL_MIN_P99_S
        self.assertFalse(self.fleet(p50=floor, p99=4 * floor).very_bad_tail_latency)         # exactly 4x: not bad
        self.assertTrue(self.fleet(p50=floor, p99=4 * floor + 0.1).very_bad_tail_latency)
        self.assertFalse(self.fleet(p50=0.0, p99=100.0).very_bad_tail_latency)               # no p50 yet: no basis

    def test_a_wide_but_harmless_tail_is_not_very_bad(self):
        # p99 is 8x p50 but only 8 s: the normal shape of this workload, far from the 20 s SLO
        self.assertFalse(self.fleet(p50=1.0, p99=8.0).very_bad_tail_latency)


class Window(unittest.TestCase):
    def setUp(self):
        fs._ttft_history.clear()
        fleet = fs.global_fleet_state
        saved = (fleet.ttft_p50_s, fleet.ttft_p99_s)
        self.addCleanup(lambda: (fs._ttft_history.clear(), setattr(fleet, "ttft_p50_s", saved[0]),
                                 setattr(fleet, "ttft_p99_s", saved[1])))

    def update(self, now, buckets):
        replica = fs.ReplicaState("u0", "0", ttft_buckets=dict(buckets))
        with mock.patch.object(fs.time, "monotonic", return_value=now):
            fs._update_fleet_ttft([replica])
        return fs.global_fleet_state.ttft_p50_s, fs.global_fleet_state.ttft_p99_s

    def test_too_few_samples_give_no_decision_basis(self):
        self.update(0, {0.5: 0, 1.0: 0, INF: 0})
        self.assertEqual(self.update(10, {0.5: 5, 1.0: 8, INF: 8}), (0.0, 0.0))      # 8 < 20 samples

    def test_enough_samples_give_quantiles_of_the_window_only(self):
        self.update(0, {0.5: 100, 1.0: 100, INF: 100})                                # old, fast history
        p50, p99 = self.update(10, {0.5: 100, 1.0: 100, INF: 100})
        self.assertEqual((p50, p99), (0.0, 0.0))                                      # nothing new in the window
        p50, p99 = self.update(20, {0.5: 100, 1.0: 100, 5.0: 130, INF: 130})          # 30 new slow requests
        self.assertGreater(p50, 1.0)                                                  # the old fast ones do not dilute it

    def test_a_restarted_worker_does_not_produce_negative_counts(self):
        self.update(0, {0.5: 100, 1.0: 100, INF: 100})
        self.assertEqual(self.update(10, {0.5: 3, 1.0: 3, INF: 3}), (0.0, 0.0))

    def test_old_snapshots_leave_the_window(self):
        self.update(0, {0.5: 0, 1.0: 0, INF: 0})
        self.assertGreater(self.update(10, {0.5: 30, 1.0: 40, INF: 40})[0], 0.0)
        self.assertEqual(self.update(200, {0.5: 30, 1.0: 40, INF: 40}), (0.0, 0.0))   # same totals 190 s later


if __name__ == "__main__":
    unittest.main()
