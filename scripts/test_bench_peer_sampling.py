#!/usr/bin/env python3
"""bench_peer.py's sampled configs send temperature, top_p and top_k to every arm, so an engine or model default never
applies to one arm only. Run: python3 -B scripts/test_bench_peer_sampling.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402

FIELDS = ("temperature", "top_p", "top_k")


def arm_payloads(cfg):
    """The sampling fields each arm's request carries, built by the harness's own payload functions."""
    return {
        "goinfer": bench_peer.goinfer_payload("m", "p", cfg),  # goinfer_old sends the same payload
        "llamacpp": bench_peer.llamacpp_payload("p", cfg),
        "mlx": bench_peer.mlx_payload("m", "p", cfg),
        "ollama": bench_peer.ollama_payload("m", "p", cfg)["options"],
    }


class SampledConfigsSendAllThree(unittest.TestCase):
    def test_every_sampled_config_every_arm(self):
        sampled = [n for n, c in bench_peer.CONFIGS.items() if c.get("goinfer", {}).get("temperature", 0) > 0]
        self.assertGreaterEqual(len(sampled), 7, "the sampled configs went missing: this test checks nothing")
        for name in sampled:
            for arm, body in arm_payloads(bench_peer.CONFIGS[name]).items():
                for f in FIELDS:
                    with self.subTest(config=name, arm=arm, field=f):
                        self.assertIn(f, body, f"{name}: the {arm} arm's request omits {f}, so its engine's default applies")

    def test_arms_agree_on_the_values(self):
        for name, cfg in bench_peer.CONFIGS.items():
            if cfg.get("goinfer", {}).get("temperature", 0) <= 0:
                continue
            bodies = arm_payloads(cfg)
            for f in FIELDS:
                with self.subTest(config=name, field=f):
                    self.assertEqual(len({bodies[a][f] for a in bodies}), 1, f"{name}: the arms send different {f}")


if __name__ == "__main__":
    unittest.main()
