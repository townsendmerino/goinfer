#!/usr/bin/env python3
"""bench_peer.py's arms are like for like: every sampled config sends temperature, top_p, top_k and the same seed to every arm
and min_p 0 to Ollama and llama-server; every arm gets the same context; and a run whose arms load different quant classes is
refused. Run: python3 -B scripts/test_bench_peer_sampling.py"""
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402

FIELDS = ("temperature", "top_p", "top_k", "seed")


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

    def test_ollama_and_llamacpp_get_min_p_0_on_sampled_configs(self):
        for name, cfg in bench_peer.CONFIGS.items():
            if not bench_peer.is_sampled(cfg):
                continue
            arms = arm_payloads(cfg)
            for arm in ("ollama", "llamacpp"):
                with self.subTest(config=name, arm=arm):
                    self.assertEqual(arms[arm].get("min_p"), 0.0, f"{name}: {arm} would apply its own min_p")


    def test_arms_agree_on_the_values(self):
        for name, cfg in bench_peer.CONFIGS.items():
            if cfg.get("goinfer", {}).get("temperature", 0) <= 0:
                continue
            bodies = arm_payloads(cfg)
            for f in FIELDS:
                with self.subTest(config=name, field=f):
                    self.assertEqual(len({bodies[a][f] for a in bodies}), 1, f"{name}: the arms send different {f}")

class EveryArmGetsTheSameContext(unittest.TestCase):
    def test_goinfer_ollama_and_llamacpp(self):
        ctx = bench_peer.cell_ctx()
        g = bench_peer.goinfer_cmd("serve", "m.gguf", "cuda", [])
        l = bench_peer.llamacpp_cmd("m.gguf", [])
        o = bench_peer.ollama_payload("m", "p", bench_peer.CONFIGS["greedy"])["options"]
        self.assertIn("-ctx", g, "goinfer is started at its own default context, not the peers'")
        self.assertEqual(int(g[g.index("-ctx") + 1]), ctx)
        self.assertEqual(int(l[l.index("--ctx-size") + 1]), ctx)
        self.assertEqual(o["num_ctx"], ctx)


def write_gguf(path, file_type):
    """A GGUF header with one key, general.file_type (u32)."""
    key = b"general.file_type"
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1))
        f.write(struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", file_type))


class QuantClassesMustMatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        gguf = os.path.join(self.tmp, "m.gguf")
        write_gguf(gguf, 15)  # Q4_K_M
        models = os.path.join(self.tmp, "ollama")
        os.makedirs(os.path.join(models, "blobs"))
        blob = os.path.join(models, "blobs", "sha256-abc")
        write_gguf(blob, 15)
        man = os.path.join(models, "manifests", "registry.ollama.ai", "library", "m-local")
        os.makedirs(man)
        with open(os.path.join(man, "latest"), "w") as f:
            json.dump({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc"}]}, f)
        self.saved = (dict(bench_peer.MODELS), bench_peer.OLLAMA_MODELS, dict(bench_peer.GOINFER_QUANT_OVERRIDE))
        bench_peer.MODELS["test-m"] = (gguf, "m-local")
        bench_peer.OLLAMA_MODELS = models

    def tearDown(self):
        bench_peer.MODELS.clear()
        bench_peer.MODELS.update(self.saved[0])
        bench_peer.OLLAMA_MODELS = self.saved[1]
        bench_peer.GOINFER_QUANT_OVERRIDE.clear()
        bench_peer.GOINFER_QUANT_OVERRIDE.update(self.saved[2])

    def test_the_peer_quant_is_read_from_the_tag_and_the_file(self):
        self.assertEqual(bench_peer.peer_quant("ollama", "test-m"), ("Q4_K_M", "4-bit"))
        self.assertEqual(bench_peer.peer_quant("llamacpp", "test-m"), ("Q4_K_M", "4-bit"))

    def test_an_8_bit_goinfer_against_a_4_bit_peer_is_refused(self):
        bench_peer.GOINFER_QUANT_OVERRIDE["test-m"] = "int8int8"
        why = bench_peer.quant_mismatch("test-m", ["goinfer", "ollama", "llamacpp"])
        self.assertIn("ollama loads Q4_K_M (4-bit)", why)

    def test_a_4_bit_goinfer_matches(self):
        for q in ("int4", "q4k"):
            bench_peer.GOINFER_QUANT_OVERRIDE["test-m"] = q
            self.assertEqual(bench_peer.quant_mismatch("test-m", ["goinfer", "ollama", "llamacpp"]), "", q)

    def test_a_quant_chosen_at_load_is_checked_from_the_decode_path(self):
        bench_peer.GOINFER_QUANT_OVERRIDE["test-m"] = "default"
        self.assertEqual(bench_peer.quant_mismatch("test-m", ["goinfer", "ollama"]), "", "nothing to compare before the load")
        label = bench_peer.decode_path_quant("cuda-resident (int8int8)")
        self.assertEqual(label, "int8int8")
        self.assertNotEqual(bench_peer.quant_mismatch("test-m", ["goinfer", "ollama"], label), "")

    def test_the_default_phi3_quant_matches_its_4_bit_peers(self):
        self.assertEqual(bench_peer.GOINFER_QUANT_CLASS[self.saved[2]["phi3-mini"]], "4-bit")


if __name__ == "__main__":
    unittest.main()
