#!/usr/bin/env python3
"""bench_peer.py's archive check covers the checkpoints the run selects, and only those: a selected model whose path resolves to the
archive refuses, one the run does not select cannot block it, and a selected MoE model's own bundle and the vision checkpoint are
checked too. Run: python3 -B scripts/test_bench_peer_benchdisk.py"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer  # noqa: E402  (import has no side effects: main() runs only under __main__)


class SelectedOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="benchdisk-")
        self.archive = os.path.join(self.tmp, "srv", "models")
        os.makedirs(self.archive)
        self.local = os.path.join(self.tmp, "models")
        os.makedirs(self.local)

        def on_archive(name):
            real = os.path.join(self.archive, name)
            open(real, "w").close()
            link = os.path.join(self.local, name)
            os.symlink(real, link)  # the shape on nobara: ~/models/<name> -> /srv/models/<name>
            return link

        def on_nvme(name):
            p = os.path.join(self.local, name)
            open(p, "w").close()
            return p

        models = {"near": (on_nvme("near.gguf"), "t1"), "far": (on_archive("far.gguf"), "t2"),
                  "moe": (on_nvme("moe.gguf"), "t3")}
        self.patches = [
            mock.patch.object(bench_peer, "ARCHIVE_ROOTS", (self.archive,)),
            mock.patch.object(bench_peer, "MODELS", models),
            mock.patch.object(bench_peer, "GOINFER_MOE_PATH", {"moe": on_archive("moe.giw")}),
            mock.patch.object(bench_peer, "VISION_MODEL_PATH", on_archive("vision")),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def check(self, **env):
        with mock.patch.dict(os.environ, {"BENCH_VISION": "", **env}):
            bench_peer.check_bench_disk(bench_peer.selected_bench_paths())

    def test_an_unselected_archive_model_does_not_block(self):
        self.check(BENCH_MODELS="near")  # returns: "far" is on the archive but this run does not load it

    def test_a_selected_archive_model_still_refuses(self):
        with self.assertRaises(SystemExit) as cm:
            self.check(BENCH_MODELS="near,far")
        self.assertIn("REFUSED", str(cm.exception))
        self.assertIn("far.gguf", str(cm.exception))
        self.assertNotIn("near.gguf", str(cm.exception))

    def test_a_selected_moe_models_own_bundle_is_checked(self):
        with self.assertRaises(SystemExit) as cm:
            self.check(BENCH_MODELS="moe")
        self.assertIn("moe.giw", str(cm.exception))

    def test_the_vision_checkpoint_is_checked_when_phase_e_is_on(self):
        self.check(BENCH_MODELS="near")
        with self.assertRaises(SystemExit) as cm:
            self.check(BENCH_MODELS="near", BENCH_VISION="1")
        self.assertIn("vision", str(cm.exception))

    def test_preflight_calls_it_with_the_selection(self):
        seen = []
        with mock.patch.object(bench_peer, "check_bench_disk", lambda paths: seen.append(list(paths))), \
                mock.patch.object(bench_peer, "IDLE_GATE", "load"), mock.patch.object(bench_peer, "_loadavg", lambda: (0.1, 0.1, 0.1)), \
                mock.patch.object(bench_peer, "_gpu_compute_apps", lambda: []), mock.patch.object(bench_peer, "_gpu_state", lambda: None), \
                mock.patch.object(bench_peer, "check_free_mem", lambda: None), \
                mock.patch.dict(os.environ, {"BENCH_MODELS": "near", "BENCH_VISION": ""}):
            bench_peer.preflight()
        self.assertEqual(seen, [[bench_peer.MODELS["near"][0]]])


if __name__ == "__main__":
    unittest.main()
