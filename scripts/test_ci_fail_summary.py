#!/usr/bin/env python3
"""scripts/ci_fail_summary.sh over fake `go test -json` files shaped like go's own: the failed tests come first whatever
passed around them, a failed test's output is printed as text, a package that failed with no failing test shows its panic
(a timeout names the tests that were running) or how its output ended, and nothing it meets makes it fail the step.
Run: python3 -B scripts/test_ci_fail_summary.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SH = os.path.join(HERE, "ci_fail_summary.sh")
MOD = "github.com/townsendmerino/goinfer"
sys.path.insert(0, HERE)
import ci_flakes  # noqa: E402


def ev(action, test=None, pkg="decoder", out=None, **more):
    """One event as go test -json writes it: Time, Action, Package, Test, Elapsed, Output, then the rest."""
    d = {"Time": "2026-11-27T16:39:47.243047-08:00", "Action": action, "Package": MOD + "/" + pkg if pkg != "." else MOD}
    if test:
        d["Test"] = test
    if out is None:
        d["Elapsed"] = 0.1
    else:
        d["Output"] = out
    d.update(more)
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def build(action, pkg, out=None):
    d = {"ImportPath": f"{MOD}/{pkg} [{MOD}/{pkg}.test]", "Action": action}
    if out is not None:
        d["Output"] = out
    return json.dumps(d, separators=(",", ":"))


def failing(test, pkg="decoder", lines=("    x_test.go:7: got 1, want 2\n",)):
    """A failed test's events, in go's order: run, its output, then the fail event."""
    return ([ev("run", test, pkg), ev("output", test, pkg, f"=== RUN   {test}\n", OutputType="frame")] +
            [ev("output", test, pkg, l) for l in lines] +
            [ev("output", test, pkg, f"--- FAIL: {test} (0.00s)\n", OutputType="frame"), ev("fail", test, pkg)])


def passing(test, pkg="decoder", lines=()):
    return ([ev("run", test, pkg)] + [ev("output", test, pkg, l) for l in lines] +
            [ev("output", test, pkg, f"--- PASS: {test} (0.00s)\n"), ev("pass", test, pkg)])


def run(events, env=None, path=None):
    """(exit code, stdout) of the script over these events; path is the file to read instead."""
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "test.json")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("\n".join(events) + "\n")
        e = dict(os.environ)
        e.pop("GITHUB_STEP_SUMMARY", None)
        e.pop("GITHUB_JOB", None)
        e.update(env or {})
        p = subprocess.run(["bash", SH, path or f], capture_output=True, text=True, env=e, encoding="utf-8")
        return p.returncode, p.stdout + p.stderr


class Names(unittest.TestCase):
    def test_a_green_file_prints_nothing(self):
        rc, out = run(passing("TestA") + [build("build-output", "decoder", "ld: warning: something\n"), ev("pass")])
        self.assertEqual((rc, out), (0, ""))

    def test_the_failed_tests_come_first_whatever_passed_around_them(self):
        noise = [f"    a_test.go:{i}: row {i} FAIL-looking panic text\n" for i in range(2000)]
        rc, out = run(passing("TestNoisy", lines=noise) + failing("TestBroken") + failing("TestTable/case_2") + failing("TestTable") +
                      passing("TestLater", lines=noise) + [ev("fail"), ev("pass", pkg="chat")])
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines()[:4], ["== FAILED: 3 tests; 1 packages, 0 of them with no failing test ==",
                                                "--- FAIL: TestBroken (decoder)", "--- FAIL: TestTable/case_2 (decoder)",
                                                "--- FAIL: TestTable (decoder)"])
        self.assertNotIn("row 5 FAIL-looking", out, "a passing test's output is not printed")

    def test_a_package_is_named_by_its_path_in_the_module(self):
        _, out = run(failing("TestRoot", pkg=".") + failing("TestDeep", pkg="internal/runtime/mem") + [ev("fail", pkg="."), ev("fail", pkg="internal/runtime/mem")])
        self.assertIn("--- FAIL: TestRoot (.)", out)
        self.assertIn("--- FAIL: TestDeep (internal/runtime/mem)", out)

    def test_an_events_field_order_does_not_matter(self):
        e = json.dumps({"Action": "fail", "Test": "TestX", "Elapsed": 0, "Package": MOD + "/chat"}, separators=(",", ":"))
        _, out = run([e])
        self.assertIn("--- FAIL: TestX (chat)", out)

    def test_a_subtest_name_with_a_quote_or_a_non_ascii_letter(self):
        _, out = run(failing('TestT/quote"d') + failing("TestT/ünï") + [ev("fail")])
        self.assertIn('--- FAIL: TestT/quote"d (decoder)', out)
        self.assertIn("--- FAIL: TestT/ünï (decoder)", out)
        self.assertIn('== TestT/quote"d (decoder), 3 lines of output ==', out)

    def test_the_flake_record_reads_the_names_from_the_log(self):
        hang = ["panic: test timed out after 30m0s\n", "\trunning tests:\n", "\t\tTestHangs (29m59s)\n", "\n", "goroutine 65 [running]:\n"]
        _, out = run(failing("TestBroken") + failing("TestTable/case_2") + [ev("fail")] +
                     [ev("run", "TestHangs", "serveapp")] + [ev("output", "TestHangs", "serveapp", l) for l in hang] + [ev("fail", pkg="serveapp")])
        log = "\n".join("2026-11-27T04:04:58.8168616Z " + l for l in out.splitlines())
        self.assertEqual(ci_flakes.failed_tests_from_log(log), (["TestBroken", "TestHangs", "TestTable/case_2"], ""))


class Output(unittest.TestCase):
    def test_a_failed_tests_output_is_printed_as_text(self):
        lines = ['    x_test.go:6: about to compare <a> & "b"\tafter a tab\n', "    x_test.go:7: got 1, want 2 (path C:\\temp\\new)\n"]
        _, out = run(failing("TestBroken", lines=lines) + [ev("fail")])
        self.assertIn('== TestBroken (decoder), 4 lines of output ==\n=== RUN   TestBroken\n    x_test.go:6: about to compare <a> & "b"\tafter a tab\n'
                      "    x_test.go:7: got 1, want 2 (path C:\\temp\\new)\n--- FAIL: TestBroken (0.00s)\n", out)

    def test_a_long_output_keeps_its_head_and_its_tail(self):
        lines = [f"    x_test.go:9: line {i}\n" for i in range(500)] + ["    x_test.go:11: the end\n"]
        _, out = run(failing("TestLong", lines=lines) + [ev("fail")])
        self.assertIn("== TestLong (decoder), 503 lines of output ==", out)
        self.assertIn("x_test.go:9: line 58\n... 303 lines not shown ...\n    x_test.go:9: line 362\n", out)
        self.assertIn("x_test.go:9: line 499\n    x_test.go:11: the end\n--- FAIL: TestLong (0.00s)\n", out)

    def test_an_output_exactly_at_the_cut_is_whole(self):
        lines = [f"    x_test.go:9: line {i}\n" for i in range(198)]  # 198 + RUN + FAIL = 200 = head + tail
        _, out = run(failing("TestEdge", lines=lines) + [ev("fail")])
        self.assertNotIn("not shown", out)
        body = out.split("== TestEdge (decoder), 200 lines of output ==\n")[1].splitlines()[:200]
        self.assertEqual(body, ["=== RUN   TestEdge"] + [f"    x_test.go:9: line {i}" for i in range(198)] + ["--- FAIL: TestEdge (0.00s)"])

    def test_a_line_that_would_be_a_workflow_command_is_indented(self):
        _, out = run(failing("TestPrints", lines=["::group::never closed\n", "::error::not ours\n"]) + [ev("fail")])
        self.assertIn("\n ::group::never closed\n ::error::not ours\n", out)
        self.assertEqual([l for l in out.splitlines() if l.startswith("::")], ["::error::--- FAIL: TestPrints (decoder)"])

    def test_a_build_failure_shows_the_compiler_output(self):
        _, out = run([build("build-output", "chat", f"# {MOD}/chat [{MOD}/chat.test]\n"), build("build-output", "chat", "chat/x_test.go:6:12: undefined: gone\n"),
                      build("build-fail", "chat"), ev("output", pkg="chat", out=f"FAIL\t{MOD}/chat [build failed]\n"),
                      ev("fail", pkg="chat", FailedBuild=f"{MOD}/chat [{MOD}/chat.test]")] + passing("TestA") + [ev("pass")])
        self.assertIn("== FAILED: 0 tests; 1 packages, 1 of them with no failing test ==\nFAIL: package chat (did not build)\n", out)
        self.assertIn("== BUILD OUTPUT ==\n# github.com/townsendmerino/goinfer/chat [github.com/townsendmerino/goinfer/chat.test]\nchat/x_test.go:6:12: undefined: gone\n", out)

    def test_a_timeout_names_the_tests_that_were_running_and_shows_the_panic(self):
        dump = ["panic: test timed out after 30m0s\n", "\trunning tests:\n", "\t\tTestHangs (29m59s)\n", "\t\tTestAlso/sub (12s)\n", "\n",
                "goroutine 65 [running]:\n"] + [f"frame {i}\n" for i in range(400)]
        events = (passing("TestFine", "serveapp") + [ev("run", "TestHangs", "serveapp")] + [ev("output", "TestHangs", "serveapp", l) for l in dump] +
                  [ev("output", pkg="serveapp", out=f"FAIL\t{MOD}/serveapp\t1800.1s\n"), ev("fail", pkg="serveapp")])
        _, out = run(events)
        self.assertEqual(out.splitlines()[:4], ["== FAILED: 0 tests; 1 packages, 1 of them with no failing test ==",
                                                "--- FAIL: TestHangs (serveapp) was running when the package timed out",
                                                "--- FAIL: TestAlso/sub (serveapp) was running when the package timed out",
                                                "FAIL: package serveapp (timed out)"])
        self.assertIn("== package serveapp: its last panic outside a failed test, the first 50 lines ==\npanic: test timed out after 30m0s\n\trunning tests:\n", out)
        self.assertIn("frame 43\n", out)
        self.assertNotIn("frame 44\n", out)

    def test_a_package_that_exits_outside_a_test_shows_how_its_output_ended(self):
        events = [ev("output", pkg="cmd/gate", out=f"line {i}\n") for i in range(100)] + [ev("output", pkg="cmd/gate", out="exit status 3\n"), ev("fail", pkg="cmd/gate")]
        _, out = run(events)
        self.assertIn("FAIL: package cmd/gate (no failing test: a panic or an exit outside a test)", out)
        self.assertIn("== package cmd/gate: the last 60 of 101 lines of output ==\nline 41\n", out)
        self.assertTrue(out.rstrip().endswith("line 99\nexit status 3"), out[-200:])

    def test_only_the_last_panic_of_a_package_is_kept(self):
        events = ([ev("output", pkg="chat", out="panic: an expected one, recovered by a subprocess\n")] + [ev("output", pkg="chat", out=f"between {i}\n") for i in range(70)] +
                  [ev("output", pkg="chat", out="fatal error: unexpected signal\n"), ev("output", pkg="chat", out="[signal SIGSEGV]\n"), ev("fail", pkg="chat")])
        _, out = run(events)
        self.assertIn("== package chat: its last panic outside a failed test, the first 2 lines ==\nfatal error: unexpected signal\n[signal SIGSEGV]", out)
        self.assertNotIn("between 3\n", out)

    def test_a_package_with_a_failed_test_and_a_timeout_shows_both(self):
        hang = ["panic: test timed out after 30m0s\n", "\trunning tests:\n", "\t\tTestHangs (29m)\n", "\n", "goroutine 65 [running]:\n"]
        events = (failing("TestBroken", "serveapp") + [ev("run", "TestHangs", "serveapp")] + [ev("output", "TestHangs", "serveapp", l) for l in hang] +
                  [ev("fail", pkg="serveapp")])
        _, out = run(events)
        self.assertEqual(out.splitlines()[:4], ["== FAILED: 1 tests; 1 packages, 0 of them with no failing test ==", "--- FAIL: TestBroken (serveapp)",
                                                "--- FAIL: TestHangs (serveapp) was running when the package timed out", "FAIL: package serveapp (timed out)"])
        self.assertIn("== TestBroken (serveapp), 3 lines of output ==", out)
        self.assertIn("== package serveapp: its last panic outside a failed test, the first 5 lines ==\npanic: test timed out after 30m0s\n", out)

    def test_a_failed_tests_own_panic_is_printed_once_in_its_output(self):
        lines = ["--- FAIL: TestPanics (0.00s)\n", "panic: assignment to entry in nil map [recovered, repanicked]\n", "\n", "goroutine 19 [running]:\n"]
        events = ([ev("run", "TestPanics", "chat")] + [ev("output", "TestPanics", "chat", l) for l in lines] +
                  [ev("fail", "TestPanics", "chat"), ev("output", pkg="chat", out=f"FAIL\t{MOD}/chat\t0.5s\n"), ev("fail", pkg="chat")])
        _, out = run(events)
        self.assertEqual(out.count("panic: assignment to entry in nil map"), 1)
        self.assertNotIn("its last panic", out)


class Step(unittest.TestCase):
    def test_the_step_summary_gets_the_names_and_not_the_output(self):
        with tempfile.TemporaryDirectory() as d:
            summary = os.path.join(d, "summary.md")
            _, out = run(failing("TestBroken") + [ev("fail")], env={"GITHUB_STEP_SUMMARY": summary, "GITHUB_JOB": "root-windows"})
            with open(summary) as f:
                self.assertEqual(f.read(), "```\n== FAILED: 1 tests; 1 packages, 0 of them with no failing test ==\n--- FAIL: TestBroken (decoder)\n```\n")
        self.assertIn("::error title=root-windows::--- FAIL: TestBroken (decoder)", out)

    def test_only_the_first_ten_failed_tests_are_annotations(self):
        events = [e for i in range(25) for e in failing(f"TestN{i}")] + [ev("fail")]
        _, out = run(events)
        self.assertEqual(len([l for l in out.splitlines() if l.startswith("::error")]), 10)
        self.assertEqual(len([l for l in out.splitlines() if l.startswith("--- FAIL: TestN") and l.endswith("(decoder)")]), 25, "every name is in the list")

    def test_it_never_fails_the_step(self):
        self.assertEqual(run([], path="/no/such/test.json"), (0, "ci_fail_summary: cannot read /no/such/test.json\n"))
        rc, out = run(["not json", '{"Action":"fail"}', '{"Action":"fail","Package":""}', "\x00\x01"])
        self.assertEqual(rc, 0, out)

    def test_without_awk_the_raw_fail_events_are_printed(self):
        with tempfile.TemporaryDirectory() as d:
            for tool in ("bash", "grep", "head"):
                os.symlink(shutil.which(tool), os.path.join(d, tool))
            rc, out = run(failing("TestBroken") + [ev("fail")], env={"PATH": d})
        self.assertEqual(rc, 0)
        self.assertIn("ci_fail_summary: no awk on PATH; the raw fail and build events follow", out)
        self.assertIn('"Action":"fail","Package":"github.com/townsendmerino/goinfer/decoder","Test":"TestBroken"', out)

    def test_every_awk_on_this_machine_prints_the_same(self):
        """The lanes that run this have different awks (gawk, mawk, the BSD one); whichever are installed here must agree."""
        events = (failing('TestT/quote"d', lines=["    x_test.go:6: <a> & \"b\"\tc:\\d\n"]) + [ev("fail")] +
                  [ev("output", "TestHangs", "serveapp", l) for l in ("panic: test timed out after 3s\n", "\trunning tests:\n", "\t\tTestHangs (3s)\n")] +
                  [ev("fail", pkg="serveapp"), build("build-output", "chat", "chat/x.go:1:1: bad\n"), build("build-fail", "chat"), ev("fail", pkg="chat", FailedBuild="x")])
        want = run(events)
        self.assertIn('--- FAIL: TestT/quote"d (decoder)', want[1])
        tried = []
        for name in ("gawk", "mawk", "original-awk", "busybox"):
            exe = shutil.which(name)
            if not exe:
                continue
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "awk"), "w") as f:
                    f.write(f'#!/bin/sh\nexec "{exe}" {"awk " if name == "busybox" else ""}"$@"\n')
                os.chmod(os.path.join(d, "awk"), 0o755)
                self.assertEqual(run(events, env={"PATH": d + os.pathsep + os.environ["PATH"]}), want, name)
            tried.append(name)
        print(f"  (awks compared with the default one: {', '.join(tried) or 'none installed'})", file=sys.stderr)


class Lanes(unittest.TestCase):
    def test_every_lane_that_counts_its_tests_in_bash_also_summarises_its_failures(self):
        with open(os.path.join(HERE, "..", ".github", "workflows", "ci.yml")) as f:
            ci = f.read()
        self.assertEqual(ci.count("bash scripts/ci_fail_summary.sh test.json"), ci.count("bash scripts/ci_skip_count.sh test.json"))
        self.assertGreaterEqual(ci.count("bash scripts/ci_fail_summary.sh test.json"), 4)


if __name__ == "__main__":
    unittest.main()
