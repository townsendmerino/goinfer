#!/usr/bin/env python3
"""Tests for migrate_citations_to_symbols.py's converter (CC0 step 2). The first real run converted only the FIRST number of
`path:241,456,464` and `path:270–299`, leaving `Name,456,464` and `Name–299` behind, which the lint cannot see; these pin the fix.
    python3 scripts/test_migrate_citations_to_symbols.py
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("migrate", os.path.join(HERE, "migrate_citations_to_symbols.py"))
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)

# a stand-in for classify(): lines 1-9 are in Alpha (a 10-line decl), 10-19 in Beta.run, 100-299 in a 200-line Long, 50 is outside everything
def fake_classify(rel, line, doc=None, first=True, prose=""):
    if rel.endswith(".sh"):
        return "hand", "no symbol form exists for this file type", None, "", "", None
    if 1 <= line <= 9:
        return "ok", "Alpha", 10, "alpha body", "cited", "Alpha"
    if 10 <= line <= 19:
        return "ok", "Beta.run", 10, "beta body", "cited", "Beta.run"
    if 100 <= line <= 299:
        return "ok", "Long", 200, "inside the long one", "cited", "Long"
    return "hand", "outside any declaration", None, "free comment", "", None


def conv(text, allow=()):
    return mig.convert_text(text, "docs/x.md", fake_classify, allow)


class TestConvertText(unittest.TestCase):
    def test_single_citation(self):
        out, ev = conv("see `pkg/a.go:5` here")
        self.assertEqual(out, "see `pkg/a.go:Alpha` here")
        self.assertEqual([e[0] for e in ev], ["ok"])

    def test_ascii_range_and_en_dash_range_are_one_unit(self):
        out, _ = conv("`pkg/a.go:12-15` and `pkg/a.go:12–15` and `pkg/a.go:150–299`")
        self.assertEqual(out, "`pkg/a.go:Beta.run` and `pkg/a.go:Beta.run` and `pkg/a.go:Long`")

    def test_a_range_that_starts_outside_every_declaration_is_left_whole(self):
        src = "`pkg/a.go:313–299`"  # a descending range some docs have: 313 is outside the stub's spans, so the whole citation waits for a human
        out, ev = conv(src)
        self.assertEqual(out, src)
        self.assertEqual([e[0] for e in ev], ["hand"])

    def test_comma_list_in_one_declaration_collapses_to_one_name(self):
        out, _ = conv("(`pkg/a.go:1,5,8`)")
        self.assertEqual(out, "(`pkg/a.go:Alpha`)")

    def test_comma_list_across_declarations_names_each_once_in_order(self):
        out, _ = conv("(`pkg/a.go:11,3,12,150`)")
        self.assertEqual(out, "(`pkg/a.go:Beta.run, pkg/a.go:Alpha, pkg/a.go:Long`)")

    def test_no_digit_of_a_list_or_range_is_left_behind(self):
        for src in ("pkg/a.go:1,5,8", "pkg/a.go:11-15", "pkg/a.go:11–15", "pkg/a.go:1,11-15,150"):
            out, _ = conv(f"x `{src}` y")
            self.assertNotRegex(out, r"\d", f"{src} -> {out}")

    def test_one_unresolvable_member_leaves_the_whole_list_untouched(self):
        src = "see `pkg/a.go:1,50,11`"
        out, ev = conv(src)
        self.assertEqual(out, src)  # half a list converted would orphan the rest
        self.assertEqual([e[0] for e in ev], ["hand"])

    def test_allowlisted_path_is_not_touched(self):
        src = "the placeholder `file.go:123`"
        out, ev = conv(src, allow={"file.go"})
        self.assertEqual(out, src)
        self.assertEqual(ev, [])

    def test_shell_line_citation_stays_a_line_number(self):
        src = "the script at `run.sh:12`"
        out, ev = conv(src)
        self.assertEqual(out, src)
        self.assertEqual([e[0] for e in ev], ["hand"])

    def test_idempotent(self):
        once, _ = conv("`pkg/a.go:5` `pkg/a.go:11-15` `pkg/a.go:1,11` `pkg/a.go:120`")
        twice, ev = conv(once)
        self.assertEqual(once, twice)
        self.assertEqual(ev, [])

    def test_review_row_only_for_a_declaration_longer_than_the_threshold(self):
        _, ev = conv("`pkg/a.go:5` and `pkg/a.go:120`")
        rows = [e[1] for e in ev if e[0] == "ok"]
        self.assertIsNone(rows[0])
        self.assertEqual(rows[1][:4], ("docs/x.md", "pkg/a.go:120", "pkg/a.go:Long", "200"))

    def test_a_list_row_is_marked_as_a_list(self):
        _, ev = conv("`pkg/a.go:120,5`")
        row = [e[1] for e in ev][0]
        self.assertIn("list", row[5])
        self.assertIn("2 declarations", row[5])

    def test_surrounding_prose_and_other_citations_are_untouched(self):
        src = "At `pkg/a.go:5` the plan is `decoder/x.go:Model.forward` (see https://example.com/a.go:5) and commit `abc1234`."
        out, _ = conv(src)
        self.assertEqual(out, src.replace("pkg/a.go:5`", "pkg/a.go:Alpha`", 1))


class TestPyEnclosing(unittest.TestCase):
    SRC = ["import os", "", "class Tool:", "    def run(self):", "        x = 1", "        return x", "", "    async def arun(self):", "        return 2", "",
           "def helper():", "    return 3", "", "VALUE = 4"]

    def test_method_is_class_dot_method(self):
        self.assertEqual(mig.py_enclosing(self.SRC, 5)[0], "Tool.run")
        self.assertEqual(mig.py_enclosing(self.SRC, 9)[0], "Tool.arun")

    def test_module_level_def(self):
        self.assertEqual(mig.py_enclosing(self.SRC, 12)[0], "helper")

    def test_outside_any_def_is_none(self):
        self.assertIsNone(mig.py_enclosing(self.SRC, 1))
        self.assertIsNone(mig.py_enclosing(self.SRC, 14))


if __name__ == "__main__":
    unittest.main()


class TestLocateByRecordedContent(unittest.TestCase):
    """The case that made the first conversion wrong: the index held the CONTENT of every cited line, and when that content moved the old lint
    accepted it and left the stale number in the prose. A number must not be trusted over what the citation was recorded as pointing at."""

    def setUp(self):
        self.lines = ["package p", "", "func First() {", "\tstep := 1", "\t_ = step", "}", "", "func Second() {", "\tneedle := 2", "\t_ = needle", "}"]
        self._saved = mig._INDEX
        mig._INDEX = {}

    def tearDown(self):
        mig._INDEX = self._saved

    def test_content_still_at_the_cited_line_is_cited(self):
        mig._INDEX = {"docs/x.md|pkg/a.go:9": "needle := 2"}
        self.assertEqual(mig.locate("docs/x.md", "pkg/a.go", 9, True, self.lines), (9, "cited"))

    def test_content_that_moved_is_found_where_it_is_now_not_at_the_stale_number(self):
        mig._INDEX = {"docs/x.md|pkg/a.go:4": "needle := 2"}  # the prose says line 4 (inside First); the recorded content is now at line 9 (inside Second)
        self.assertEqual(mig.locate("docs/x.md", "pkg/a.go", 4, True, self.lines), (9, "moved"))

    def test_content_found_twice_or_gone_is_not_guessed(self):
        mig._INDEX = {"docs/x.md|pkg/a.go:4": "this text is nowhere"}
        got, why = mig.locate("docs/x.md", "pkg/a.go", 4, True, self.lines)
        self.assertIsNone(got)
        self.assertIn("gone", why)
        lines2 = self.lines + ["\tneedle := 2"]
        mig._INDEX = {"docs/x.md|pkg/a.go:3": "needle := 2"}
        got, why = mig.locate("docs/x.md", "pkg/a.go", 3, True, lines2)
        self.assertIsNone(got)
        self.assertIn("2 places", why)

    def test_a_recorded_anchor_resolves_to_that_declaration(self):
        mig._INDEX = {"docs/x.md|pkg/a.go:4": "anchor: func Second() {"}
        self.assertEqual(mig.locate("docs/x.md", "pkg/a.go", 4, True, self.lines), (8, "anchor"))

    def test_no_record_means_unkeyed_and_later_list_numbers_are_always_unkeyed(self):
        self.assertEqual(mig.locate("docs/x.md", "pkg/a.go", 4, True, self.lines), (4, "unkeyed"))
        mig._INDEX = {"docs/x.md|pkg/a.go:4": "_ = step"}
        self.assertEqual(mig.locate("docs/x.md", "pkg/a.go", 9, False, self.lines), (9, "unkeyed"))

    def test_an_unkeyed_citation_needs_the_prose_to_name_the_declaration(self):
        self.assertTrue(mig._prose_names("Model.generateInto", "the loop in `generateInto` stops on cancel"))
        self.assertFalse(mig._prose_names("Model.generateInto", "the loop in `generate` stops on cancel"))  # a prefix of a longer name is not a mention
        self.assertFalse(mig._prose_names("normalize", "the loop stops on cancel"))
