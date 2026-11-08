"""Tests for scripts/apidiff_check.sh's classification of apidiff's incompatible lines (--classify).

A Hard-tier name's incompatible change is fatal, except the two lines testdata/apidiff/hard_tier_aliases.txt licenses for a
listed alias: "T: changed from T to <its package>.T", and a Hard-tier method whose signature text is unchanged and names the
alias, because only its result type's identity moved. Everything else about a Hard-tier name stays fatal. The lines are the
ones apidiff printed for DC1b (docs/completed/task-decoder-config-2026-11.md §8), and plants beside them.
"""
import os
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "apidiff_check.sh")
MOD = "github.com/townsendmerino/goinfer"

HARD = ("decoder Config\ndecoder (*Model).Config\ndecoder Options\ndecoder (*Model).Dims\ndecoder Options.Cfg\n"
        "decoder (*Model).Foo\ndecoder (*Model).Bar\ndecoder (*Model).Baz\ndecoder/config Config\ndecoder/config Config.*\n")
ALIASES = "# header\ndecoder Config = decoder/config.Config\n"


def classify(lines, hard=HARD, aliases=ALIASES, pkg="decoder"):
    with tempfile.TemporaryDirectory() as d:
        hp, ap = os.path.join(d, "hard.txt"), os.path.join(d, "aliases.txt")
        with open(hp, "w") as f:
            f.write(hard)
        with open(ap, "w") as f:
            f.write(aliases)
        env = dict(os.environ, APIDIFF_HARD=hp, APIDIFF_ALIASES=ap)
        r = subprocess.run(["bash", SCRIPT, "--classify", pkg], input="\n".join(lines) + "\n",
                           capture_output=True, text=True, env=env)
        verdicts = [l.split(" ", 1)[0] for l in r.stdout.splitlines()]
        return r.returncode, verdicts


class ClassifyTest(unittest.TestCase):
    def test_listed_alias_is_compatible(self):
        rc, v = classify([f"- Config: changed from Config to {MOD}/decoder/config.Config"])
        self.assertEqual((rc, v), (0, ["ALIAS"]))

    def test_method_whose_text_is_unchanged_through_the_alias_is_compatible(self):
        rc, v = classify(["- (*Model).Config: changed from func() *Config to func() *Config"])
        self.assertEqual((rc, v), (0, ["ALIAS"]))

    def test_unlisted_alias_stays_fatal(self):
        rc, v = classify([f"- Options: changed from Options to {MOD}/decoder/opts.Options"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_listed_entry_with_a_different_name_stays_fatal(self):
        rc, v = classify([f"- Config: changed from Config to {MOD}/decoder/config.Settings"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_listed_name_into_another_package_stays_fatal(self):
        rc, v = classify([f"- Config: changed from Config to {MOD}/decoder/other.Config"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_method_whose_text_changed_stays_fatal(self):
        rc, v = classify(["- (*Model).Config: changed from func() *Config to func(int) *Config"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_unchanged_text_naming_no_listed_alias_stays_fatal(self):
        rc, v = classify(["- (*Model).Dims: changed from func() Dims to func() Dims"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_unchanged_text_on_a_field_stays_fatal(self):
        rc, v = classify(["- Options.Cfg: changed from Config to Config"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_listed_alias_removed_stays_fatal(self):
        rc, v = classify(["- Config: removed"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_without_the_list_the_alias_is_fatal(self):
        rc, v = classify([f"- Config: changed from Config to {MOD}/decoder/config.Config"], aliases="")
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_method_with_another_type_in_its_signature_stays_fatal(self):
        # Options' identity could have moved too, and the text would not show it.
        rc, v = classify(["- (*Model).Foo: changed from func(Options) *Config to func(Options) *Config"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_method_taking_the_alias_stays_fatal(self):
        rc, v = classify(["- (*Model).Bar: changed from func(*Config) int to func(*Config) int"])
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_method_returning_the_alias_with_builtin_params_is_compatible(self):
        rc, v = classify(["- (*Model).Baz: changed from func(int, string) *Config to func(int, string) *Config"])
        self.assertEqual((rc, v), (0, ["ALIAS"]))

    def test_a_removed_field_of_the_aliased_type_is_fatal(self):
        # The promise moves with the type: decoder/config's Config keeps every field and method decoder.Config had.
        rc, v = classify(["- Config.MaxPositions: removed"], pkg="decoder/config")
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_a_changed_method_of_the_aliased_type_is_fatal(self):
        rc, v = classify(["- (*Config).EOSIDs: changed from func() []int to func() []int64"], pkg="decoder/config")
        self.assertEqual((rc, v), (1, ["HARD"]))

    def test_the_wildcard_covers_members_only(self):
        rc, v = classify(["- ConfigX: removed", "- Configs.A: removed"], pkg="decoder/config")
        self.assertEqual((rc, v), (0, ["EXPERIMENTAL", "EXPERIMENTAL"]))

    def test_experimental_names_stay_informational(self):
        rc, v = classify([f"- ActKind: changed from ActKind to {MOD}/internal/runtime/program.ActKind"])
        self.assertEqual((rc, v), (0, ["EXPERIMENTAL"]))


if __name__ == "__main__":
    unittest.main()
