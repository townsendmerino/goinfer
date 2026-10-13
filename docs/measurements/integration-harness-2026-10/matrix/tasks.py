"""The five tasks of I3 (docs/tasks/task-integration-harness-2026-10.md), verbatim, and their mechanical checks. A check
that needs the tree runs inside the sandbox on the finished fixture copy (the model wrote the code it would run)."""
import hashlib, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixture")
PARSE_TEST_SHA = hashlib.sha256(open(os.path.join(FIXTURE, "parse_test.go.tmpl"), "rb").read()).hexdigest()


def copy_fixture(dst):
    """The fixture is stored as *.tmpl so that no tool of this repository reads it as a module or as packages of the root module."""
    os.makedirs(dst)
    for n in os.listdir(FIXTURE):
        open(os.path.join(dst, n[:-5] if n.endswith(".tmpl") else n), "wb").write(open(os.path.join(FIXTURE, n), "rb").read())


def inject_parse_bug(work):
    p = os.path.join(work, "parse.go")
    s = open(p).read()
    assert "\t\tf = strings.TrimSpace(f)\n" in s
    open(p, "w").write(s.replace("\t\tf = strings.TrimSpace(f)\n", ""))


def make_bulk(lines):
    """Eight data files for the overflow runs: `lines` records each, about 21 tokens a line under the Qwen2.5 tokenizer. 120 lines is
    about 2,500 tokens a file (reading them one after another grows a history step by step); 430 is about 9,000 (one read
    passes any window the matrix serves, whether or not the model keeps going)."""
    def setup(work):
        os.makedirs(os.path.join(work, "data"))
        for n in range(1, 9):
            with open(os.path.join(work, "data", f"part-{n}.txt"), "w") as f:
                for i in range(1, lines + 1):
                    f.write(f"part {n} record {i:03d}: node-{(n * 131 + i * 17) % 997:03d} reported {(n * 7919 + i * 104729) % 100003} events in zone {chr(97 + (n + i) % 26)}\n")
    return setup


TASKS = {
    "read": dict(prompt="What does `Add(2, 3)` in calc.go return? Answer with the number.",
                 answer=lambda a: re.search(r"(?<![\d.])5(?![\d.])", a) is not None),
    "search": dict(prompt="Which file defines `ErrNotFound`?", answer=lambda a: "store.go" in a),
    "edit": dict(prompt="Rename `sum` to `total` everywhere.",
                 check="go build ./... && ! grep -rqs 'sum(' --include='*.go' ."),
    "write": dict(prompt="Add a test for `Reverse` and make it pass.",
                  check="go test ./... && grep -qs Reverse *_test.go"),
    "fix": dict(prompt="`TestParse` fails; fix the code, not the test.", setup=inject_parse_bug,
                check="go test ./... && [ \"$(sha256sum parse_test.go | cut -d' ' -f1)\" = %s ]" % PARSE_TEST_SHA),
}
_BULK = "Show me the full contents of each of the files data/part-1.txt through data/part-8.txt, one file at a time, using your tools. After the last one, answer with the word done."
TASKS["bulk"] = dict(prompt=_BULK, setup=make_bulk(120), answer=lambda a: "done" in a.lower())
TASKS["bulkbig"] = dict(prompt=_BULK, setup=make_bulk(430), answer=lambda a: "done" in a.lower())
ORDER = ["read", "search", "edit", "write", "fix"]
