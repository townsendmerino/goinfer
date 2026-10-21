"""The five tasks of I3 (docs/tasks/task-integration-harness-2026-10.md), verbatim, and their mechanical checks. A check
that needs the tree runs inside the sandbox on the finished fixture copy (the model wrote the code it would run)."""
import hashlib, json, os, random, re

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


CHAIN_LEN, CHAIN_DECOYS, CHAIN_LINES = 24, 40, 48


def make_chain(work):
    """I1-v2's `chain` (task doc, "I1 live gate, second design"): CHAIN_LEN files of about 1,000 tokens, each ending with the
    name of the next under names drawn for this run, the last ending with END and a code word; CHAIN_DECOYS decoys of the same
    shape chained only to one another. Every line starts with its file's own name, so a file's content names it. Writes the
    key (the chain in order, the decoys, the code word) beside the work directory, out of the agent's reach."""
    rng = random.Random(os.urandom(16))
    names = []
    while len(names) < CHAIN_LEN - 1 + CHAIN_DECOYS:
        n = f"{rng.randrange(16 ** 6):06x}"
        if n not in names:
            names.append(n)
    chain, decoys = ["start"] + names[:CHAIN_LEN - 1], names[CHAIN_LEN - 1:]
    code = "-".join(rng.choice(["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor"]) for _ in range(3)) + f"-{rng.randrange(1000):03d}"
    os.makedirs(os.path.join(work, "data"))

    def write(stem, tail):
        with open(os.path.join(work, "data", stem + ".txt"), "w") as f:
            for i in range(1, CHAIN_LINES + 1):
                f.write(f"{stem} record {i:03d}: node-{rng.randrange(997):03d} reported {rng.randrange(100003)} events in zone {chr(97 + rng.randrange(26))}\n")
            f.write(tail + "\n")
    for i, stem in enumerate(chain):
        write(stem, f"next: data/{chain[i + 1]}.txt" if i + 1 < len(chain) else f"END. The code word is {code}.")
    for i, stem in enumerate(decoys):
        write(stem, f"next: data/{decoys[(i + 1) % len(decoys)]}.txt")
    json.dump(dict(chain=chain, decoys=decoys, code=code), open(os.path.join(os.path.dirname(work), "chain-key.json"), "w"))


CHAIN_PROMPT = ("Start by reading data/start.txt. Each file ends with a line `next: <path>`: read that file next, one file per tool call, "
                "and keep following the chain until a file says END. Then answer with the code word that file gives.")

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
# The answer is checked against the run's own key (run_cell reads chain-key.json).
TASKS["chain"] = dict(prompt=CHAIN_PROMPT, setup=make_chain, keyed=True)
ORDER = ["read", "search", "edit", "write", "fix"]
