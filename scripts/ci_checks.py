#!/usr/bin/env python3
"""ci_checks.py — DERIVE the repo-hygiene command set from .github/workflows/ci.yml.

B0. CI went red on `staticcheck -tags cuda` and stayed red for three commits, because the local
sequence (gofmt, build, test) and CI's check set are different and **nothing declares the
relationship**. `scripts/gpu_gate.sh` group 5 had the identical gap from the other side: it ran
gofmt and vet but not staticcheck, so running the gate — the thing you run *instead of*
remembering — would not have caught it either.

Adding staticcheck to one script fixes the instance, not the class. The class closes only if the
gate's list is DERIVED from CI's, so that the next check CI gains cannot reopen it silently.

This prints one TSV row per hygiene-class step CI runs:

    job <TAB> step-name <TAB> kind <TAB> env <TAB> command

`env` matters as much as the command. CI's root job has NO go.work — the module-boundary guard is
meaningful precisely because the root module graph is checked in isolation — while the gpu/cuda jobs
create one with `go work init`. A developer box usually has a go.work committed, so running the root
job's guard here without `GOWORK=off` reports a FALSE RED: every submodule appears in the graph
because the workspace unions them. Found exactly that way, the first time this ran.

So the environment is derived too: a job that contains a `workspace` step runs with the workspace,
and a job that does not runs with `GOWORK=off`. Reproducing the command without reproducing the
environment is not reproducing the check.

`kind` is `local` for steps the gate can execute here, or `runner:<reason>` for steps that need
something only the GitHub runner provides. The distinction is emitted rather than applied, because
a script that silently drops what it cannot run is the B0a shape — a check that skips and a check
that passes must not look the same.

Deliberately NOT a general YAML-to-shell translator: it selects the hygiene class (formatting,
build, vet, staticcheck, and the module-boundary guard) and leaves test execution to the gate's own
groups, which already run suites with their own timeouts and reporting.
"""

import re
import sys
import pathlib

try:
    import yaml
except ImportError:  # pragma: no cover - environment guard
    sys.stderr.write("ci_checks.py: PyYAML not available; cannot DERIVE the CI check set.\n")
    sys.stderr.write("Install it, or the gate must report this as a skip rather than pass.\n")
    sys.exit(2)

# Hygiene class, matched on the step's name. Test steps are excluded on purpose: the gate runs
# suites in its own groups with their own timeouts, and duplicating them here would double a
# 25-minute race run.
HYGIENE = re.compile(r"gofmt|staticcheck|^vet|^build|cleanliness|lint", re.I)
SKIP = re.compile(r"^test|^install|^workspace|^checkout|^setup|^sampler gates", re.I)

# Steps whose command cannot be run as-is off a GitHub runner, with the reason. Anything matching
# is emitted as runner:<reason> rather than dropped.
RUNNER_ONLY = [
    (re.compile(r"\bgh\b|GITHUB_|actions/"), "uses GitHub Actions context"),
]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    ci = root / ".github" / "workflows" / "ci.yml"
    if not ci.exists():
        sys.stderr.write(f"ci_checks.py: {ci} not found\n")
        return 2
    doc = yaml.safe_load(ci.read_text())
    rows = 0
    for job_name, job in (doc.get("jobs") or {}).items():
        steps = job.get("steps") or []
        # Derived, not hardcoded: a job that sets up a go workspace runs its checks inside one, and
        # a job that does not must run them with GOWORK=off or a developer box's committed go.work
        # silently changes the answer.
        uses_workspace = any(
            re.match(r"^workspace", (st.get("name") or "").strip(), re.I) for st in steps
        )
        # "-" rather than an empty field: bash's `read` treats tab as IFS whitespace and COLLAPSES
        # consecutive delimiters, so an empty column silently shifts every field after it. Found by
        # the shifted output, not by reasoning about it.
        env = "-" if uses_workspace else "GOWORK=off"
        for step in steps:
            name = (step.get("name") or "").strip()
            run = step.get("run")
            if not name or not run:
                continue
            if SKIP.search(name) or not HYGIENE.search(name):
                continue
            kind = "local"
            for pat, reason in RUNNER_ONLY:
                if pat.search(run):
                    kind = "runner:" + reason
                    break
            # Collapse to one line; multi-line `run:` blocks are shell scripts and stay verbatim
            # apart from the newline encoding, which the caller decodes.
            cmd = run.strip().replace("\n", "\\n")
            print(f"{job_name}\t{name}\t{kind}\t{env}\t{cmd}")
            rows += 1
    if rows == 0:
        # A zero-row derivation is indistinguishable from "CI has no hygiene checks", which is the
        # zero-means-either shape. Fail loudly instead.
        sys.stderr.write("ci_checks.py: derived ZERO hygiene steps — the workflow shape changed "
                         "and the selector no longer matches. Refusing to report an empty set.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
