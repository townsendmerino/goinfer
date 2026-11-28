#!/usr/bin/env bash
# What failed in a `go test -json` file, for a lane that prints nothing else (-json hides go test's plain output) and may have no python on
# PATH (the Windows lanes). In this order: one `--- FAIL: <Test> (<package>)` line per failed test, per test that was running when its
# package timed out, and one line per package that failed with no failing test (a build failure, a panic outside a test, a timeout); the
# build output; each failed test's own output; and a failed package's last panic outside those, or else, where no test failed, how its
# output ended. A green file prints nothing.
# It reports and never judges: the go test step's exit code fails the job, so this exits 0 whatever it finds or cannot read.
#   bash scripts/ci_fail_summary.sh test.json
# Tests: python3 -B scripts/test_ci_fail_summary.py
set -u
f=${1:?usage: ci_fail_summary.sh test.json}
[ -r "$f" ] || { echo "ci_fail_summary: cannot read $f"; exit 0; }
grep -q -E '"Action":"(fail|build-fail)"' "$f" || exit 0  # build output alone is a warning from a build that passed
raw() { echo "ci_fail_summary: $1; the raw fail and build events follow"; grep -E '"Action":"(fail|build-fail|build-output)"' "$f" | head -300; exit 0; }
command -v awk >/dev/null 2>&1 || raw "no awk on PATH"

# The fail events come first, on stdin: a fail event follows the output it explains, so which tests and packages failed must be known
# before the file is read for their output. A value is cut out of its event by key, so the order of an event's fields does not matter.
out=$(grep -F '"Action":"fail"' "$f" | awk -v MOD="github.com/townsendmerino/goinfer" -v JOB="${GITHUB_JOB:-}" '
BEGIN {
    HEAD = 60; TAIL = 140     # lines kept of one failed test: its first HEAD and its last TAIL
    PKGTAIL = 60; PANIC = 50  # of a failed package: the first PANIC lines of its last panic; with no failing test and no panic, its last PKGTAIL
    BUILD = 200; SHOWN = 40   # build output lines kept; failed tests whose output is printed
    NOTES = 10                # failed tests raised as annotations on the run
}

# field(s, key): the still-escaped value of "key":"..." in event s, or "". An escaped quote inside any value is \", so the pattern
# cannot match inside one.
function field(s, key,    re) {
    re = "\"" key "\":\"([^\"\\\\]|\\\\.)*\""
    if (!match(s, re)) return ""
    return substr(s, RSTART + length(key) + 4, RLENGTH - length(key) - 5)
}

# text(v): a JSON string value as the text it stands for, without its final newline.
function text(v,    out, i, n, c, h) {
    if (index(v, "\\") == 0) return v
    out = ""; n = length(v)
    for (i = 1; i <= n; i++) {
        c = substr(v, i, 1)
        if (c != "\\") { out = out c; continue }
        c = substr(v, ++i, 1)
        if (c == "n") out = out "\n"
        else if (c == "t") out = out "\t"
        else if (c == "r" || c == "b" || c == "f") continue
        else if (c == "u") {
            h = tolower(substr(v, i + 1, 4)); i += 4
            if (h == "003c") out = out "<"
            else if (h == "003e") out = out ">"
            else if (h == "0026") out = out "&"
            else if (h == "0027") out = out "\047"
            else if (h != "001b") out = out "?"  # an escape character is dropped: its colour code prints as text
        } else out = out c
    }
    return out
}

# line(v): text(v) as one log line. A leading "::" would be read by the runner as a workflow command, so it is indented.
function line(v) { v = text(v); sub(/\n$/, "", v); if (substr(v, 1, 2) == "::") v = " " v; return v }

function short(p) {
    if (p == MOD) return "."
    if (index(p, MOD "/") == 1) return substr(p, length(MOD) + 2)
    return p
}

pass == 1 {
    p = field($0, "Package"); t = field($0, "Test")
    if (p == "") next
    if (t != "") {
        if (!((p, t) in failed)) { failed[p, t] = 1; tp[++ntests] = p; tt[ntests] = t; withtest[p] = 1 }
    } else if (!(p in redpkg)) { redpkg[p] = 1; pk[++npkgs] = p; if (index($0, "\"FailedBuild\":")) nobuild[p] = 1 }
    next
}

index($0, "\"Action\":\"build-fail\"") { nbroken++; next }

index($0, "\"Action\":\"build-output\"") {
    if (++nbuild <= BUILD) build[nbuild] = line(field($0, "Output"))
    next
}

index($0, "\"Action\":\"output\"") {
    p = field($0, "Package")
    if (!(p in redpkg) && !(p in withtest)) next
    t = field($0, "Test")
    o = line(field($0, "Output"))
    if ((p, t) in failed) {  # a failed test that panicked has the panic in its own output
        n = ++seen[p, t]
        if (n <= HEAD) head[p, t, n] = o
        else tail[p, t, (n - HEAD - 1) % TAIL] = o
        next
    }
    # A panic that no failed test owns ends the test binary: a timeout, a crash in the runtime, a panic on another goroutine.
    if (o ~ /^(panic: |fatal error: )/) { npanic[p] = 0; nrun[p] = 0; inpanic[p] = 1; timedout[p] = (o ~ /^panic: test timed out/) }
    if (inpanic[p] && npanic[p] < PANIC) panic[p, ++npanic[p]] = o
    if (!(p in withtest)) ptail[p, pseen[p] % PKGTAIL] = o
    pseen[p]++
    # go test lists the tests that were running under its timeout panic: "\t\tTestName (12s)".
    if (timedout[p] && o ~ /^\t\tTest[^ ]* \(/ && nrun[p] < 5) { sub(/^\t\t/, "", o); sub(/ \(.*$/, "", o); running[p, ++nrun[p]] = o }
}

END {
    ntests += 0; npkgs += 0; nbuild += 0  # a count nothing incremented prints as a number
    if (ntests + npkgs + nbroken == 0) exit 0
    lone = 0
    for (i = 1; i <= npkgs; i++) if (!(pk[i] in withtest)) lone++
    # The first block, up to the empty line, is also the step summary.
    print "== FAILED: " ntests " tests; " npkgs " packages, " lone " of them with no failing test =="
    for (i = 1; i <= ntests; i++) print "--- FAIL: " text(tt[i]) " (" short(tp[i]) ")"
    for (i = 1; i <= npkgs; i++) {
        p = pk[i]
        for (j = 1; j <= nrun[p]; j++) print "--- FAIL: " running[p, j] " (" short(p) ") was running when the package timed out"
        if (p in nobuild) print "FAIL: package " short(p) " (did not build)"
        else if (timedout[p]) print "FAIL: package " short(p) " (timed out)"
        else if (!(p in withtest)) print "FAIL: package " short(p) " (no failing test: a panic or an exit outside a test)"
    }
    print ""

    # Annotations put the names on the run page and in `gh run view`, where the log is not open.
    note = "::error" (JOB != "" ? " title=" JOB : "") "::"
    for (i = 1; i <= ntests && i <= NOTES; i++) print note "--- FAIL: " text(tt[i]) " (" short(tp[i]) ")"
    for (i = 1; i <= npkgs; i++) {
        p = pk[i]
        for (j = 1; j <= nrun[p]; j++) print note "--- FAIL: " running[p, j] " (" short(p) ") was running when the package timed out"
        if (!(p in withtest)) print note "FAIL: package " short(p)
    }

    if (nbuild > 0) {
        print "== BUILD OUTPUT =="
        for (i = 1; i <= nbuild && i <= BUILD; i++) print build[i]
        if (nbuild > BUILD) print "... " (nbuild - BUILD) " more lines"
    }
    for (i = 1; i <= ntests && i <= SHOWN; i++) {
        p = tp[i]; t = tt[i]; n = seen[p, t] + 0
        print ""; print "== " text(t) " (" short(p) "), " n " lines of output =="
        for (j = 1; j <= n && j <= HEAD; j++) print head[p, t, j]
        if (n > HEAD + TAIL) print "... " (n - HEAD - TAIL) " lines not shown ..."
        for (j = (n > HEAD + TAIL ? n - TAIL + 1 : HEAD + 1); j <= n; j++) print tail[p, t, (j - HEAD - 1) % TAIL]
    }
    if (ntests > SHOWN) { print ""; print "... the output of " (ntests - SHOWN) " more failed tests is not shown; the names are above" }
    for (i = 1; i <= npkgs; i++) {
        p = pk[i]
        if (p in nobuild) continue
        n = pseen[p] + 0
        if (npanic[p] > 0) {
            print ""; print "== package " short(p) ": its last panic outside a failed test, the first " npanic[p] " lines =="
            for (j = 1; j <= npanic[p]; j++) print panic[p, j]
            continue
        }
        if (n == 0 || p in withtest) continue
        print ""; print "== package " short(p) ": the last " (n < PKGTAIL ? n : PKGTAIL) " of " n " lines of output =="
        for (j = (n > PKGTAIL ? n - PKGTAIL : 0); j < n; j++) print ptail[p, j % PKGTAIL]
    }
}
' pass=1 - pass=2 "$f") || raw "awk failed (exit $?)"
[ -n "$out" ] || exit 0
printf '%s\n' "$out"
# The step summary is appended here, not from awk: on a Windows runner its path holds backslashes.
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && { echo '```'; printf '%s\n' "${out%%$'\n\n'*}"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
exit 0
