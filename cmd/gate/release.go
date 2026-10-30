package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

// `gate release vX.Y.Z`: RELEASING.md's pre-flight as one command. Each read is a row that is `done`, `OWED` or `UNKNOWN` (a read that could
// not be made: offline, no gh), with the record that proves it or that is missing. It runs nothing long: it reads records and calls the
// functions the existing checks are built on, so the command and the checks cannot disagree. Exit 0 only when every row is done, 1 when
// any is owed or unknown, 2 when refused (a precondition for tagging is unmet, so neither answer would be true).
//
// The rows are graders (grade_release.go) of the reads this runner makes for them (liveRelEnv); with -logdir it records every
// read, and -replay regrades a record.
//
// Why it exists: v0.21.0 and v0.22.0 shipped with the Metal GPU-gate run and no recorded CUDA one, because that read was a sentence nobody counted.

func osRun(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

// ---- the command ----

var tagRE = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+$`)

// relRefusal returns why tagging cannot start, or "".
func relRefusal(e *relEnv) string {
	if !tagRE.MatchString(e.tag) {
		return fmt.Sprintf("%q is not a version tag like v0.23.0", e.tag)
	}
	if br, err := gitOut(e.root, "rev-parse", "--abbrev-ref", "HEAD"); err != nil || br != "main" {
		return "not on main (on " + br + ")"
	}
	if st, err := gitOut(e.root, "status", "--porcelain"); err != nil || st != "" {
		return "the working tree is not clean"
	}
	if _, err := gitOut(e.root, "rev-parse", "-q", "--verify", "refs/tags/"+e.tag); err == nil {
		return e.tag + " already exists"
	}
	return ""
}

func runRelease(args []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate release", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	offline := fs.Bool("offline", false, "do not use the network: the CI row reads UNKNOWN")
	logDir := fs.String("logdir", "", "write the run record (every read the rows made) here, for -replay")
	replay := fs.String("replay", "", "regrade a run record instead of reading the repository")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *replay != "" {
		return replayRelease(w, *replay)
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(w, "usage: gate release [-offline] [-logdir DIR] vX.Y.Z | gate release -replay FILE")
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintln(w, err)
		return 2
	}
	e := liveRelEnv(relEnv{root: root, tag: fs.Arg(0), offline: *offline})
	if why := relRefusal(e); why != "" {
		fmt.Fprintf(w, "gate release: refused: %s\n", why)
		return 2
	}
	prev, err := newestVersionTag(root, e.tag)
	if err != nil {
		fmt.Fprintf(w, "gate release: refused: %v\n", err)
		return 2
	}
	e.prev = prev
	code := reportRelease(w, e, releaseRows)
	if *logDir != "" {
		if err := os.MkdirAll(*logDir, 0o755); err != nil {
			fmt.Fprintf(w, "gate: the run record was not written: %v\n", err)
		} else {
			writeRunRecord(w, *logDir, &gateConfig{Name: "release", Decision: "rows"}, nil, gatherProvenance(nil), false,
				runRecordExtra{Release: &releaseInputs{Tag: e.tag, Prev: e.prev, Offline: e.offline, Looks: e.looks}})
		}
	}
	return code
}

// liveRelEnv makes e's reader live: each read is made against the repository at e.root (commands through e.run, osRun when
// nil), and the first answer to each distinct read is recorded in e.looks, which is all a replay can give back.
func liveRelEnv(e relEnv) *relEnv {
	if e.run == nil {
		e.run = osRun
	}
	p := &e
	seen := map[string]bool{}
	p.look = func(kind string, args ...string) (string, error) {
		out, err := relLiveRead(p, kind, args)
		if k := relLookupKey(kind, args); !seen[k] {
			seen[k] = true
			l := relLookup{Kind: kind, Args: args, Out: out}
			if err != nil {
				l.Err, l.Failed = err.Error(), true
			}
			p.looks = append(p.looks, l)
		}
		return out, err
	}
	return p
}

// relLiveRead makes one read of the kinds relEnv's rows ask for.
func relLiveRead(e *relEnv, kind string, args []string) (string, error) {
	switch kind {
	case "read":
		b, err := os.ReadFile(filepath.Join(e.root, args[0]))
		return string(b), err
	case "glob":
		files, err := filepath.Glob(filepath.Join(e.root, args[0]))
		if err != nil {
			return "", err
		}
		sort.Strings(files)
		for i, f := range files {
			if rel, err := filepath.Rel(e.root, f); err == nil {
				files[i] = filepath.ToSlash(rel)
			}
		}
		return strings.Join(files, "\n"), nil
	case "git":
		return gitOut(e.root, args...)
	case "run":
		return e.run(filepath.Join(e.root, args[0]), args[1], args[2:]...)
	case "records":
		var recs any
		var err error
		switch args[0] {
		case "gpu":
			recs, err = loadGPURecords(filepath.Join(e.root, gpuRecordDirDefault))
		case "parity":
			recs, err = loadParityRecords(filepath.Join(e.root, parityRecordDirDefault))
		case "quick":
			recs, err = loadQuickRecords(filepath.Join(e.root, quickRecordDirDefault))
		default:
			return "", fmt.Errorf("no record kind %q", args[0])
		}
		if err != nil {
			return "", err
		}
		b, err := json.Marshal(recs)
		return string(b), err
	case "parity-paths":
		mb, err := paritykey.ManifestJSON(paritykey.Dir(e.root))
		if err != nil {
			return "", err
		}
		paths, err := parityQualifyingPaths(mb)
		return strings.Join(paths, "\n"), err
	case "pending":
		b, err := json.Marshal(awaitingFirstConfirmation)
		return string(b), err
	case "changelog":
		var buf bytes.Buffer
		if rc := runChangelog(args, &buf); rc != 0 {
			return buf.String(), fmt.Errorf("gate changelog exited %d", rc)
		}
		return buf.String(), nil
	}
	return "", fmt.Errorf("no read of kind %q", kind)
}

// replayRelease regrades a release run record: the same report and exit code. A record that cannot be read, or that lacks a
// read a row asks for, is exit 2 and prints no report.
func replayRelease(w io.Writer, path string) int {
	rec, err := readRunRecord(path)
	if err != nil {
		fmt.Fprintf(w, "gate release -replay: %v\n", err)
		return 2
	}
	in := rec.Release
	if in == nil {
		fmt.Fprintf(w, "gate release -replay: %s is not a release run record\n", path)
		return 2
	}
	var missing []string
	e := &relEnv{tag: in.Tag, prev: in.Prev, offline: in.Offline, look: relReplayLook(in.Looks, &missing)}
	var buf bytes.Buffer
	code := reportRelease(&buf, e, releaseRows)
	if len(missing) > 0 {
		fmt.Fprintf(w, "gate release -replay: %s holds no answer for: %s\n", path, strings.Join(missing, "; "))
		return 2
	}
	w.Write(buf.Bytes())
	return code
}
