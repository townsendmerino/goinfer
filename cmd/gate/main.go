package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"strings"
	"time"
)

func gatherProvenance(fields [][2]string) provenance {
	p := provenance{Commit: "?", Date: time.Now().UTC().Format(time.RFC3339), Fields: fields}
	if out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output(); err == nil {
		p.Commit = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("git", "status", "--porcelain").Output(); err == nil && len(strings.TrimSpace(string(out))) > 0 {
		p.Dirty = true
	}
	if out, err := exec.Command("uname", "-sm").Output(); err == nil {
		p.Host = strings.TrimSpace(string(out))
	}
	return p
}

func usage() {
	fmt.Fprintf(os.Stderr, `gate — one runner over `+"`go test -json`"+` for goinfer's tallying gates (QUEUE E8)

usage:
  gate census [-stream FILE] [-- go-test-args…]   PASS/SKIP/FAIL census, SKIPs bucketed by reason
  gate heavy                                       the pure-Go HEAVY tier (real checkpoints); -stream is refused, use -replay
  gate census|heavy|parity -replay FILE            grade the run record a run wrote (gate_<name>_run.json in its -logdir)
  gate parity [-record DIR]                        the release-tag parity sweep (named-gate checkset); writes a JSON record
  gate composition [-v]                            the sweep's coverage composition (family × quant × loader)
  gate selector                                    tests that EXIST vs tests a selector RUNS
  gate gpu [-record DIR]                           the pre-tag GPU correctness gate (cuda | metal); writes a JSON record
  gate cuda-heavy [-list] [-group a,b]             gpu's heavy main group, one process per model family: the partition, or
                                                   chosen groups run alone with each test binary's VmHWM and peak VmSwap
  gate ptx [-keep DIR]                             every cuda/testdata/*.ptx rebuilt from its .cu at the NVRTC it records, compared
                                                   byte for byte; needs NVRTC, no GPU (GOINFER_NVRTC_DIRS names the installs)
  gate release [-offline] vX.Y.Z                   the pre-flight as rows: done / OWED / UNKNOWN each, with its record; exit 0 only when all are done
  gate changelog [-base TAG] [-section S]          new routes, flags, families and the aikit pin since TAG that the CHANGELOG section does not name
  gate gpu-records [-head REV] [-dir DIR]          which backends have a record that still qualifies at REV (exit 1 if one does not)
  gate mutation <name> <file> <sed-expr> <cmd...>  prove a gate can FAIL: green -> mutate -> red -> restore -> green
  gate quick [-base REF] [-files a,b] [-count1] [-j N] [-select] [-v]
                                                   the day loop: gofmt, vet, staticcheck and the tests that can
                                                   observe the diff, one go test per package in parallel (TE7)
  gate identity <old-rev> <new-rev> [-backend cpu|metal|webgpu|cuda] [-families a,b] [-assets tiny|real]
               [-quant q,…] [-steps N] [-only asset,…] [-keep]
                                                   inherit validation by identity (TE6(b)): dump full logits of
                                                   each family's parity prompt at both revs, compare bytes
  gate identity <old-rev> <new-rev> -giw bytes|logits [-configs quant:target[:e4h],…] [-assets tiny|real] …
                                                   the same, through each rev's OWN transcode: every source's .giw
                                                   by sha256 (bytes), or the logits dumped from each side's own
                                                   .giw (logits)
  gate comments-only <base> [--worktree] [-no-checks] [-v]
                                                   a diff changed Go comments and docs and nothing else (code-comments
                                                   CC2): token streams equal, directives and Example comments byte-identical,
                                                   the moved-text report, then gofmt, go vet and the citation lint
  gate shadow [-json FILE] <before> <after>       VK5: every parity family judged by today's key and by the shadow key
                                                   (kernel fingerprints in place of the aikit pin) across one push
  gate ledger promote --gate G --value V --by YOU  record a person's confirmation of a gate's value (B14)
  gate ledger classify --gate G                    CONFIRMED | FIRST-RUN | SOURCE-CHANGED | UNKNOWN-GATE
  gate ledger reconcile [--gates a,b]              the ledger's three checks, as the parity sweep prints them
  gate ledger seed --log FILE --by YOU             bulk-seed PASSing gates from a sweep log (marked BULK-SEEDED)

census env:
  GOINFER_REQUIRE_FIXTURES=1   exit 1 if any missing-fixture skip (release ritual)
parity env:
  REALCKPT=0             skip the real-checkpoint gates
  EMIT_MANIFEST=1        fold measured PARITY_ROW lines into the manifest and re-render the matrix
  TIMEOUT                per-cell timeout (default 120m)
gpu env:
  GOINFER_GATE_BACKEND   cuda | metal (default: auto — nvidia-smi ⇒ cuda, darwin ⇒ metal)
  GOINFER_GATE_SKIP_HEAVY  skip the ~28min real-model tier (reported as a counted SKIP)
  GOINFER_NVRTC_DIRS     colon-separated NVRTC toolchain dirs for the PTX check (an OVERRIDE)
heavy env:
  GOINFER_GATE_MODELS    dir holding the real checkpoints (default: $HOME/models)
  GOINFER_HEAVY_RUN      `+"`go test -run`"+` regex to narrow the run (default: all)
  GOINFER_HEAVY_PKGS     space-separated packages (default: "./decoder/ ./internal/serveapp/")
  GOINFER_HEAVY_TIMEOUT  per-package timeout (default 120m)
common:
  -logdir DIR            where raw `+"`go test -json`"+` streams are written (default: $TMPDIR)

Exit: 0 green, 1 red, 2 refused (a precondition is unmet — neither green nor red would be true).
`)
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout))
}

func run(argv []string, w io.Writer) int {
	if len(argv) == 0 {
		usage()
		return 2
	}
	name := argv[0]
	rest := argv[1:]

	// `mutation` takes a verify COMMAND as trailing argv, which carries its own flags (-run, -count)
	// and must not be parsed as ours. Handled before the flagset sees anything.
	if name == "mutation" {
		return runMutation(rest, w)
	}
	// `ledger` has its own subcommands and flags (ported from scripts/gate_ledger.py).
	if name == "ledger" {
		return runLedger(rest, w)
	}
	// `quick` has its own flags (quick.go).
	if name == "quick" {
		return runQuick(rest, w)
	}
	// `comments-only` takes a rev and its own flags (comments_only.go).
	if name == "comments-only" {
		return runCommentsOnly(rest, w)
	}
	// `release` is RELEASING.md's pre-flight as rows (release.go).
	if name == "release" {
		return runRelease(rest, w)
	}
	// `changelog` compares the tree with a tag and lists what the CHANGELOG section does not name (changelog.go).
	if name == "changelog" {
		return runChangelog(rest, w)
	}
	// `gpu-records` reads the records `gate gpu` wrote (gpu_record.go).
	if name == "gpu-records" {
		return runGPURecords(rest, w)
	}
	// `cuda-heavy` runs the heavy tier's main-group processes alone (heavy_groups.go).
	if name == "cuda-heavy" {
		return runCUDAHeavy(rest, w)
	}
	// `shadow` takes two revs as positionals after its flags (shadow.go).
	if name == "shadow" {
		return runShadow(rest, w)
	}
	// `identity` takes two revs as positionals, interleaved with its flags (identity.go).
	if name == "identity" {
		return runIdentity(rest, w)
	}

	// Split at `--`: everything after it is verbatim `go test` args, not our flags.
	var passthrough []string
	for i, a := range rest {
		if a == "--" {
			passthrough = rest[i+1:]
			rest = rest[:i]
			break
		}
	}

	fs := flag.NewFlagSet("gate "+name, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Bool("v", false, "composition: print the per-gate table as well as the counts")
	logDir := fs.String("logdir", os.TempDir(), "directory for raw go test -json logs")
	recordDir := fs.String("record", gpuRecordDirDefault, "gpu and parity: directory the run's JSON record is written to (empty: none; parity's default is "+parityRecordDirDefault+"); the release command reads it")
	stream := fs.String("stream", "", "parse an already-recorded -json stream instead of running go test")
	replay := fs.String("replay", "", "census, heavy and parity: grade the run record a run wrote (gate_<name>_run.json in its -logdir) instead of running go test")
	keep := fs.String("keep", "", "ptx: directory each rebuilt .ptx that differs from its committed form is written to (empty: none)")
	if err := fs.Parse(rest); err != nil {
		return 2
	}

	// The sweep is a CHECKSET, not a tally: its decision is per-named-gate and it delegates to the
	// asset registry, the composition census and the gate ledger, so it owns its own entry point
	// rather than pretending to be a gateConfig with an unusual report.
	if name == "gpu" {
		return runGPU(w, *logDir, *recordDir)
	}
	if name == "ptx" {
		return runPTX(w, *keep)
	}
	if name == "selector" {
		return selectorCoverage(w)
	}
	if name == "composition" {
		return composition(w, sliceHas(rest, "-v"))
	}
	if name == "parity" {
		if *replay != "" {
			return replayParity(w, *replay)
		}
		if *stream != "" {
			fmt.Fprintf(os.Stderr, "gate: parity has no -stream mode (it must run the cells to resolve assets)\n")
			return 2
		}
		rec := parityRecordDirDefault // -record's default is the GPU gate's directory; the sweep has its own unless one was given
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "record" {
				rec = *recordDir
			}
		})
		return runParity(w, *logDir, rec)
	}

	var cfg *gateConfig
	var fields [][2]string
	switch name {
	case "census":
		cfg = censusConfig(passthrough)
		fields = [][2]string{{"matrix", cfg.Cells[0].Name}}
	case "heavy":
		cfg = heavyConfig()
		var pkgs []string
		for _, c := range cfg.Cells {
			pkgs = append(pkgs, c.Name)
		}
		runFilter := cfg.Cells[0].Run
		if runFilter == "" {
			runFilter = "(all)"
		}
		fields = [][2]string{
			{"models", cfg.Cells[0].Env["GOINFER_MODELS_DIR"]},
			{"run", runFilter},
			{"packages", strings.Join(pkgs, " ")},
		}
	default:
		fmt.Fprintf(os.Stderr, "gate: unknown gate %q\n\n", name)
		usage()
		return 2
	}

	// Preconditions REFUSE rather than decide. With the assets absent, both "green" and "red"
	// would be lies, and exit 2 is how a caller tells "the gate ran and disagreed with you" from
	// "the gate could not run at all".
	if name == "heavy" && *stream != "" {
		// A stream carries no cell exit codes or durations, so it cannot reproduce a heavy run's verdict
		// (docs/measurements/sb12-2026-10.md §8).
		fmt.Fprintf(w, "gate heavy: -stream cannot reproduce a run (a stream has no cell exit codes or durations); "+
			"grade the run's record instead: gate heavy -replay FILE (gate_heavy_run.json in the run's -logdir)\n")
		return 2
	}
	if *replay != "" {
		rcfg, res, cells, prov, requireFixtures, err := loadRunRecord(*replay)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gate: %v\n", err)
			return 2
		}
		if rcfg.Name != cfg.Name {
			fmt.Fprintf(os.Stderr, "gate: %s is a record of %q, not %q\n", *replay, rcfg.Name, cfg.Name)
			return 2
		}
		return gradeTallyRun(w, rcfg, res, cells, prov, requireFixtures)
	}
	if cfg.Precondition != nil && *stream == "" {
		if why, ok := cfg.Precondition(); !ok {
			fmt.Fprintf(w, "%s== %s ==%s\n  %sREFUSED: %s%s\n", bold, cfg.Name, off, amber, why, off)
			return 2
		}
	}

	res := newResults()
	var cells []cellResult

	if *stream != "" {
		f, err := os.Open(*stream)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gate: %v\n", err)
			return 2
		}
		defer f.Close()
		res.consume(f)
		cells = append(cells, cellResult{Cell: cell{Name: *stream}, LogPath: *stream})
		fields = append(fields, [2]string{"stream", *stream})
	} else {
		for _, c := range cfg.Cells {
			// Every cell runs. A red cell is recorded and the loop continues — losing the count
			// on the first failure is precisely what `set -e` would have done, and what the shell
			// gates had to omit `set -e` to avoid.
			cells = append(cells, runCell(c, cfg, res, *logDir))
		}
	}

	prov := gatherProvenance(fields)
	requireFixtures := os.Getenv("GOINFER_REQUIRE_FIXTURES") != ""
	if *stream == "" {
		writeRunRecord(os.Stderr, *logDir, cfg, cells, prov, requireFixtures, runRecordExtra{})
	}
	return gradeTallyRun(w, cfg, res, cells, prov, requireFixtures)
}

func sliceHas(xs []string, want string) bool {
	return slices.Contains(xs, want)
}
