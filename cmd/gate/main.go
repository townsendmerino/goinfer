package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"
)

// provenance is the header every gate prints before its verdict. It exists because the same
// invocation error produced a false "15 BLOCKER(S)" three separate times while the tree was fine:
// a gate that reports its FINDING without reporting WHAT IT EXAMINED cannot be debugged from its
// own output.
type provenance struct {
	Commit string
	Dirty  bool
	Date   string
	Host   string
	Fields [][2]string // gate-specific rows (models dir, run filter, packages, …)
}

func (p provenance) write(w io.Writer) {
	d := ""
	if p.Dirty {
		d = " +dirty"
	}
	fmt.Fprintf(w, "  repo        %s%s\n", p.Commit, d)
	fmt.Fprintf(w, "  date (UTC)  %s\n", p.Date)
	fmt.Fprintf(w, "  host        %s\n", p.Host)
	for _, f := range p.Fields {
		fmt.Fprintf(w, "  %-11s %s\n", f[0], f[1])
	}
	if p.Dirty {
		fmt.Fprintf(w, "  NOTE: working tree DIRTY — this verdict does not describe a committed state.\n")
	}
}

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
  gate heavy                                       the pure-Go HEAVY tier (real checkpoints)
  gate parity                                      the release-tag parity sweep (named-gate checkset)
  gate composition [-v]                            the sweep's coverage composition (family × quant × loader)
  gate selector                                    tests that EXIST vs tests a selector RUNS
  gate gpu                                         the pre-tag GPU correctness gate (cuda | metal)

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
	stream := fs.String("stream", "", "parse an already-recorded -json stream instead of running go test")
	if err := fs.Parse(rest); err != nil {
		return 2
	}

	// The sweep is a CHECKSET, not a tally: its decision is per-named-gate and it delegates to the
	// asset registry, the composition census and the gate ledger, so it owns its own entry point
	// rather than pretending to be a gateConfig with an unusual report.
	if name == "gpu" {
		return runGPU(w, *logDir)
	}
	if name == "selector" {
		return selectorCoverage(w)
	}
	if name == "composition" {
		return composition(w, sliceHas(rest, "-v"))
	}
	if name == "parity" {
		if *stream != "" {
			fmt.Fprintf(os.Stderr, "gate: parity has no -stream mode (it must run the cells to resolve assets)\n")
			return 2
		}
		return runParity(w, *logDir)
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
	switch cfg.Decision {
	case "census":
		prov.write(w)
		return reportCensus(w, cfg, res, os.Getenv("GOINFER_REQUIRE_FIXTURES") != "")
	default:
		return reportTally(w, cfg, res, cells, prov)
	}
}

func sliceHas(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
