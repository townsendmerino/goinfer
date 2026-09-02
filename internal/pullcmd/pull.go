// Package pullcmd is the `pull` subcommand shared by the goinfer binaries.
//
// It lives outside both chatapp and serveapp so `goinfer-chat pull` and `goinfer-serve pull`
// are one implementation with two dispatchers, rather than the same flags and error messages
// maintained twice. internal/modelpull stays the library underneath; this is only the CLI.
package pullcmd

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/internal/modelpull"
)

// pullUsage is printed for `pull -h` and on a malformed reference.
const pullUsage = `%[1]s pull — fetch a GGUF checkpoint from HuggingFace

  pull <owner/repo>              list the .gguf files in a repo
  pull <owner/repo>:<quant>      fetch that quant       (e.g. :q4_k_m — case-insensitive)
  pull <owner/repo>:<file.gguf>  fetch that exact file
  pull <ref> -embed [os/arch...] fetch, then bake the model INTO a single static binary
  pull demo:<tier>               fetch a model goinfer itself vets and pins (see below)

Examples:
  %[1]s pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
  %[1]s pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m
  %[1]s pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m -embed darwin/arm64 linux/amd64
  %[1]s pull demo:1.5b

demo: refs are shorthand for an exact repo and filename that this build PINS a digest for —
the same models the goinfer-chat-0.5b/-1.5b releases ship — so an upstream re-upload is
refused rather than silently substituted. Everything else is an explicit owner/repo, which
is the real interface; this is not a name registry.

The download is verified against the sha256 HuggingFace declares for the file before the
transfer starts, and lands in the user cache dir unless -o says otherwise. Point --model at
the result; a plain .gguf is transcoded to a sidecar .giw cache automatically on first use,
so there is no separate conversion step to run.

Anonymous only. A gated repo is detected up front and named, rather than failing after a
multi-gigabyte transfer — a community GGUF re-upload of the same model is usually ungated.
`

// runPull implements `goinfer-chat pull`. Returns a process exit code.
// Run executes `pull`. args excludes the program name and the "pull" word. Returns an exit code.
func Run(args []string) int {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, pullUsage, self()); fs.PrintDefaults() }
	outDir := fs.String("o", "", "directory to download into (default: <user cache>/goinfer/models/<owner>/<repo>)")
	embed := fs.Bool("embed", false, "after fetching, bake the model into a single static binary per target (default: the host). Requires a goinfer source checkout and a Go toolchain — it drives demo/chat/build-embed.sh, the same pipeline the released goinfer-chat-0.5b/1.5b binaries are built with")
	embedGGUF := fs.Bool("embed-gguf", false, "with -embed: bake the raw GGUF and quantize at launch (smaller binary, slower start, full-size weight heap) instead of the default prequant bundle (~5x faster cold start, ~10x less heap)")
	embedName := fs.String("name", "", "with -embed: output basename, producing <name>-<os>-<arch> (default: derived from the repo name)")
	// Go's flag package stops parsing at the first non-flag argument, so a bare
	// fs.Parse(args) silently treats `pull <ref> -embed` — the form the usage text above
	// documents, and the one anyone would type — as a ref plus a positional "-embed",
	// then prints usage. Lift a leading ref out first so BOTH orders work.
	var refArg string
	rest := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		refArg, rest = args[0], args[1:]
	}
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	targets := fs.Args()
	if refArg == "" {
		if fs.NArg() < 1 {
			fs.Usage()
			return 2
		}
		refArg, targets = fs.Arg(0), fs.Args()[1:]
	}
	// Trailing args are os/arch targets, and only -embed consumes them.
	if len(targets) > 0 && !*embed {
		fmt.Fprintf(os.Stderr, "goinfer-chat pull: unexpected argument %q (os/arch targets are only meaningful with -embed)\n", targets[0])
		return 2
	}

	ref, err := modelpull.ParseRef(refArg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "goinfer-chat pull: %v\n", err)
		return 2
	}

	// Ctrl-C cancels the transfer; the partial .part file is cleaned up by Download.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := modelpull.CheckAccess(ctx, ref.Repo); err != nil {
		fmt.Fprintf(os.Stderr, "goinfer-chat pull: %v\n", err)
		return 1
	}
	files, err := modelpull.List(ctx, ref.Repo)
	if err != nil {
		fmt.Fprintf(os.Stderr, "goinfer-chat pull: %v\n", err)
		return 1
	}

	// No selector ⇒ this is a listing request, not a failed fetch. Printing to stdout and
	// exiting 0 makes `pull <repo>` a usable discovery step rather than an error to decode.
	if ref.File == "" && ref.Quant == "" {
		fmt.Printf("%s — %d GGUF file(s):\n", ref.Repo, len(files))
		for _, f := range files {
			fmt.Printf("  %-52s %10s\n", f.Path, modelpull.HumanBytes(f.Size))
		}
		fmt.Printf("\nfetch one with:  %s pull %s:<quant>\n", self(), ref.Repo)
		return 0
	}

	f, err := modelpull.Select(files, ref)
	if err != nil {
		fmt.Fprintf(os.Stderr, "goinfer-chat pull: %v\n", err)
		return 1
	}

	dir := *outDir
	if dir == "" {
		if dir, err = modelpull.CacheDir(ref.Repo); err != nil {
			fmt.Fprintf(os.Stderr, "goinfer-chat pull: locating cache dir: %v\n", err)
			return 1
		}
	}

	fmt.Printf("%s\n  %s  (%s)\n  sha256 %s\n  -> %s\n",
		ref.Repo, f.Path, modelpull.HumanBytes(f.Size), shortSHA(f.SHA256), dir)

	start := time.Now()
	// A terminal gets a single carriage-return-updated line; a pipe or log file gets
	// newline-terminated lines at a slower cadence. Emitting \r into a non-TTY smears the
	// whole transfer onto one unreadable line, which is exactly the case where someone is
	// reading the output later to work out whether it stalled.
	tty := isTerminal(os.Stderr)
	interval := 5 * time.Second
	if tty {
		interval = time.Second
	}
	var lastPrint time.Time
	path, err := modelpull.Download(ctx, ref.Repo, f, dir, func(done, total int64) {
		el := time.Since(start).Seconds()
		if el <= 0 || (!tty && time.Since(lastPrint) < interval) {
			return
		}
		lastPrint = time.Now()
		rate := float64(done) / el
		line := fmt.Sprintf("  %s / %s  %s/s", modelpull.HumanBytes(done), modelpull.HumanBytes(total), modelpull.HumanBytes(int64(rate)))
		// ETA only once there is enough of a sample to mean anything, and only when the
		// total is known — a confidently wrong estimate is worse than none, because it
		// gets planned around.
		if total > 0 && done > 0 && el > 3 {
			eta := time.Duration(float64(total-done)/rate) * time.Second
			line += fmt.Sprintf("  eta %s", eta.Round(time.Second))
		}
		if tty {
			fmt.Fprintf(os.Stderr, "\r%-72s", line)
		} else {
			fmt.Fprintln(os.Stderr, line)
		}
	})
	if tty {
		fmt.Fprintln(os.Stderr)
	}
	if err != nil {
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "goinfer-chat pull: cancelled")
			return 130
		}
		fmt.Fprintf(os.Stderr, "goinfer-chat pull: %v\n", err)
		return 1
	}

	verified := "sha256 verified"
	if f.SHA256 == "" {
		// Small non-LFS files carry no oid. Say so rather than implying a check happened.
		verified = "no sha256 published by HuggingFace for this file — NOT verified"
	}
	fmt.Printf("done in %s — %s\n%s\n", time.Since(start).Round(time.Second), verified, path)
	if !*embed {
		fmt.Printf("\nrun it:\n  %s --model %s\n", self(), path)
		return 0
	}
	return runEmbed(path, ref, *embedName, *embedGGUF, targets)
}

// runEmbed bakes the freshly-pulled model into a standalone binary by driving
// demo/chat/build-embed.sh — the SAME script that builds the released goinfer-chat-0.5b and
// -1.5b assets.
//
// Calling it rather than reimplementing it is deliberate. The staging rules (the asset must
// sit next to the //go:embed directive and symlinks are not followed), the prequant-vs-raw
// build tag, and the CGO_ENABLED=0 -trimpath cross-compile line are all decisions that already
// have one owner. A Go reimplementation here would be a second copy of them, and the copy is
// what goes stale — the failure this repo keeps re-encountering with duplicated facts.
//
// -embed therefore needs a source checkout and a Go toolchain, which is the right constraint:
// the person BUILDING a distributable is a developer, and the person RECEIVING it needs
// nothing at all. That asymmetry is the whole point of the feature.
func runEmbed(model string, ref modelpull.Ref, name string, raw bool, targets []string) int {
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "\ngoinfer-chat pull -embed: %v\n", err)
		return 1
	}
	script := filepath.Join(root, "demo", "chat", "build-embed.sh")
	if _, err := exec.LookPath("go"); err != nil {
		fmt.Fprintln(os.Stderr, "\ngoinfer-chat pull -embed: no `go` toolchain on PATH — -embed cross-compiles, so it needs one")
		return 1
	}
	if name == "" {
		name = embedName(ref)
	}

	argv := []string{}
	if raw {
		argv = append(argv, "--gguf")
	}
	argv = append(argv, "--name", name, model)
	argv = append(argv, targets...)

	fmt.Printf("\nbaking %s into a standalone binary (%s)\n", filepath.Base(model), name)
	cmd := exec.Command(script, argv...)
	cmd.Dir = root
	cmd.Stdout, cmd.Stderr, cmd.Stdin = os.Stdout, os.Stderr, nil
	if err := cmd.Run(); err != nil {
		// The script is `set -euo pipefail`, so a non-zero exit already printed its own cause;
		// do not bury it under a wrapper message.
		fmt.Fprintf(os.Stderr, "%s pull -embed: build-embed.sh failed: %v\n", self(), err)
		return 1
	}
	fmt.Printf("\nbinaries are in %s\n", filepath.Join(root, "demo", "chat", "dist"))
	return 0
}

// repoRoot walks up from the working directory looking for the checkout, identified by the
// build script itself rather than by go.mod — go.mod matches ANY module, including one that
// merely depends on goinfer, and building the wrong tree would fail confusingly late.
func repoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if st, err := os.Stat(filepath.Join(dir, "demo", "chat", "build-embed.sh")); err == nil && !st.IsDir() {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("-embed needs a goinfer source checkout (it drives demo/chat/build-embed.sh) and none was found above %q.\n"+
				"  Run it from a clone: go run ./demo/chat pull <ref> -embed\n"+
				"  The model itself downloaded fine — only the bake step needs the tree", mustGetwd())
		}
		dir = parent
	}
}

func mustGetwd() string {
	d, err := os.Getwd()
	if err != nil {
		return "."
	}
	return d
}

// embedName derives the output basename from the repo, so `pull …-1.5B-Instruct-GGUF -embed`
// yields goinfer-chat-qwen2.5-coder-1.5b-instruct rather than clobbering a previous build.
// Sanitised to a conservative charset because it becomes a filename.
func embedName(ref modelpull.Ref) string {
	_, repo, _ := strings.Cut(ref.Repo, "/")
	repo = strings.ToLower(repo)
	repo = strings.TrimSuffix(repo, "-gguf")
	var b strings.Builder
	for _, r := range repo {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "goinfer-chat"
	}
	return "goinfer-chat-" + out
}

// self is the invoked binary's name, so the printed next-step command is one the user can
// actually paste (they may have renamed it, or be running `go run ./demo/chat`).
func self() string {
	if len(os.Args) > 0 && os.Args[0] != "" {
		return filepath.Base(os.Args[0])
	}
	return "goinfer-chat"
}

// isTerminal reports whether f is a character device. Uses only os.Stat's mode bits so the
// pure-Go / no-new-dependency property holds (golang.org/x/term would be a module addition
// for one bit of information).
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

func shortSHA(s string) string {
	if s == "" {
		return "(none published)"
	}
	if len(s) > 16 {
		return s[:16] + "…"
	}
	return s
}
