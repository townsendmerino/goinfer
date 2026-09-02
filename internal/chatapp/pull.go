package chatapp

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/townsendmerino/goinfer/internal/modelpull"
)

// pullUsage is printed for `pull -h` and on a malformed reference.
const pullUsage = `goinfer-chat pull — fetch a GGUF checkpoint from HuggingFace

  pull <owner/repo>              list the .gguf files in a repo
  pull <owner/repo>:<quant>      fetch that quant       (e.g. :q4_k_m — case-insensitive)
  pull <owner/repo>:<file.gguf>  fetch that exact file

Examples:
  goinfer-chat pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
  goinfer-chat pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m

The download is verified against the sha256 HuggingFace declares for the file before the
transfer starts, and lands in the user cache dir unless -o says otherwise. Point --model at
the result; a plain .gguf is transcoded to a sidecar .giw cache automatically on first use,
so there is no separate conversion step to run.

Anonymous only. A gated repo is detected up front and named, rather than failing after a
multi-gigabyte transfer — a community GGUF re-upload of the same model is usually ungated.
`

// runPull implements `goinfer-chat pull`. Returns a process exit code.
func runPull(args []string) int {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, pullUsage); fs.PrintDefaults() }
	outDir := fs.String("o", "", "directory to download into (default: <user cache>/goinfer/models/<owner>/<repo>)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return 2
	}

	ref, err := modelpull.ParseRef(fs.Arg(0))
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
		fmt.Printf("\nfetch one with:  goinfer-chat pull %s:<quant>\n", ref.Repo)
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
	fmt.Printf("done in %s — %s\n%s\n\nrun it:\n  %s --model %s\n",
		time.Since(start).Round(time.Second), verified, path, self(), path)
	return 0
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
