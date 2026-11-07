package pullcmd

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/pull"
)

// RunCache is `goinfer-chat cache`: what the pull cache holds on disk (task-checkpoint-fetch-2026-09.md P8), one line per
// model with its size, its sidecars and whether its pull finished, then the path to pass to --model. On-disk and loaded
// are different questions; this answers the first. It reads sizes only, so it is fast on a large cache.
// giwVersion is the .giw format the caller's build reads (decoder.GIWFormatVersion); prune-sidecars keeps it and anything newer.
func RunCache(args []string, giwVersion int) int {
	if len(args) > 0 && args[0] == "prune-sidecars" {
		return runPruneSidecars(args[1:], giwVersion)
	}
	if len(args) > 0 {
		fmt.Fprintf(os.Stderr, "usage: %[1]s cache   (lists the pull cache; takes no arguments)\n"+
			"       %[1]s cache prune-sidecars [-apply] [-dir <models dir>]   (stale .giw sidecar versions)\n", self())
		return 2
	}
	root, err := pull.CacheRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s cache: %v\n", self(), err)
		return 1
	}
	entries, err := pull.CacheEntries()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s cache: %v\n", self(), err)
		return 1
	}
	if len(entries) == 0 {
		fmt.Printf("The pull cache (%s) holds no models. Fetch one with `%s pull <owner/repo>:<quant>`.\n", root, self())
		return 0
	}
	fmt.Printf("%s\n", root)
	var total int64
	for _, e := range entries {
		fmt.Print(cacheLine(e))
		total += e.Bytes + e.Sidecars
	}
	fmt.Printf("\n%d model(s), %s on disk with their sidecars. Load one with --model <path>; re-run a pull to finish an incomplete one.\n", len(entries), pull.HumanBytes(total))
	return 0
}

// cacheLine is one model's line: repo, what it is, size, sidecars, completeness, and the --model path.
func cacheLine(e pull.CacheEntry) string {
	what := e.Kind
	if e.Kind == "split" {
		what = fmt.Sprintf("split, %d shards", e.Shards)
	}
	state := ""
	if !e.Complete {
		state = "  INCOMPLETE (re-run the pull to resume)"
	}
	side := ""
	if e.Sidecars > 0 {
		side = fmt.Sprintf(" + %s sidecar", pull.HumanBytes(e.Sidecars))
	}
	return fmt.Sprintf("  %-44s %-18s %10s%s%s\n    %s\n", e.Repo, what, pull.HumanBytes(e.Bytes), side, state, e.Path)
}

const pruneUsage = `%[1]s cache prune-sidecars — reclaim .giw sidecars in a format this build no longer reads

  %[1]s cache prune-sidecars [-apply] [-dir <models dir>]

A load transcodes a model once into a .giw sidecar beside it, named with the format version (".v%[2]d.giw" for this
build). A sidecar in an older format is never opened again, and stays on disk until removed. This lists them (a dry run)
and, with -apply, deletes them and their .verified markers. It never touches this build's version or a newer one, a
legacy unversioned name, or a transcode in progress. By default it looks in the pull cache; -dir looks under another
directory instead (sidecars sit beside their source model).
`

// runPruneSidecars is `cache prune-sidecars`: a dry run by default, -apply to delete.
func runPruneSidecars(args []string, giwVersion int) int {
	fs := flag.NewFlagSet("prune-sidecars", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, pruneUsage, self(), giwVersion); fs.PrintDefaults() }
	apply := fs.Bool("apply", false, "delete the stale sidecars listed (default: list them only)")
	dir := fs.String("dir", "", "look under this directory instead of the pull cache")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fs.Usage()
		return 2
	}
	root := *dir
	if root == "" {
		r, err := pull.CacheRoot()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s cache prune-sidecars: %v\n", self(), err)
			return 1
		}
		root = r
	}
	stale, err := pull.StaleSidecars(root, giwVersion)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s cache prune-sidecars: %v\n", self(), err)
		return 1
	}
	if len(stale) == 0 {
		fmt.Printf("No stale .giw sidecars under %s (this build reads v%d).\n", root, giwVersion)
		return 0
	}
	var freed int64
	failed := 0
	for _, s := range stale {
		note := ""
		if s.SharedWith != "" {
			note = fmt.Sprintf("  (the same file as %s: frees nothing)", s.SharedWith)
		}
		if *apply {
			if err := pull.RemoveStaleSidecar(s, giwVersion); err != nil {
				fmt.Printf("  FAILED  %s: %v\n", s.Path, err)
				failed++
				continue
			}
			fmt.Printf("  removed v%d  %10s  %s%s\n", s.Version, pull.HumanBytes(s.Bytes), s.Path, note)
		} else {
			fmt.Printf("  v%d  %10s  %s%s\n", s.Version, pull.HumanBytes(s.Bytes), s.Path, note)
		}
		freed += s.Freed()
	}
	switch {
	case !*apply:
		fmt.Printf("\n%d stale sidecar(s), %s to reclaim (this build reads v%d). Re-run with -apply to delete them.\n", len(stale), pull.HumanBytes(freed), giwVersion)
	case failed > 0:
		fmt.Printf("\n%d of %d removed; %d failed.\n", len(stale)-failed, len(stale), failed)
		return 1
	default:
		fmt.Printf("\n%d stale sidecar(s) removed, %s reclaimed.\n", len(stale), pull.HumanBytes(freed))
	}
	return 0
}
