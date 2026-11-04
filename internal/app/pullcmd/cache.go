package pullcmd

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/pull"
)

// RunCache is `goinfer-chat cache`: what the pull cache holds on disk (task-checkpoint-fetch-2026-09.md P8), one line per
// model with its size, its sidecars and whether its pull finished, then the path to pass to --model. On-disk and loaded
// are different questions; this answers the first. It reads sizes only, so it is fast on a large cache.
func RunCache(args []string) int {
	if len(args) > 0 {
		fmt.Fprintf(os.Stderr, "usage: %s cache   (lists the pull cache; takes no arguments)\n", self())
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
