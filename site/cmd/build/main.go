// Command build writes goinfer.dev from the repository into a directory of plain static files.
//
//	cd site && GOWORK=off go run ./cmd/build -repo .. -out _site
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/site/internal/site"
)

func main() {
	repo := flag.String("repo", "..", "the goinfer repository root")
	out := flag.String("out", "_site", "where to write the site")
	drafts := flag.Bool("drafts", false, "also build the unreviewed writeups, for a preview (never for a deploy)")
	flag.Parse()
	cfg := site.DefaultConfig()
	cfg.Drafts = *drafts
	rep, err := site.Build(*repo, *out, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "site:", err)
		os.Exit(1)
	}
	fmt.Printf("site: %d families, %d files written to %s\n", rep.Families, len(rep.Files), *out)
}
