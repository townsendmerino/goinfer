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
	flag.Parse()
	rep, err := site.Build(*repo, *out, site.DefaultConfig())
	if err != nil {
		fmt.Fprintln(os.Stderr, "site:", err)
		os.Exit(1)
	}
	fmt.Printf("site: %d families, %d files written to %s\n", rep.Families, len(rep.Files), *out)
}
