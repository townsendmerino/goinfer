// Command testcmdtags checks `go test` command lines against the tests they name: a test whose file the command's tags do not
// compile never runs, and the run still prints ok. Pass the lines as arguments or on stdin, one per line; run it from the
// repository root. It prints each finding and exits 1 on any.
//
//	go run ./cmd/testcmdtags "go test -count=1 ./backend/metal/ -run 'TestS18ShrinkCtxToFit' -v"
package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/internal/guard/testcmdtags"
)

func main() {
	lines := os.Args[1:]
	if len(lines) == 0 {
		sc := bufio.NewScanner(os.Stdin)
		for sc.Scan() {
			lines = append(lines, sc.Text())
		}
	}
	bad, checked := 0, 0
	for _, l := range lines {
		c, ok := testcmdtags.Parse(l)
		if !ok {
			continue
		}
		f, err := testcmdtags.Check(".", c)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testcmdtags: %v\n", err)
			os.Exit(2)
		}
		checked++
		for _, x := range f {
			fmt.Println(x)
			bad++
		}
	}
	if checked == 0 {
		fmt.Fprintln(os.Stderr, "testcmdtags: no go test command in the input")
		os.Exit(2)
	}
	if bad > 0 {
		os.Exit(1)
	}
	fmt.Printf("ok: %d command(s), every named test compiles under its tags\n", checked)
}
