package main

import (
	"fmt"
	"io"
	"strings"
)

// outcome classifies a cell and, after reconcile, a whole run. The format is aikit's tools/gate and tack's tools/gate:
// a cell is ok, n/a, FAIL or INCONCLUSIVE, and the run ends with one VERDICT line. Keep the three identical.
type outcome string

const (
	ok           outcome = "ok"
	na           outcome = "n/a"
	fail         outcome = "FAIL"
	inconclusive outcome = "INCONCLUSIVE"
)

type cell struct {
	name    string
	outcome outcome
	fields  []string // "key=value", printed on the cell's line
	detail  string   // printed indented under a cell that did not pass
}

type report struct {
	total, pass, fail, incon, na int
	outcome                      outcome
	exit                         int // 0 PASS, 1 FAIL, 2 INCONCLUSIVE
}

// reconcile tallies cells. A FAIL dominates an INCONCLUSIVE, so a broken build is not masked by a check that could not
// run; a run with no applicable cell is INCONCLUSIVE, since it judged nothing.
func reconcile(cells []cell) report {
	r := report{total: len(cells)}
	for _, c := range cells {
		switch c.outcome {
		case ok:
			r.pass++
		case na:
			r.na++
		case fail:
			r.fail++
		default:
			r.incon++
		}
	}
	switch {
	case r.fail > 0:
		r.outcome, r.exit = fail, 1
	case r.incon > 0 || r.pass == 0:
		r.outcome, r.exit = inconclusive, 2
	default:
		r.outcome, r.exit = ok, 0
	}
	return r
}

func verdict(o outcome, msg string) string {
	word := string(o)
	if o == ok {
		word = "PASS"
	}
	return "VERDICT: " + word + " — " + msg
}

// printReport writes one line per cell, the detail of each cell that did not pass, and the VERDICT line.
func printReport(w io.Writer, name string, cells []cell) report {
	for _, c := range cells {
		line := fmt.Sprintf("%-13s %s", c.outcome, c.name)
		if len(c.fields) > 0 {
			line += " " + strings.Join(c.fields, " ")
		}
		fmt.Fprintln(w, line)
		if c.outcome != ok && c.outcome != na && strings.TrimSpace(c.detail) != "" {
			for _, l := range strings.Split(strings.TrimRight(c.detail, "\n"), "\n") {
				fmt.Fprintln(w, "    "+l)
			}
		}
	}
	r := reconcile(cells)
	fmt.Fprintln(w, verdict(r.outcome, fmt.Sprintf("%s: %d/%d applicable checks passed (%d failed, %d inconclusive, %d n/a)",
		name, r.pass, r.total-r.na, r.fail, r.incon, r.na)))
	return r
}
