package gatehygiene

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

const ledgerRel = "testdata/gate_ledger.json"

var ledgerRequired = []string{"gate", "value", "promoted_by", "date", "commit"}

type ledgerDoc struct {
	Entries []map[string]any `json:"entries"`
}

func ledgerPath(root string) string { return filepath.Join(root, ledgerRel) }

func loadLedger(root string) (*ledgerDoc, error) {
	b, err := os.ReadFile(ledgerPath(root))
	if os.IsNotExist(err) {
		return &ledgerDoc{Entries: []map[string]any{}}, nil
	}
	if err != nil {
		return nil, err
	}
	var d ledgerDoc
	if err := json.Unmarshal(b, &d); err != nil {
		return nil, fmt.Errorf("%s: %w", ledgerRel, err)
	}
	if d.Entries == nil {
		d.Entries = []map[string]any{}
	}
	return &d, nil
}

func encodeLedger(d *ledgerDoc) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(map[string]any{"entries": d.Entries}); err != nil {
		return nil, err
	}
	return asciiEscape(buf.Bytes()), nil
}

func asciiEscape(b []byte) []byte {
	var out bytes.Buffer
	for len(b) > 0 {
		r, n := utf8.DecodeRune(b)
		switch {
		case r < 0x80:
			out.WriteByte(b[0])
		case r > 0xFFFF:
			r -= 0x10000
			fmt.Fprintf(&out, `\u%04x\u%04x`, 0xD800+(r>>10), 0xDC00+(r&0x3FF))
		default:
			fmt.Fprintf(&out, `\u%04x`, r)
		}
		b = b[n:]
	}
	return out.Bytes()
}

func saveLedger(root string, d *ledgerDoc) error {
	b, err := encodeLedger(d)
	if err != nil {
		return err
	}
	return os.WriteFile(ledgerPath(root), b, 0o644)
}

func findEntry(d *ledgerDoc, gate string) map[string]any {
	for _, e := range d.Entries {
		if s, _ := e["gate"].(string); s == gate {
			return e
		}
	}
	return nil
}

func entryStr(e map[string]any, k string) string { s, _ := e[k].(string); return s }

func gateFuncSource(root, name string) (string, bool) {
	pat := regexp.MustCompile(`^func ` + regexp.QuoteMeta(name) + `\(`)
	var src string
	found := false
	_ = filepath.WalkDir(root, func(p string, de fs.DirEntry, err error) error {
		if err != nil || found {
			if found {
				return fs.SkipAll
			}
			return nil
		}
		if de.IsDir() || !strings.HasSuffix(p, "_test.go") || strings.Contains(filepath.ToSlash(p), "/testdata/") {
			return nil
		}
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			return nil
		}
		lines := strings.SplitAfter(strings.ToValidUTF8(string(b), "\uFFFD"), "\n")
		for i, ln := range lines {
			if !pat.MatchString(ln) {
				continue
			}
			var out strings.Builder
			out.WriteString(ln)
			for _, nxt := range lines[i+1:] {
				out.WriteString(nxt)
				if strings.HasPrefix(nxt, "}") {
					break
				}
			}
			src, found = out.String(), true
			return fs.SkipAll
		}
		return nil
	})
	return src, found
}

func gateSourceKey(root, name string) (string, bool) {
	src, ok := gateFuncSource(root, name)
	if !ok {
		return "", false
	}
	sum := sha256.Sum256([]byte(src))
	return hex.EncodeToString(sum[:])[:16], true
}

func gitShortHead(root string) string {
	out, err := exec.Command("git", "-C", root, "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// ClassifyGate returns CONFIRMED, FIRST-RUN, SOURCE-CHANGED, or UNKNOWN-GATE for gate under root.
func ClassifyGate(root, gate string) string {
	if _, err := os.Stat(ledgerPath(root)); err != nil {
		return "CONFIRMED"
	}
	d, err := loadLedger(root)
	if err != nil {
		return "CONFIRMED"
	}
	e := findEntry(d, gate)
	key, keyed := gateSourceKey(root, gate)
	switch {
	case e == nil && !keyed:
		return "UNKNOWN-GATE"
	case e == nil:
		return "FIRST-RUN"
	case keyed && key != entryStr(e, "source_sha256"):
		return "SOURCE-CHANGED"
	}
	return "CONFIRMED"
}

// ReconcileLedger prints the ledger's incomplete, first-run, stale, and source-changed checks.
func ReconcileLedger(w io.Writer, root string, gates []string) {
	d, err := loadLedger(root)
	if err != nil {
		fmt.Fprintf(w, "  gate ledger: unreadable (%v)\n", err)
		return
	}
	known := map[string]bool{}
	for _, e := range d.Entries {
		known[entryStr(e, "gate")] = true
	}
	fmt.Fprintf(w, "  gate ledger: %d confirmed entr(ies) — %s\n", len(d.Entries), ledgerRel)

	var incomplete []string
	for _, e := range d.Entries {
		for _, f := range ledgerRequired {
			if v, _ := e[f].(string); v == "" && !nonEmptyNonString(e[f]) {
				incomplete = append(incomplete, entryStr(e, "gate"))
				break
			}
		}
	}
	if len(incomplete) > 0 {
		slices.Sort(incomplete)
		fmt.Fprintf(w, "    INCOMPLETE (%d): missing a required field — a note, not a confirmation: %s\n",
			len(incomplete), strings.Join(incomplete, ", "))
	}
	checked := map[string]bool{}
	for _, g := range gates {
		checked[g] = true
	}
	if len(gates) > 0 {
		var firstrun []string
		for g := range checked {
			if !known[g] {
				firstrun = append(firstrun, g)
			}
		}
		slices.Sort(firstrun)
		if len(firstrun) > 0 {
			fmt.Fprintf(w, "    FIRST-RUN (%d): no confirmed prior result, so a failure here has no delta to compute — reported, never a blocker\n", len(firstrun))
			for _, g := range firstrun {
				fmt.Fprintf(w, "      %s\n", g)
			}
		}
	}
	var stale []string
	if len(gates) > 0 {
		for g := range known {
			if !checked[g] {
				stale = append(stale, g)
			}
		}
	}
	slices.Sort(stale)
	if len(stale) > 0 {
		fmt.Fprintf(w, "    STALE (%d): ledger entry with no matching gate — reported and IGNORED. A removed or renamed gate is ordinary; a ledger that blocks on its own leftovers trains people to delete entries\n", len(stale))
		for _, g := range stale {
			fmt.Fprintf(w, "      %s\n", g)
		}
	}
	type changed struct{ gate, commit string }
	var ch []changed
	for _, e := range d.Entries {
		if k, ok := gateSourceKey(root, entryStr(e, "gate")); ok && k != entryStr(e, "source_sha256") {
			c := entryStr(e, "commit")
			if _, has := e["commit"]; !has {
				c = "?"
			}
			ch = append(ch, changed{entryStr(e, "gate"), c})
		}
	}
	if len(ch) > 0 {
		fmt.Fprintf(w, "    CONFIRMED BEFORE THE GATE LAST CHANGED (%d) — WARNING, does not block.\n", len(ch))
		fmt.Fprintf(w, "      The assertion moved since the confirming commit, so the recorded value may be a confirmation of different semantics.\n")
		for _, c := range ch {
			fmt.Fprintf(w, "      %s  (confirmed at %s)\n", c.gate, c.commit)
		}
	}
}

func nonEmptyNonString(v any) bool {
	switch x := v.(type) {
	case nil, string:
		return false
	case bool:
		return x
	case float64:
		return x != 0
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	return true
}

func sortEntries(d *ledgerDoc) {
	slices.SortStableFunc(d.Entries, func(a, b map[string]any) int { return cmp.Compare(entryStr(a, "gate"), entryStr(b, "gate")) })
}

var seedPassRe = regexp.MustCompile(`(?m)^--- PASS: (\S+) \(`)

// RunLedger implements `gate ledger promote|classify|reconcile|seed`.
func RunLedger(argv []string, w io.Writer) int {
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "gate ledger: promote | classify | reconcile | seed")
		return 2
	}
	root, err := gatecore.RepoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate ledger: %v\n", err)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	fset := flag.NewFlagSet("gate ledger "+sub, flag.ContinueOnError)
	switch sub {
	case "classify":
		gate := fset.String("gate", "", "gate (test function) name")
		if fset.Parse(rest) != nil || *gate == "" {
			return 2
		}
		fmt.Fprintln(w, ClassifyGate(root, *gate))
		return 0
	case "reconcile":
		gates := fset.String("gates", "", "comma-separated gates the sweep checked")
		if fset.Parse(rest) != nil {
			return 2
		}
		ReconcileLedger(w, root, splitGates(*gates))
		return 0
	case "promote":
		gate := fset.String("gate", "", "gate (test function) name")
		value := fset.String("value", "", "the value a person declares correct")
		by := fset.String("by", "", "who is confirming it")
		date := fset.String("date", "", "absolute date (default today)")
		commit := fset.String("commit", "", "commit reviewed (default HEAD)")
		note := fset.String("note", "", "free text")
		force := fset.Bool("force", false, "re-promote an already-confirmed gate")
		if fset.Parse(rest) != nil || *gate == "" || *value == "" || *by == "" {
			fmt.Fprintln(os.Stderr, "gate ledger promote: --gate, --value and --by are required")
			return 2
		}
		d, err := loadLedger(root)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gate ledger: %v\n", err)
			return 1
		}
		if findEntry(d, *gate) != nil && !*force {
			fmt.Fprintf(os.Stderr, "gate_ledger: %s already confirmed — pass --force to re-promote\n", *gate)
			return 1
		}
		key, ok := gateSourceKey(root, *gate)
		if !ok {
			fmt.Fprintf(os.Stderr, "gate_ledger: cannot find `func %s(` in any _test.go — refusing to record a confirmation that cannot be keyed to source\n", *gate)
			return 1
		}
		var kept []map[string]any
		for _, e := range d.Entries {
			if entryStr(e, "gate") != *gate {
				kept = append(kept, e)
			}
		}
		d.Entries = append(kept, map[string]any{
			"gate": *gate, "value": *value, "promoted_by": *by,
			"date": orDefault(*date, time.Now().Format("2006-01-02")), "commit": orDefault(*commit, gitShortHead(root)),
			"source_sha256": key, "note": *note,
		})
		sortEntries(d)
		if err := saveLedger(root, d); err != nil {
			fmt.Fprintf(os.Stderr, "gate ledger: %v\n", err)
			return 1
		}
		fmt.Fprintf(w, "gate_ledger: confirmed %s = %s (by %s, source key %s)\n", *gate, *value, *by, key)
		return 0
	case "seed":
		logPath := fset.String("log", "", "a sweep log whose --- PASS lines to seed from")
		by := fset.String("by", "", "who is seeding")
		gates := fset.String("gates", "", "restrict to these comma-separated gates")
		date := fset.String("date", "", "absolute date (default today)")
		commit := fset.String("commit", "", "commit (default HEAD)")
		if fset.Parse(rest) != nil || *logPath == "" || *by == "" {
			fmt.Fprintln(os.Stderr, "gate ledger seed: --log and --by are required")
			return 2
		}
		logb, err := os.ReadFile(*logPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gate ledger: %v\n", err)
			return 1
		}
		set := map[string]bool{}
		for _, m := range seedPassRe.FindAllStringSubmatch(string(logb), -1) {
			set[m[1]] = true
		}
		passed := slices.Sorted(maps.Keys(set))
		if only := splitGates(*gates); len(only) > 0 {
			keep := map[string]bool{}
			for _, g := range only {
				keep[g] = true
			}
			var f []string
			for _, g := range passed {
				if keep[g] {
					f = append(f, g)
				}
			}
			passed = f
		}
		d, err := loadLedger(root)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gate ledger: %v\n", err)
			return 1
		}
		added := 0
		for _, g := range passed {
			if findEntry(d, g) != nil {
				continue
			}
			key, ok := gateSourceKey(root, g)
			if !ok {
				continue
			}
			d.Entries = append(d.Entries, map[string]any{
				"gate": g, "value": "PASS", "promoted_by": *by,
				"date": orDefault(*date, time.Now().Format("2006-01-02")), "commit": orDefault(*commit, gitShortHead(root)),
				"source_sha256": key,
				"note":          "BULK-SEEDED from a sweep log, not an individual judgement — upgrade with `go run ./cmd/gate ledger promote` when a person actually checks this gate's value",
			})
			added++
		}
		sortEntries(d)
		if err := saveLedger(root, d); err != nil {
			fmt.Fprintf(os.Stderr, "gate ledger: %v\n", err)
			return 1
		}
		fmt.Fprintf(w, "gate_ledger: seeded %d gate(s) as PASS (bulk, by %s); ledger now %d entr(ies)\n", added, *by, len(d.Entries))
		return 0
	}
	fmt.Fprintf(os.Stderr, "gate ledger: unknown subcommand %q (promote | classify | reconcile | seed)\n", sub)
	return 2
}

func splitGates(s string) []string {
	var out []string
	for g := range strings.SplitSeq(s, ",") {
		if g != "" {
			out = append(out, g)
		}
	}
	return out
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
