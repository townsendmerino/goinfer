package serveapp

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestSideRequests_snapshotPrefixes checks a run's restored prefixes against the ids the run itself wrote (KV7 step 2, the night rerun with bodies kept,
// docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8). SIDE_OUT is a matrix.py output directory, SIDE_CONFIG the configuration (its serve log is
// logs/serve-<name>.log and its request bodies bodies-<name>/NNNNN.req.json.gz, in the order of the log's request lines). For every request whose line
// carries snap=<rows>, the first <rows> ids of its rendered prompt must be the first <rows> ids of an EARLIER request's rendered prompt: a restore puts
// back rows a prefill wrote for those ids, and nothing else is a legitimate source. It also reports, per request, the rows reused, the time to first token
// and whether it restored. A diagnostic that asserts that one thing; it skips without SIDE_OUT.
func TestSideRequests_snapshotPrefixes(t *testing.T) {
	out, name := os.Getenv("SIDE_OUT"), os.Getenv("SIDE_CONFIG")
	if out == "" {
		t.Skip("SIDE_OUT names a matrix.py output directory to read")
	}
	if name == "" {
		t.Fatal("SIDE_CONFIG names the configuration")
	}
	tk, err := modelload.Tokenizer(filepath.Join("..", "..", "testdata", "mellum2-tokenizer"))
	if err != nil {
		t.Skipf("needs the Mellum2 tokenizer: %v", err)
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := registerFlags(fs)
	_ = fs.Parse(nil)
	s := &server{cfg: sf.cfg}
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	re := regexp.MustCompile(`^request: POST (\S+) .*prompt_tokens=(\d+) .*ttft=(\d+)ms .*reused=(\S+) .*?(?: snap=(\d+))?$`)
	logRaw, err := os.ReadFile(filepath.Join(out, "logs", "serve-"+name+".log"))
	if err != nil {
		t.Fatal(err)
	}
	var prompts [][]int
	var bad int
	i := 0
	for _, l := range strings.Split(string(logRaw), "\n") {
		m := re.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		i++
		f, err := os.Open(filepath.Join(out, "bodies-"+name, fmt.Sprintf("%05d.req.json.gz", i)))
		if err != nil {
			t.Fatal(err)
		}
		zr, _ := gzip.NewReader(f)
		raw, _ := io.ReadAll(zr)
		f.Close()
		route := strings.SplitN(m[1], "?", 2)[0]
		ids, _, err := kv0RenderIDs(s, lm, route, raw, false)
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		ttft, _ := strconv.Atoi(m[3])
		snap, _ := strconv.Atoi(m[5])
		best := 0
		for _, p := range prompts {
			best = max(best, idsLCP(ids, p))
		}
		verdict := "cold or continued"
		if snap > 0 {
			verdict = "restored"
			if best < snap {
				verdict = "RESTORED ROWS NO EARLIER PROMPT WROTE"
				bad++
			}
		}
		t.Logf("#%d %s %d ids, %.1f s to first token, reused %s, snap %d, longest shared prefix with an earlier prompt %d: %s", i, route, len(ids), float64(ttft)/1000, m[4], snap, best, verdict)
		prompts = append(prompts, ids)
	}
	if i == 0 {
		t.Fatal("no request lines")
	}
	if bad > 0 {
		t.Errorf("%d restores put back rows that no earlier prompt wrote", bad)
	}
}

// TestSideRequests_sessionStartShare answers a report-only question about Claude Code session starts (KV7 step 2, tier 2 record): across real sessions, how
// often is the first user message's leading reminder (CLAUDE.md and the memory index in it) byte-identical, how far do two session starts agree, and what
// would a boundary after that reminder save over the boundary at the first turn? SIDE_BODIES lists Anthropic Messages bodies (files, or directories whose
// *.req.json.gz are all taken); only session starts are used (tools, no assistant turn). Nothing is asserted and the boundary is not changed.
func TestSideRequests_sessionStartShare(t *testing.T) {
	list := os.Getenv("SIDE_BODIES")
	if list == "" {
		t.Skip("SIDE_BODIES lists the bodies to read")
	}
	var paths []string
	for _, p := range strings.Split(list, ",") {
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			m, _ := filepath.Glob(filepath.Join(p, "*.req.json.gz"))
			paths = append(paths, m...)
		} else {
			paths = append(paths, p)
		}
	}
	tk, err := modelload.Tokenizer(filepath.Join("..", "..", "testdata", "mellum2-tokenizer"))
	if err != nil {
		t.Skipf("needs the Mellum2 tokenizer: %v", err)
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := registerFlags(fs)
	_ = fs.Parse(nil)
	s := &server{cfg: sf.cfg}
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	type sess struct {
		name  string
		ids   []int
		bound int // the first turn's content begins here (the first user message with its first character changed marks it)
		lead  string
	}
	var ss []sess
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		zr, err := gzip.NewReader(f)
		if err != nil {
			f.Close()
			continue
		}
		raw, _ := io.ReadAll(zr)
		f.Close()
		var b struct {
			Tools    []json.RawMessage `json:"tools"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(raw, &b) != nil || len(b.Tools) < 15 || len(b.Messages) == 0 || b.Messages[0].Role != "user" {
			continue
		}
		assistant := false
		for _, m := range b.Messages {
			assistant = assistant || m.Role == "assistant"
		}
		var blocks []struct {
			Text string `json:"text"`
		}
		if assistant || json.Unmarshal(b.Messages[0].Content, &blocks) != nil || len(blocks) == 0 {
			continue
		}
		ids, _, err := kv0RenderIDs(s, lm, "/v1/messages", raw, false)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		// the same body with the first character of the first user block changed: the ids agree with ids exactly up to where that turn's content begins
		mod := strings.Replace(string(raw), blocks[0].Text[:12], "~"+blocks[0].Text[1:12], 1)
		ids2, _, err := kv0RenderIDs(s, lm, "/v1/messages", []byte(mod), false)
		if err != nil {
			t.Fatalf("%s (changed): %v", p, err)
		}
		ss = append(ss, sess{name: filepath.Base(filepath.Dir(filepath.Dir(p))) + "/" + filepath.Base(p), ids: ids, bound: idsLCP(ids, ids2), lead: blocks[0].Text})
	}
	if len(ss) < 2 {
		t.Fatalf("%d session starts read; need at least two", len(ss))
	}
	const idsPerSec, pass = 93.7, 512
	// Two starts share the stable section only when they ran in the same working directory (Claude Code writes it into the system prompt, early); the
	// others agree for as long as that prefix does and no boundary helps them. They are counted apart.
	var same, shared, apart int
	var extra []int
	for i := range ss {
		t.Logf("%s: %d ids, the first turn's content begins at %d (pass boundary %d)", ss[i].name, len(ss[i].ids), ss[i].bound, ss[i].bound/pass*pass)
		for j := i + 1; j < len(ss); j++ {
			l := idsLCP(ss[i].ids, ss[j].ids)
			q := min(ss[i].bound, ss[j].bound) / pass * pass
			if l < q {
				apart++
				t.Logf("  %s vs %s: agree for %d ids only, short of the stable section (%d): another working directory, or another tools list", ss[i].name, ss[j].name, l, q)
				continue
			}
			shared++
			ident := ss[i].lead == ss[j].lead
			if ident {
				same++
			}
			extra = append(extra, l-q)
			t.Logf("  %s vs %s: leading reminder byte-identical %v; agree for %d ids, %d past the pass boundary %d (%.1f s of prefill at %.1f ids/s)",
				ss[i].name, ss[j].name, ident, l, l-q, q, float64(l-q)/idsPerSec, idsPerSec)
		}
	}
	slices.Sort(extra)
	if len(extra) == 0 {
		t.Logf("SUMMARY: %d session starts; no pair shares the stable section (%d pairs apart)", len(ss), apart)
		return
	}
	t.Logf("SUMMARY: %d session starts. %d pairs share the stable section and %d do not (a different working directory or tools). Of the sharing pairs the leading reminder is byte-identical in %d of %d; "+
		"the ids two such starts still agree on past the pass boundary: min %d, median %d, max %d (median %.1f s of prefill a boundary after the reminder would save)",
		len(ss), shared, apart, same, shared, extra[0], extra[len(extra)/2], extra[len(extra)-1], float64(extra[len(extra)/2])/idsPerSec)
}
