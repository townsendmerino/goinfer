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
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// TestSideRequests_evictionCounterfactual reads a night or day run's output (SIDE_OUT, a matrix.py output directory whose
// configuration ran with bodies and --log-token-ids on /v1/chat/completions) and, for every request that took the one GPU KV
// slot from an earlier conversation, says what that slot could have served had it been kept. It is a measurement reader, not
// a gate: it asserts nothing and skips without SIDE_OUT.
//
// A request is a SIDE request when it shares at most the first 5 prompt ids with the slot it found (a different system
// prompt: a title generator, a summarizer). For each main request that follows one, the slot it lost is the last main
// request's prompt and reply; the counterfactual is that slot's common prefix with this main prompt and the ROLLBACK, how far
// before the slot's end that prefix lies. A windowed-KV resident serves a rollback only up to its slack less one compaction
// step (448 at the planned slack; cuda/kvwindow.go), so a prefix further back than that is lost whether or not the slot was
// kept. The run's serve configuration decides the floor; pass it as SIDE_ROLLBACK (default 448).
func TestSideRequests_evictionCounterfactual(t *testing.T) {
	out := os.Getenv("SIDE_OUT")
	if out == "" {
		t.Skip("SIDE_OUT names a matrix.py output directory to read")
	}
	name := os.Getenv("SIDE_CONFIG")
	if name == "" {
		t.Fatal("SIDE_CONFIG names the configuration (its serve log is logs/serve-<name>.log)")
	}
	floor := 448
	if v := os.Getenv("SIDE_ROLLBACK"); v != "" {
		floor, _ = strconv.Atoi(v)
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

	reqRE := regexp.MustCompile(`^request: POST (\S+) .*prompt_tokens=(\d+) completion_tokens=(\d+) ttft=(\d+)ms .*reused=(\S+) slot=(\S+) div=(\S+)`)
	logRaw, err := os.ReadFile(filepath.Join(out, "logs", "serve-"+name+".log"))
	if err != nil {
		t.Fatal(err)
	}
	type line struct {
		route                    string
		prompt, completion, ttft int
		reused, slot             int
		div                      string
		prompts, slotIDs, gen    []int
		side                     bool
	}
	var ls []line
	for _, l := range strings.Split(string(logRaw), "\n") {
		if m := reqRE.FindStringSubmatch(l); m != nil {
			x := line{route: m[1], div: m[7]}
			x.prompt, _ = strconv.Atoi(m[2])
			x.completion, _ = strconv.Atoi(m[3])
			x.ttft, _ = strconv.Atoi(m[4])
			x.reused, _ = strconv.Atoi(m[5])
			x.slot, _ = strconv.Atoi(m[6])
			ls = append(ls, x)
		}
	}
	tokRaw, _ := os.ReadFile(filepath.Join(out, "logs", "token-ids-"+name+".jsonl"))
	toks := strings.Split(strings.TrimSpace(string(tokRaw)), "\n")
	if len(toks) < len(ls) {
		t.Fatalf("%d request lines but %d token-id lines", len(ls), len(toks))
	}
	for i := range ls {
		var g struct {
			Gen []int `json:"gen"`
		}
		_ = json.Unmarshal([]byte(toks[i]), &g)
		ls[i].gen = g.Gen
		f, err := os.Open(filepath.Join(out, "bodies-"+name, fmt.Sprintf("%05d.req.json.gz", i+1)))
		if err != nil {
			t.Fatal(err)
		}
		zr, _ := gzip.NewReader(f)
		raw, _ := io.ReadAll(zr)
		f.Close()
		ids, _, err := kv0RenderIDs(s, lm, ls[i].route, raw, false)
		if err != nil {
			t.Fatalf("request %d: %v", i+1, err)
		}
		ls[i].prompts = ids
		ls[i].slotIDs = append(append([]int(nil), ids...), g.Gen...)
	}
	// A side request is one that took a slot another conversation held: it shares at most the preamble's first ids.
	var mainAt int = -1
	for i := range ls {
		if mainAt >= 0 {
			if idsLCP(ls[i].prompts, ls[mainAt].slotIDs) <= 5 && ls[i].prompt < 4000 {
				ls[i].side = true
			}
		}
		if !ls[i].side {
			mainAt = i
		}
	}
	var evictions, lost int
	mainAt = -1
	pendingSide := false
	for i, x := range ls {
		if x.side {
			pendingSide = true
			t.Logf("#%d SIDE: %d prompt ids + %d generated, %.1f s to first token", i+1, x.prompt, x.completion, float64(x.ttft)/1000)
			continue
		}
		if pendingSide && mainAt >= 0 {
			held := ls[mainAt].slotIDs
			lcp := idsLCP(x.prompts, held)
			back := len(held) - lcp
			evictions++
			verdict := "lost either way: the prefix lies further back than the window serves"
			if back <= floor {
				verdict = "RECOVERABLE: a kept slot would have served it"
				lost += lcp
			}
			t.Logf("#%d MAIN after a side request: slot it lost held %d ids (request #%d), shares %d of its %d prompt ids, rollback %d (floor %d), %.1f s to first token, reused %d: %s",
				i+1, len(held), mainAt+1, lcp, x.prompt, back, floor, float64(x.ttft)/1000, x.reused, verdict)
		}
		pendingSide = false
		mainAt = i
	}
	t.Logf("main requests that followed a side request: %d; prompt ids a kept slot would have served: %d", evictions, lost)
}

// TestSideRequests_sessionStartPrefix reads a run's output directory in which every configuration is ONE session on its own
// server (the G-ids-3 shape: SIDE_OUT, SIDE_ROUTE, default /v1/responses) and answers, for a session that starts after another on a
// one-slot card: how much of its first request's prompt the previous session's slot holds (the common prefix of the two prompts),
// how far past that prefix the previous session ended (its final slot length less the prefix), and whether that fits the windowed
// rollback (SIDE_ROLLBACK, default 448: the follow-up may diverge that far before the slot's newest position). Every ordered pair of
// distinct sessions is read, whatever the order they ran in. It is a measurement reader, not a gate: it asserts nothing and skips
// without SIDE_OUT.
func TestSideRequests_sessionStartPrefix(t *testing.T) {
	out := os.Getenv("SIDE_OUT")
	if out == "" {
		t.Skip("SIDE_OUT names a matrix.py output directory to read")
	}
	route := os.Getenv("SIDE_ROUTE")
	if route == "" {
		route = "/v1/responses"
	}
	floor := 448
	if v := os.Getenv("SIDE_ROLLBACK"); v != "" {
		floor, _ = strconv.Atoi(v)
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

	dirs, _ := filepath.Glob(filepath.Join(out, "bodies-*"))
	reqRE := regexp.MustCompile(`^request: POST \S+ .*prompt_tokens=(\d+) completion_tokens=(\d+)`)
	type session struct {
		name   string
		first  []int // the first request's prompt ids
		finalL int   // the slot's length after the last request: its prompt and everything generated
		n      int   // requests
	}
	var ss []session
	for _, d := range dirs {
		name := strings.TrimPrefix(filepath.Base(d), "bodies-")
		f, err := os.Open(filepath.Join(d, "00001.req.json.gz"))
		if err != nil {
			continue
		}
		zr, _ := gzip.NewReader(f)
		raw, _ := io.ReadAll(zr)
		f.Close()
		ids, _, err := kv0RenderIDs(s, lm, route, raw, false)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		logRaw, _ := os.ReadFile(filepath.Join(out, "logs", "serve-"+name+".log"))
		x := session{name: name, first: ids}
		for _, l := range strings.Split(string(logRaw), "\n") {
			if m := reqRE.FindStringSubmatch(l); m != nil {
				p, _ := strconv.Atoi(m[1])
				c, _ := strconv.Atoi(m[2])
				x.finalL, x.n = p+c, x.n+1
			}
		}
		ss = append(ss, x)
	}
	if len(ss) < 2 {
		t.Fatalf("%d sessions under %s", len(ss), out)
	}
	var shareFrac, over []float64
	fits, pairs := 0, 0
	for i, a := range ss { // a starts after b
		for j, b := range ss {
			if i == j {
				continue
			}
			p := idsLCP(a.first, b.first)
			past := b.finalL - p
			pairs++
			shareFrac = append(shareFrac, float64(p)/float64(len(a.first)))
			over = append(over, float64(past))
			if past <= floor {
				fits++
			}
		}
	}
	sort.Float64s(shareFrac)
	sort.Float64s(over)
	q := func(v []float64, f float64) float64 { return v[int(f*float64(len(v)-1))] }
	var finals, firsts []int
	for _, x := range ss {
		finals, firsts = append(finals, x.finalL), append(firsts, len(x.first))
	}
	sort.Ints(finals)
	sort.Ints(firsts)
	t.Logf("%d sessions, %d ordered pairs; first prompts %d to %d ids (median %d); final slot lengths %d to %d (median %d)",
		len(ss), pairs, firsts[0], firsts[len(firsts)-1], firsts[len(firsts)/2], finals[0], finals[len(finals)-1], finals[len(finals)/2])
	t.Logf("share of a session's first prompt that the previous session's prompt holds: min %.1f%%, median %.1f%%, max %.1f%%",
		100*q(shareFrac, 0), 100*q(shareFrac, .5), 100*q(shareFrac, 1))
	t.Logf("how far past the shared prefix the previous session ended: min %.0f, median %.0f, max %.0f ids (floor %d)", q(over, 0), q(over, .5), q(over, 1), floor)
	t.Logf("pairs where a one-slot card could serve the prefix (ended within %d of it): %d of %d", floor, fits, pairs)
	// A prefix snapshot is taken at the end of the first prefill, b = n/64*64 rows, and serves a later session within the rollback of that
	// end: it does not qualify when the later session shares less than b-rollback ids with it.
	disq, worst := 0, 0
	for i, a := range ss {
		b := len(a.first) / kvmgr.BlockRows * kvmgr.BlockRows
		for j, o := range ss {
			if i == j {
				continue
			}
			past := b - idsLCP(a.first, o.first)
			worst = max(worst, past)
			if past > floor {
				disq++
			}
		}
	}
	t.Logf("snapshot qualification: of %d pairs (a snapshot taken at the end of one session's first prefill, a later session sharing a prefix), %d do not qualify (the snapshot ends more than %d past the shared prefix); the largest is %d ids past", pairs, disq, floor, worst)
	for _, x := range ss {
		t.Logf("  %-28s first prompt %5d ids, %2d requests, final slot %5d ids", x.name, len(x.first), x.n, x.finalL)
	}
}
