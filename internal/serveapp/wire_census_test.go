package serveapp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/inproc"
)

// The wire census (docs/tasks/task-subsystems-2026-10.md § "SB7 — the wire census"): what a client receives from each
// scripted request, byte for byte after normalizing ids and times, compared with goldens under testdata/wire/golden/, so
// a move of the serving shell onto an engine interface can show it changed nothing a client sees. -wire.update rewrites
// the goldens.
var updateWire = flag.Bool("wire.update", false, "rewrite the wire census goldens under testdata/wire/golden/")

// wireCase is one request. Model "tiny" is the tiny chat fixture, whose cases force every token with logit_bias;
// model "scripted" streams Script's token ids through the server's scripted-generation seam. Each case gets a fresh
// server, so no record depends on the cases before it.
type wireCase struct {
	Name        string          `json:"name"`
	About       string          `json:"about,omitempty"`
	Method      string          `json:"method"`
	Path        string          `json:"path"`
	Body        json.RawMessage `json:"body,omitempty"`
	RawBody     string          `json:"raw_body,omitempty"`
	Script      *string         `json:"script,omitempty"`
	CancelAfter int             `json:"cancel_after_events,omitempty"`
	Warm        bool            `json:"warm,omitempty"` // send the request once first, unrecorded: the record then shows prefix reuse
}

// wireHeaders are the response headers a client reads; every other header is left out of the record.
var wireHeaders = []string{"Content-Type", "Cache-Control", "Retry-After", "X-Accel-Buffering"}

// wireNorms are the normalizations, in order. Each golden lists the ones that fired.
var wireNorms = []struct {
	name string
	re   *regexp.Regexp
	repl func(string) string
}{
	{"created", regexp.MustCompile(`"(created|created_at)":\s*[0-9]+`), func(m string) string {
		return m[:strings.Index(m, ":")+1] + "0"
	}},
	{"logprob", regexp.MustCompile(`"logprob":\s*-?[0-9.]+(e-?[0-9]+)?`), func(string) string { return `"logprob":0` }},
}

// wireIDRE matches a minted id; each distinct one becomes its prefix and its order of first appearance.
var wireIDRE = regexp.MustCompile(`\b((?:chatcmpl|cmpl|msg|resp|call|toolu|fc|rs|item|req)[-_])[A-Za-z0-9]{8,}\b`)

// normalizeSelfTest replaces /health's "selftest" array with a placeholder. It lists the self-tests recorded in this
// process, a decoder-wide registry other tests in the package add to (TestServe_healthReportsSelfTests records a fake
// GPU backend), so its contents depend on which tests ran first; that test checks the field itself.
func normalizeSelfTest(body string) (string, bool) {
	const key = `"selftest":[`
	i := strings.Index(body, key)
	if i < 0 {
		return body, false
	}
	depth, j := 0, i+len(key)-1
	for ; j < len(body); j++ {
		switch body[j] {
		case '[':
			depth++
		case ']':
			depth--
		}
		if depth == 0 {
			break
		}
	}
	return body[:i] + `"selftest":"<process-wide>"` + body[j+1:], true
}

func normalizeWire(body string) (string, []string) {
	var fired []string
	if b, ok := normalizeSelfTest(body); ok {
		body, fired = b, append(fired, "selftest")
	}
	for _, n := range wireNorms {
		if n.re.MatchString(body) {
			fired = append(fired, n.name)
			body = n.re.ReplaceAllStringFunc(body, n.repl)
		}
	}
	seen := map[string]string{}
	body = wireIDRE.ReplaceAllStringFunc(body, func(id string) string {
		if v, ok := seen[id]; ok {
			return v
		}
		p := wireIDRE.FindStringSubmatch(id)[1]
		v := fmt.Sprintf("%s<%d>", p, len(seen)+1)
		seen[id] = v
		return v
	})
	if len(seen) > 0 {
		fired = append(fired, "ids")
	}
	return body, fired
}

// wireServer is the in-process handler on the tiny chat fixture, plus a model "scripted" on the same tokenizer and
// template whose generations are set per case.
func wireServer(t *testing.T) (*http.Client, *loadedModel, io.Closer) {
	t.Helper()
	h, closer, err := NewHandler([]string{"-model", "tiny=" + tinyChatCheckpoint(t), "-backend", "cpu", "-quant", "int8int8"})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	srv := closer.(*handlerCloser).srv
	srv.regMu.Lock()
	tiny := srv.models["tiny"]
	sc := &loadedModel{tk: tiny.tk, tmpl: tiny.tmpl, stopIDs: tiny.stopIDs, eosIDs: tiny.eosIDs, vocab: tiny.vocab,
		sampling: tiny.sampling, name: "scripted", scripted: []int{}}
	srv.models["scripted"] = sc
	srv.regMu.Unlock()
	return &http.Client{Transport: inproc.Transport(h)}, sc, closer
}

// runWireCase sends c and returns its record: the request, the status, the headers a client reads, and the body.
func runWireCase(t *testing.T, cl *http.Client, sc *loadedModel, c wireCase) string {
	t.Helper()
	if c.Script != nil {
		ids, err := sc.tk.Encode(*c.Script, false)
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := sc.tk.Decode(ids); got != *c.Script {
			t.Fatalf("the tiny tokenizer does not round-trip the script: %q", got)
		}
		sc.scripted = append([]int{}, ids...)
	}
	body := c.RawBody
	if len(c.Body) > 0 {
		var b bytes.Buffer
		if err := json.Compact(&b, c.Body); err != nil {
			t.Fatal(err)
		}
		body = b.String()
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, c.Method, "http://goinfer"+c.Path, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := cl.Do(req)
	if err != nil {
		t.Fatalf("%s: %v", c.Name, err)
	}
	defer resp.Body.Close()
	var raw []byte
	if c.CancelAfter > 0 {
		// read whole lines until that many data events have arrived, then hang up
		br := bufio.NewReader(resp.Body)
		for n := 0; n < c.CancelAfter; {
			line, err := br.ReadString('\n')
			raw = append(raw, line...)
			if strings.HasPrefix(line, "data: ") {
				n++
			}
			if err != nil {
				break
			}
		}
		if line, err := br.ReadString('\n'); err == nil && line == "\n" {
			raw = append(raw, line...) // the event's blank separator
		}
		cancel()
	} else if raw, err = io.ReadAll(resp.Body); err != nil {
		t.Fatalf("%s: reading the body: %v", c.Name, err)
	}
	checkForced(t, sc, c, body, raw)
	norm, fired := normalizeWire(string(raw))
	var out strings.Builder
	fmt.Fprintf(&out, "# %s\n", c.Name)
	if c.About != "" {
		fmt.Fprintf(&out, "# %s\n", c.About)
	}
	fmt.Fprintf(&out, "# normalized: %s\n", strings.Join(append([]string{"none"}, fired...)[min(1, len(fired)):], ", "))
	fmt.Fprintf(&out, "%s %s\n%s\n\nHTTP %d\n", c.Method, c.Path, body, resp.StatusCode)
	for _, h := range wireHeaders {
		if v := resp.Header.Get(h); v != "" {
			fmt.Fprintf(&out, "%s: %s\n", h, v)
		}
	}
	out.WriteString("\n")
	out.WriteString(norm)
	if !strings.HasSuffix(norm, "\n") {
		out.WriteString("\n")
	}
	return out.String()
}

// checkForced holds the census to no logit: in a case on the tiny model whose request forces tokens with logit_bias,
// every generated string the response carries (a "content" or "text" value, whole or streamed) is made of those tokens
// only, so the record is the same on every machine.
func checkForced(t *testing.T, sc *loadedModel, c wireCase, body string, raw []byte) {
	t.Helper()
	var req struct {
		Model     string             `json:"model"`
		LogitBias map[string]float64 `json:"logit_bias"`
	}
	if json.Unmarshal([]byte(body), &req) != nil || req.Model != "tiny" || len(req.LogitBias) == 0 {
		return
	}
	forced := map[string]bool{}
	for k := range req.LogitBias {
		var id int
		fmt.Sscan(k, &id)
		s, _ := sc.tk.Decode([]int{id})
		forced[s] = true
	}
	var walk func(v any, key string)
	walk = func(v any, key string) {
		switch x := v.(type) {
		case map[string]any:
			for k, e := range x {
				walk(e, k)
			}
		case []any:
			for _, e := range x {
				walk(e, key)
			}
		case string:
			if key != "content" && key != "text" {
				return
			}
			for _, r := range x {
				if !forced[string(r)] {
					t.Errorf("%s: generated %q holds %q, which logit_bias does not force: the record would depend on a logit", c.Name, x, r)
					return
				}
			}
		}
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimPrefix(line, "data: ")
		var v any
		if json.Unmarshal([]byte(line), &v) == nil {
			walk(v, "")
		}
	}
}

// TestWireCensus replays testdata/wire/cases.json through the in-process server and compares each record with
// testdata/wire/golden/<case>.txt.
func TestWireCensus(t *testing.T) {
	var cases []wireCase
	raw, err := os.ReadFile(filepath.Join("testdata", "wire", "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			cl, sc, closer := wireServer(t)
			defer closer.Close()
			if c.Warm {
				runWireCase(t, cl, sc, c)
			}
			got := runWireCase(t, cl, sc, c)
			p := filepath.Join("testdata", "wire", "golden", c.Name+".txt")
			if *updateWire {
				if err := os.WriteFile(p, []byte(got), 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(p)
			if err != nil {
				t.Fatalf("no golden (record with -wire.update): %v", err)
			}
			if string(want) != got {
				t.Errorf("%s differs from today's wire record:\n%s", p, firstDiff(string(want), got))
			}
		})
	}
}
