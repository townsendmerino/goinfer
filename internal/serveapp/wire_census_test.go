package serveapp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

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
	Upload      *wireUpload     `json:"upload,omitempty"`
	Until       *wireUntil      `json:"until,omitempty"`
	// Steps, when set, are the case's requests in order (the case's own request fields are then unused), each recorded.
	// A step's path and body may name an earlier step's response field as {{N.field}} (N from 1, a top-level field).
	Steps []wireCase `json:"steps,omitempty"`
	// Racy names top-level response fields whose value races the request's own work (a batch's counts while its lines
	// run): each is recorded as "<racing>".
	Racy []string `json:"racy,omitempty"`
	// Flags are extra serve flags for the case's server.
	Flags []string `json:"flags,omitempty"`
}

// wireUpload is a multipart file upload, as /v1/files takes one: the form's purpose and one file.
type wireUpload struct {
	Purpose  string `json:"purpose"`
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// wireUntil repeats a step until its response's top-level Field is one of In (a job or batch reaching a terminal
// state); only the last response is recorded.
type wireUntil struct {
	Field string   `json:"field"`
	In    []string `json:"in"`
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
	{"times", regexp.MustCompile(`"(created|started|finished|[a-z_]+_at)":\s*("[0-9]{4}-[0-9]{2}-[0-9]{2}T[^"]*"|[1-9][0-9]{8,})`), func(m string) string {
		return m[:strings.Index(m, ":")+1] + `"<time>"`
	}},
	{"logprob", regexp.MustCompile(`"logprob":\s*-?[0-9.]+(e-?[0-9]+)?`), func(string) string { return `"logprob":0` }},
}

// wireIDRE matches a minted id; each distinct one becomes its prefix and its order of first appearance.
var wireIDRE = regexp.MustCompile(`\b((?:chatcmpl|cmpl|msg|resp|call|toolu|fc|rs|item|req|job|batch_req|batch|file)[-_])[A-Za-z0-9]{8,}\b`)

// replaceJSONValue replaces the value of the first "key": in body (an object, an array, a string or a scalar) with
// placeholder, matching brackets and quotes so nested values go whole. ok is false when body has no such key.
func replaceJSONValue(body, key, placeholder string) (string, bool) {
	k := `"` + key + `":`
	i := strings.Index(body, k)
	if i < 0 {
		return body, false
	}
	start, j, depth, inStr := i+len(k), i+len(k), 0, false
	for ; j < len(body); j++ {
		c := body[j]
		if inStr {
			if c == '\\' {
				j++
			} else if c == '"' {
				inStr = false
				if depth == 0 {
					j++
					break
				}
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '[', '{':
			depth++
		case ']', '}':
			if depth == 0 {
				return body[:start] + placeholder + body[j:], true
			}
			depth--
			if depth == 0 {
				j++
				return body[:start] + placeholder + body[j:], true
			}
		case ',':
			if depth == 0 {
				return body[:start] + placeholder + body[j:], true
			}
		}
	}
	return body[:start] + placeholder + body[j:], true
}

// normalizeSelfTest replaces /health's "selftest" array with a placeholder. It lists the self-tests recorded in this
// process, a decoder-wide registry other tests in the package add to (TestServe_healthReportsSelfTests records a fake
// GPU backend), so its contents depend on which tests ran first; that test checks the field itself.
func normalizeSelfTest(body string) (string, bool) {
	if !strings.Contains(body, `"selftest":[`) {
		return body, false
	}
	return replaceJSONValue(body, "selftest", `"<process-wide>"`)
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
func wireServer(t *testing.T, flags ...string) (*http.Client, *loadedModel, io.Closer) {
	t.Helper()
	h, closer, err := NewHandler(append([]string{"-model", "tiny=" + tinyChatCheckpoint(t), "-backend", "cpu", "-quant", "int8int8"}, flags...))
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

// wireRefRE is a reference to an earlier step's response field.
var wireRefRE = regexp.MustCompile(`\{\{([0-9]+)\.([a-z_]+)\}\}`)

// substitute replaces each {{N.field}} in s with step N's response field (a string or a number).
func substitute(t *testing.T, s string, prev [][]byte) string {
	return wireRefRE.ReplaceAllStringFunc(s, func(m string) string {
		g := wireRefRE.FindStringSubmatch(m)
		var n int
		fmt.Sscan(g[1], &n)
		if n < 1 || n > len(prev) {
			t.Fatalf("census: %s names step %d of %d", m, n, len(prev))
		}
		var obj map[string]any
		if err := json.Unmarshal(prev[n-1], &obj); err != nil {
			t.Fatalf("census: %s: step %d's body is not a JSON object: %v", m, n, err)
		}
		switch v := obj[g[2]].(type) {
		case string:
			return v
		case float64:
			return fmt.Sprint(v)
		}
		t.Fatalf("census: %s: no such field in step %d's body", m, n)
		return ""
	})
}

// runWireCase sends c (each of its steps, in order) and returns its record.
func runWireCase(t *testing.T, cl *http.Client, sc *loadedModel, c wireCase) string {
	t.Helper()
	steps := c.Steps
	if len(steps) == 0 {
		steps = []wireCase{c}
	}
	var prev [][]byte
	var raws []string
	for _, st := range steps {
		rec, raw := runWireStep(t, cl, sc, c.Name, st, prev)
		prev = append(prev, raw)
		raws = append(raws, rec)
	}
	norm, fired := normalizeWire(strings.Join(raws, "\n"))
	var out strings.Builder
	fmt.Fprintf(&out, "# %s\n", c.Name)
	if c.About != "" {
		fmt.Fprintf(&out, "# %s\n", c.About)
	}
	fmt.Fprintf(&out, "# normalized: %s\n", strings.Join(append([]string{"none"}, fired...)[min(1, len(fired)):], ", "))
	out.WriteString(norm)
	return out.String()
}

// runWireStep sends one request and returns its record (the request, the status, the headers a client reads, the
// body, not yet normalized) and its raw body.
func runWireStep(t *testing.T, cl *http.Client, sc *loadedModel, name string, c wireCase, prev [][]byte) (string, []byte) {
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
	body, ctype := c.RawBody, ""
	if len(c.Body) > 0 {
		var b bytes.Buffer
		if err := json.Compact(&b, c.Body); err != nil {
			t.Fatal(err)
		}
		body = b.String()
	}
	if body != "" {
		ctype = "application/json"
	}
	if u := c.Upload; u != nil {
		var b bytes.Buffer
		mw := multipart.NewWriter(&b)
		if err := mw.SetBoundary("wirecensusboundary"); err != nil {
			t.Fatal(err)
		}
		_ = mw.WriteField("purpose", u.Purpose)
		fw, _ := mw.CreateFormFile("file", u.Filename)
		_, _ = fw.Write([]byte(substitute(t, u.Content, prev)))
		_ = mw.Close()
		body, ctype = b.String(), mw.FormDataContentType()
	}
	body, path := substitute(t, body, prev), substitute(t, c.Path, prev)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var resp *http.Response
	for attempt := 0; ; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, c.Method, "http://goinfer"+path, strings.NewReader(body))
		if ctype != "" {
			req.Header.Set("Content-Type", ctype)
		}
		var err error
		if resp, err = cl.Do(req); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if c.Until == nil {
			break
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var obj map[string]any
		_ = json.Unmarshal(raw, &obj)
		if v, _ := obj[c.Until.Field].(string); slices.Contains(c.Until.In, v) {
			resp.Body = io.NopCloser(bytes.NewReader(raw))
			break
		}
		if attempt == 2000 {
			t.Fatalf("%s: %s never reached %s in %v: %s", name, path, c.Until.Field, c.Until.In, raw)
		}
		time.Sleep(5 * time.Millisecond)
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
	} else {
		var err error
		if raw, err = io.ReadAll(resp.Body); err != nil {
			t.Fatalf("%s: reading the body: %v", name, err)
		}
	}
	checkForced(t, sc, name, body, raw)
	for _, k := range c.Racy {
		r, ok := replaceJSONValue(string(raw), k, `"<racing>"`)
		if !ok {
			t.Fatalf("%s: racy field %q is not in the response", name, k)
		}
		raw = []byte(r)
	}
	var out strings.Builder
	fmt.Fprintf(&out, "%s %s\n%s\n\nHTTP %d\n", c.Method, path, body, resp.StatusCode)
	for _, h := range wireHeaders {
		if v := resp.Header.Get(h); v != "" {
			fmt.Fprintf(&out, "%s: %s\n", h, v)
		}
	}
	out.WriteString("\n")
	out.Write(raw)
	if !bytes.HasSuffix(raw, []byte("\n")) {
		out.WriteString("\n")
	}
	return out.String(), raw
}

// checkForced holds the census to no logit: in a case on the tiny model whose request forces tokens with logit_bias,
// every generated string the response carries (a "content" or "text" value, whole or streamed, or a logprob's "token")
// is made of those tokens
// only, so the record is the same on every machine.
func checkForced(t *testing.T, sc *loadedModel, name, body string, raw []byte) {
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
			if key != "content" && key != "text" && key != "token" {
				return
			}
			for _, r := range x {
				if !forced[string(r)] {
					t.Errorf("%s: generated %q holds %q, which logit_bias does not force: the record would depend on a logit", name, x, r)
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
			cl, sc, closer := wireServer(t, c.Flags...)
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
