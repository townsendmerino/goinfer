package chatapp

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/batchio"
)

func inputFile(t *testing.T, ids ...string) string {
	t.Helper()
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(`{"custom_id":"` + id + `","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"user","content":"hi ` + id + `"}]}}` + "\n")
	}
	p := filepath.Join(t.TempDir(), "in.jsonl")
	if err := os.WriteFile(p, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestPlanBatch(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out.jsonl")
	in := inputFile(t, "a", "b", "c", "d")

	p, err := planBatch(in, out)
	if err != nil || len(p.lines) != 4 || len(p.todo) != 4 || p.prog.GoodBytes != 0 {
		t.Fatalf("fresh plan: %+v, %v", p, err)
	}

	// An output that already holds a and c (b failed earlier: an error line is not done).
	os.WriteFile(out, bytes.Join([][]byte{
		batchio.OKLine("i1", "a", 200, map[string]any{}),
		batchio.ErrLine("i2", "b", "api_error", "x"),
		batchio.OKLine("i3", "c", 200, map[string]any{}),
		batchio.OKLine("i4", "zzz", 200, map[string]any{}), // from some other input
	}, nil), 0o644)
	p, err = planBatch(in, out)
	if err != nil {
		t.Fatal(err)
	}
	var todo []string
	for _, i := range p.todo {
		todo = append(todo, p.lines[i].CustomID)
	}
	if strings.Join(todo, ",") != "b,d" {
		t.Errorf("todo = %v, want b,d (a and c are done; the errored b and the untouched d are not)", todo)
	}
	if len(p.orphans) != 1 || p.orphans[0] != "zzz" {
		t.Errorf("orphans = %v, want [zzz]", p.orphans)
	}

	for name, tc := range map[string]struct {
		in, out, want string
	}{
		"no output":  {in, "", "go together"},
		"no input":   {"", out, "go together"},
		"same file":  {in, in, "overwrite it"},
		"dup ids":    {inputFile(t, "a", "b", "a"), out, "unique"},
		"missing":    {filepath.Join(dir, "nope.jsonl"), out, syscall.ENOENT.Error()}, // the OS's own not-found text (Windows words it differently)
		"bad json":   {writeTemp(t, "{oops\n"), out, "line 1: invalid JSON"},
		"empty file": {writeTemp(t, "\n"), out, "no request lines"},
	} {
		if _, err := planBatch(tc.in, tc.out); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want it to contain %q", name, err, tc.want)
		}
	}

	// A corrupt output is refused rather than appended after.
	bad := writeTemp(t, "not json\n")
	if _, err := planBatch(in, bad); err == nil || !strings.Contains(err.Error(), "refusing to resume") {
		t.Errorf("corrupt output: %v", err)
	}
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "f.jsonl")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// fakeRun answers every line with its own custom_id as the content, failing the ids in fail, and records the order it ran them.
type fakeRun struct {
	fail map[string]bool
	ran  []string
	// after, when set, runs once each line has been answered: a test uses it to interrupt.
	after func(id string)
}

func (f *fakeRun) run(ctx context.Context, l batchio.Line) lineResult {
	f.ran = append(f.ran, l.CustomID)
	var res lineResult
	if f.fail[l.CustomID] {
		res = lineErr(500, "api_error", "boom "+l.CustomID)
	} else {
		res = lineResult{status: 200, nTok: 5, body: map[string]any{"content": "reply to " + l.CustomID}}
	}
	if f.after != nil {
		f.after(l.CustomID)
	}
	return res
}

// outputs reads an output file back as custom_id → body content, in file order.
func outputs(t *testing.T, path string) (ids []string, content map[string]string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if errors := os.IsNotExist(err); errors {
		return nil, nil
	}
	if err != nil {
		t.Fatal(err)
	}
	content = map[string]string{}
	for raw := range bytes.SplitSeq(bytes.TrimSpace(data), []byte("\n")) {
		if len(raw) == 0 {
			continue
		}
		var rec struct {
			CustomID string `json:"custom_id"`
			Response struct {
				Body map[string]string `json:"body"`
			} `json:"response"`
		}
		if err := json.Unmarshal(raw, &rec); err != nil {
			t.Fatalf("%s: %v in %q", path, err, raw)
		}
		ids = append(ids, rec.CustomID)
		content[rec.CustomID] = rec.Response.Body["content"]
	}
	return ids, content
}

func TestRunBatchLines_splitsSuccessAndFailure(t *testing.T) {
	in := inputFile(t, "a", "b", "c")
	out := filepath.Join(t.TempDir(), "out.jsonl")
	plan, err := planBatch(in, out)
	if err != nil {
		t.Fatal(err)
	}
	fr := &fakeRun{fail: map[string]bool{"b": true}}
	var log bytes.Buffer
	sum, err := runBatchLines(context.Background(), plan, fr.run, &log)
	if err != nil {
		t.Fatal(err)
	}
	if sum.OK != 2 || sum.Failed != 1 || sum.Interrupted || sum.Remaining != 0 {
		t.Errorf("summary = %+v", sum) // nothing was left unreached; the failure is counted on its own and a rerun retries it
	}
	ids, content := outputs(t, out)
	if strings.Join(ids, ",") != "a,c" || content["a"] != "reply to a" {
		t.Errorf("output = %v %v, want a,c in input order", ids, content)
	}
	errData, err := os.ReadFile(batchio.ErrorPath(out))
	if err != nil || !strings.Contains(string(errData), `"custom_id":"b"`) || !strings.Contains(string(errData), "boom b") || strings.Count(string(errData), "\n") != 1 {
		t.Errorf("error file = %q, %v", errData, err)
	}
	if !strings.Contains(log.String(), "[3/3] c ok") || !strings.Contains(log.String(), "FAILED api_error") {
		t.Errorf("progress log:\n%s", log.String())
	}
	if got := sum.report(plan); !strings.Contains(got, "1 failed") || !strings.Contains(got, "a rerun retries them") {
		t.Errorf("report: %q", got)
	}

	// The rerun: a and c are done, only b runs — and this time it succeeds. The stale error file goes away with it.
	plan, err = planBatch(in, out)
	if err != nil {
		t.Fatal(err)
	}
	fr = &fakeRun{}
	if sum, err = runBatchLines(context.Background(), plan, fr.run, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if strings.Join(fr.ran, ",") != "b" || sum.OK != 1 || sum.Already != 2 {
		t.Errorf("rerun ran %v, summary %+v; want only b", fr.ran, sum)
	}
	if _, err := os.Stat(batchio.ErrorPath(out)); !os.IsNotExist(err) {
		t.Errorf("a clean rerun must not leave last run's error file behind: %v", err)
	}
	ids, _ = outputs(t, out)
	if strings.Join(ids, ",") != "a,c,b" { // appended: the output is in completion order across runs
		t.Errorf("output after the rerun = %v", ids)
	}
}

// The property the feature exists for: a run killed part-way and resumed produces the same set of results as one that was not.
func TestRunBatchLines_interruptAndResume(t *testing.T) {
	const n = 7
	var names []string
	for i := range n {
		names = append(names, string(rune('a'+i)))
	}
	in := inputFile(t, names...)

	// Reference: uninterrupted.
	refOut := filepath.Join(t.TempDir(), "ref.jsonl")
	plan, _ := planBatch(in, refOut)
	if _, err := runBatchLines(context.Background(), plan, (&fakeRun{}).run, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	_, want := outputs(t, refOut)

	// Interrupted while line 4 ("d") is in flight: d's result arrives with the context cancelled and must be dropped.
	out := filepath.Join(t.TempDir(), "out.jsonl")
	plan, _ = planBatch(in, out)
	ctx, cancel := context.WithCancel(context.Background())
	fr := &fakeRun{after: func(id string) {
		if id == "d" {
			cancel()
		}
	}}
	sum, err := runBatchLines(ctx, plan, fr.run, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if !sum.Interrupted || sum.OK != 3 || sum.Remaining != n-3 {
		t.Errorf("interrupted summary = %+v, want 3 ok and %d remaining", sum, n-3)
	}
	ids, _ := outputs(t, out)
	if strings.Join(ids, ",") != "a,b,c" {
		t.Errorf("interrupted output = %v: the in-flight line must not be recorded", ids)
	}

	// Kill -9 instead: half of the next line is on disk.
	f, _ := os.OpenFile(out, os.O_WRONLY|os.O_APPEND, 0o644)
	half := batchio.OKLine("x", "d", 200, map[string]any{"content": "reply to d"})
	f.Write(half[:len(half)/2])
	f.Close()

	// Resume.
	plan, err = planBatch(in, out)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.prog.Torn || len(plan.todo) != n-3 {
		t.Fatalf("resume plan: torn=%v todo=%d", plan.prog.Torn, len(plan.todo))
	}
	fr = &fakeRun{}
	var log bytes.Buffer
	if _, err = runBatchLines(context.Background(), plan, fr.run, &log); err != nil {
		t.Fatal(err)
	}
	if strings.Join(fr.ran, "") != "defg" {
		t.Errorf("resume ran %v, want d..g only (a-c were done)", fr.ran)
	}
	if !strings.Contains(log.String(), "partial line") {
		t.Errorf("the torn tail went unreported:\n%s", log.String())
	}
	ids, got := outputs(t, out)
	if strings.Join(ids, "") != "abcdefg" {
		t.Errorf("resumed output = %v", ids)
	}
	for id, w := range want {
		if got[id] != w {
			t.Errorf("%s: resumed %q, uninterrupted %q", id, got[id], w)
		}
	}
	// Every line of the file parses: no torn fragment survived in the middle.
	data, _ := os.ReadFile(out)
	if _, err := batchio.ScanOutput(out); err != nil || !bytes.HasSuffix(data, []byte("\n")) {
		t.Errorf("resumed file is not clean: %v", err)
	}
}

func TestProgressLine(t *testing.T) {
	ok := lineResult{status: 200, nTok: 100}
	got := progressLine(5, 20, 30, 50, "req-1", ok, 4*time.Second, 20*time.Second)
	if !strings.HasPrefix(got, "[35/50] req-1 ok  100 tok  25.0 tok/s  elapsed 20s") {
		t.Errorf("got %q", got)
	}
	if strings.Contains(got, "eta") {
		t.Error("an estimate from 5 lines is a guess; none is shown before 10")
	}
	got = progressLine(10, 20, 0, 20, "r", ok, time.Second, 50*time.Second)
	if !strings.Contains(got, "eta ~50s") { // 5 s/line × 10 left
		t.Errorf("eta: %q", got)
	}
	if got := progressLine(20, 20, 0, 20, "r", ok, time.Second, time.Minute); strings.Contains(got, "eta") {
		t.Errorf("no estimate on the last line: %q", got)
	}
	bad := progressLine(1, 2, 0, 2, "r", lineErr(400, "invalid_request_error", "x"), time.Second, time.Second)
	if !strings.Contains(bad, "FAILED invalid_request_error") {
		t.Errorf("failure not marked: %q", bad)
	}
	if !strings.HasPrefix(progressLine(3, 9, 0, 12, "r", ok, time.Second, time.Second), "[ 3/12]") {
		t.Error("position is padded to the width of the total so a log lines up")
	}
}

// batchLine's refusals happen before the model is touched, so they run on a session with no model. Each names the thing it
// refuses — the rule is that a request for something this runner cannot do is an error, never a quietly different answer.
func TestBatchLine_refusesWhatItCannotDo(t *testing.T) {
	s := &session{}
	bd := batchDefaults{maxTok: 8}
	run := func(body string) lineResult {
		return s.batchLine(context.Background(), bd, "m", batchio.Line{CustomID: "x", Body: json.RawMessage(body)})
	}
	msg := `"messages":[{"role":"user","content":"hi"}]`
	cases := map[string]struct{ body, want string }{
		"not json":           {`[1,2]`, "not a chat completion request"},
		"no messages":        {`{"messages":[]}`, "messages is required"},
		"tools":              {`{` + msg + `,"tools":[{"type":"function","function":{"name":"f"}}]}`, "tools are not supported"},
		"logprobs":           {`{` + msg + `,"logprobs":true}`, "logprobs are not supported"},
		"top_logprobs":       {`{` + msg + `,"top_logprobs":3}`, "logprobs are not supported"},
		"confidence":         {`{` + msg + `,"goinfer_confidence":true}`, "goinfer_confidence"},
		"n=2":                {`{` + msg + `,"n":2}`, "n must be 1"},
		"image":              {`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:,"}}]}]}`, "image inputs"},
		"tool message":       {`{"messages":[{"role":"user","content":"a"},{"role":"tool","content":"r"}]}`, "tool messages"},
		"assistant calls":    {`{"messages":[{"role":"assistant","content":"","tool_calls":[{"id":"1"}]}]}`, "tool_calls"},
		"reasoning format":   {`{` + msg + `,"reasoning_format":"deepseek-legacy"}`, "reasoning_format"},
		"enable_thinking":    {`{` + msg + `,"chat_template_kwargs":{"enable_thinking":"yes"}}`, "must be a boolean"},
		"negative budget":    {`{` + msg + `,"thinking_token_budget":-1}`, "must be positive"},
		"response_format":    {`{` + msg + `,"response_format":{"type":"xml"}}`, "unsupported response_format"},
		"schema missing":     {`{` + msg + `,"response_format":{"type":"json_schema"}}`, "requires a schema"},
		"negative temp":      {`{` + msg + `,"temperature":-1}`, "temperature must be >= 0"},
		"top_p range":        {`{` + msg + `,"top_p":1.5}`, "top_p must be in [0,1]"},
		"max_tokens zero":    {`{` + msg + `,"max_tokens":0}`, "max_tokens must be >= 1"},
		"max_tokens ceil":    {`{` + msg + `,"max_completion_tokens":999999}`, "exceeds the ceiling"},
		"schema won't parse": {`{` + msg + `,"response_format":{"type":"json_schema","json_schema":{"schema":{"type":7}}}}`, ""},
	}
	for name, tc := range cases {
		res := run(tc.body)
		if res.errType == "" || res.body != nil {
			t.Errorf("%s: a refusal was expected, got %+v", name, res)
			continue
		}
		if res.status != 400 || !strings.Contains(res.errMsg, tc.want) {
			t.Errorf("%s: status %d, %q; want 400 containing %q", name, res.status, res.errMsg, tc.want)
		}
	}
	// A tool list beside tool_choice "none" is not a request for tools: it is not refused on those grounds.
	if res := run(`{` + msg + `,"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none","temperature":-1}`); !strings.Contains(res.errMsg, "temperature") {
		t.Errorf("tool_choice none must let the line through to the next check: %+v", res)
	}
}

// The defaults are serve's API defaults unless the flag was passed — the point of "a file means the same thing in both places".
func TestNewBatchDefaults(t *testing.T) {
	parse := func(args ...string) batchDefaults {
		fs := flag.NewFlagSet("t", flag.ContinueOnError)
		cf := registerFlags(fs)
		if err := fs.Parse(args); err != nil {
			t.Fatal(err)
		}
		explicit := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { explicit[f.Name] = true })
		return newBatchDefaults(cf, explicit)
	}
	bd := parse("--batch", "in", "-o", "out")
	if bd.sp.Temperature != 1.0 || bd.sp.TopK != 0 || bd.sp.TopP != 0 || bd.maxTok != 512 || bd.seedSet || bd.system != "" {
		t.Errorf("untouched flags must give the API's defaults (temp 1, no top-k/p, 512 tokens, random seed, no system): %+v", bd)
	}
	bd = parse("-temp", "0", "-top-k", "5", "-seed", "9", "-max", "64", "-system", "be brief")
	if bd.sp.Temperature != 0 || bd.sp.TopK != 5 || bd.sp.Seed != 9 || !bd.seedSet || bd.maxTok != 64 || bd.system != "be brief" {
		t.Errorf("passed flags must be honoured: %+v", bd)
	}
}

func TestStopAt(t *testing.T) {
	cases := []struct {
		text  string
		from  int
		stops []string
		cut   int
		which string
		hit   bool
	}{
		{"hello END world", 0, []string{"END"}, 6, "END", true},
		{"a STOP b END", 0, []string{"END", "STOP"}, 2, "STOP", true}, // earliest wins, not first listed
		{"no stops here", 0, []string{"END"}, 0, "", false},
		{"xxEND", 3, []string{"END"}, 0, "", false}, // a match that starts before `from` was searched already
		{"xxEN", 0, []string{"END"}, 0, "", false},
	}
	for _, c := range cases {
		cut, which, hit := stopAt(c.text, c.from, c.stops)
		if hit != c.hit || (hit && (cut != c.cut || which != c.which)) {
			t.Errorf("stopAt(%q, %d, %v) = %d %q %v", c.text, c.from, c.stops, cut, which, hit)
		}
	}
	if got := parseStopField(json.RawMessage(`["a","","b"]`)); strings.Join(got, ",") != "a,b" {
		t.Errorf("empty stop strings must be dropped: %v", got)
	}
	if got := parseStopField(json.RawMessage(`"x"`)); len(got) != 1 || got[0] != "x" {
		t.Errorf("string form: %v", got)
	}
	if parseStopField(json.RawMessage(`""`)) != nil || parseStopField(nil) != nil {
		t.Error("an empty stop is no stop")
	}
}

func TestContentText(t *testing.T) {
	if s, img := contentText(json.RawMessage(`"plain"`)); s != "plain" || img {
		t.Errorf("string: %q %v", s, img)
	}
	if s, img := contentText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`)); s != "ab" || img {
		t.Errorf("parts: %q %v", s, img)
	}
	if _, img := contentText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"image_url"}]`)); !img {
		t.Error("an image part must be reported")
	}
	if s, img := contentText(nil); s != "" || img {
		t.Errorf("absent: %q %v", s, img)
	}
}

// A batch line's reasoning_effort beats the session's --reasoning-effort for gpt-oss; an effort gpt-oss does not know is ignored
// (the session's stands); and no other family's prompt moves, so a file written for one model and run on another is not changed by it.
func TestBatchTemplate_reasoningEffort(t *testing.T) {
	turns := []chat.Turn{{Role: "user", Content: "Hi"}}
	effortLine := func(tm *chat.Template) string {
		for _, e := range []string{"low", "medium", "high"} {
			if strings.Contains(tm.Render("", turns), "Reasoning: "+e+"\n\n") {
				return e
			}
		}
		return "?"
	}
	oss := &session{tmpl: chat.Harmony()}
	if got := effortLine(oss.batchTemplate(chat.ThinkTemplate, false, "", false)); got != "medium" {
		t.Errorf("no effort anywhere: %s, want the template's medium", got)
	}
	if got := effortLine(oss.batchTemplate(chat.ThinkTemplate, false, "low", false)); got != "low" {
		t.Errorf("line effort low: %s", got)
	}
	oss.effort = "high"
	if got := effortLine(oss.batchTemplate(chat.ThinkTemplate, false, "", false)); got != "high" {
		t.Errorf("session --reasoning-effort high, none on the line: %s", got)
	}
	if got := effortLine(oss.batchTemplate(chat.ThinkTemplate, false, "low", false)); got != "low" {
		t.Errorf("the line must beat the session: %s", got)
	}
	for _, bad := range []string{"bogus", "none", "xhigh"} {
		if got := effortLine(oss.batchTemplate(chat.ThinkTemplate, false, bad, false)); got != "high" {
			t.Errorf("line effort %q must be ignored, leaving the session's high: %s", bad, got)
		}
	}
	if got := effortLine(oss.batchTemplate(chat.ThinkOff, true, "low", true)); got != "low" {
		t.Errorf("effort must survive an explicit thinking mode and a constrained line: %s", got)
	}
	for _, tm := range []*chat.Template{chat.ChatML(), chat.Llama3()} {
		other := &session{tmpl: tm, effort: "high"}
		if a, b := other.batchTemplate(chat.ThinkTemplate, false, "low", false).Render("", turns), tm.Render("", turns); a != b {
			t.Errorf("%s: an effort moved a non-gpt-oss prompt", tm.Name())
		}
	}
}

// The REPL: /effort sets it and buildPrompt writes it into gpt-oss's system block; a bad value changes nothing.
func TestEffort_replAndBuildPrompt(t *testing.T) {
	s := &session{tmpl: chat.Harmony(), history: []msg{{"user", "Hi"}}}
	if prompt, _, _ := s.buildPrompt(); !strings.Contains(prompt, "Reasoning: medium\n\n") {
		t.Fatalf("default prompt:\n%s", prompt)
	}
	s.command("/effort HIGH")
	if s.effort != "high" {
		t.Fatalf("/effort HIGH set %q", s.effort)
	}
	if prompt, _, _ := s.buildPrompt(); !strings.Contains(prompt, "Reasoning: high\n\n") {
		t.Errorf("buildPrompt ignored the session's effort:\n%s", prompt)
	}
	s.command("/effort sideways")
	s.command("/effort")
	if s.effort != "high" {
		t.Errorf("a bad /effort changed the setting to %q", s.effort)
	}
	// Another family: the same session setting moves nothing.
	o := &session{tmpl: chat.ChatML(), effort: "high", history: []msg{{"user", "Hi"}}}
	p1, _, _ := o.buildPrompt()
	o.effort = ""
	if p2, _, _ := o.buildPrompt(); p1 != p2 {
		t.Error("--reasoning-effort moved a ChatML prompt")
	}
	// The flag exists and is empty (= template default) unless given.
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	cf := registerFlags(fs)
	if err := fs.Parse(nil); err != nil || *cf.effort != "" {
		t.Errorf("--reasoning-effort default = %q, %v", *cf.effort, err)
	}
}

// R19: the GPU context is the limit -> the message names -ctx and the model's window; the model's own window is the limit -> it does not.
func TestBatchContextMessage(t *testing.T) {
	got := batchContextMessage(11137, 8192, 32768)
	for _, want := range []string{"11137 tokens", "window is 8192", "context_length_exceeded", "-ctx 11138", "up to 32768", "not the model's limit"} {
		if !strings.Contains(got, want) {
			t.Errorf("GPU-context message lacks %q: %s", want, got)
		}
	}
	own := batchContextMessage(40000, 32768, 32768)
	if strings.Contains(own, "-ctx") || !strings.Contains(own, "context_length_exceeded") {
		t.Errorf("the model's own window is the limit; message = %s", own)
	}
}
