package chatapp

// `goinfer-chat --batch in.jsonl -o out.jsonl` — J5 (docs/tasks/task-work-queue-2026-09.md): the same batch file serve's
// POST /v1/batches reads, run locally and in-process, no server. The file format is internal/batchio's, shared with serve's
// code rather than copied, so one file works in both places.
//
// It is RESUMABLE, which is the point of a runner you leave alone for hours. Every finished line is appended to the output
// file and fsynced before the next starts; on restart the output is scanned, the custom_ids that already hold a response are
// skipped, and a final line the process died in the middle of is cut off and run again. A 20,000-line job that dies at line
// 14,000 starts at 14,001.
//
// Files: out.jsonl holds the lines that produced a response, in input order; out.errors.jsonl (beside it, rewritten each run)
// holds the lines that failed this run, in the shape of serve's error file. A failed line is NOT done, so a rerun retries it.
// Ctrl-C finishes nothing half-way: the line in flight is dropped (and run again next time), the rest stay untouched.
//
// Scope is serve's own batch scope — text chat. A line asking for something this runner would otherwise drop silently (tools,
// images, logprobs, n>1) is refused with an error line naming it, never quietly answered without it.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	mathrand "math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/batchio"
)

// apiDefaultMaxTokens is serve's default reply length for a request that names none (internal/serveapp defaultMaxTokens).
const apiDefaultMaxTokens = 512

// apiMaxOutputTokens is serve's ceiling on a request's max_tokens (maxOutputTokensCeiling).
const apiMaxOutputTokens = 131072

// batchPlan is what a run starts from: the parsed input and what an existing output file says is already done.
type batchPlan struct {
	in, out string
	lines   []batchio.Line
	todo    []int // indexes into lines still to run, in input order
	prog    batchio.Progress
	orphans []string // done ids the input does not contain (an output file from a different input)
}

// planBatch reads the input and the existing output, before any model is loaded — a bad file should cost a second, not a load.
func planBatch(in, out string) (*batchPlan, error) {
	if in == "" || out == "" {
		return nil, errors.New("--batch <in.jsonl> and -o <out.jsonl> go together")
	}
	if same, _ := sameFile(in, out); same {
		return nil, fmt.Errorf("-o %s is the input file; the output would overwrite it", out)
	}
	data, err := os.ReadFile(in)
	if err != nil {
		return nil, err
	}
	lines, err := batchio.ParseInput(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", in, err)
	}
	if err := batchio.CheckUnique(lines); err != nil {
		return nil, fmt.Errorf("%s: %w", in, err)
	}
	prog, err := batchio.ScanOutput(out)
	if err != nil {
		return nil, err
	}
	p := &batchPlan{in: in, out: out, lines: lines, prog: prog, orphans: batchio.Orphans(prog.Done, lines)}
	for i, l := range lines {
		if !prog.Done[l.CustomID] {
			p.todo = append(p.todo, i)
		}
	}
	return p, nil
}

func sameFile(a, b string) (bool, error) {
	ia, err := os.Stat(a)
	if err != nil {
		return false, err
	}
	ib, err := os.Stat(b)
	if err != nil {
		return false, err // the output not existing yet is the normal case
	}
	return os.SameFile(ia, ib), nil
}

// lineResult is one line's outcome: a response body, or an error.
type lineResult struct {
	body    map[string]any // the chat.completion body; nil on error
	status  int
	errType string // "" on success
	errMsg  string
	nTok    int // generated tokens, for the progress line only
}

func lineErr(status int, errType, msg string) lineResult {
	return lineResult{status: status, errType: errType, errMsg: msg}
}

// batchSummary is how a run ended.
type batchSummary struct {
	Already     int // lines the output file already held
	OK, Failed  int // lines finished this run
	Remaining   int // lines this run never got to (it was interrupted); failed lines are counted in Failed
	Interrupted bool
}

// runBatchLines runs plan.todo in order through run, appending each result as it lands. run gets a context that is cancelled
// on interrupt; a result that arrives with ctx cancelled is dropped, never recorded (it may be a partial generation).
func runBatchLines(ctx context.Context, plan *batchPlan, run func(context.Context, batchio.Line) lineResult, log io.Writer) (batchSummary, error) {
	sum := batchSummary{Already: len(plan.lines) - len(plan.todo)}
	if plan.prog.Torn {
		fmt.Fprintf(log, "%s ended in a partial line (a run killed mid-write): cutting it off, that line runs again\n", plan.out)
	}
	if len(plan.orphans) > 0 {
		fmt.Fprintf(log, "note: %s holds %d custom_id(s) this input does not contain (first: %q) — kept as they are; is this the right output file?\n",
			plan.out, len(plan.orphans), plan.orphans[0])
	}
	out, err := batchio.OpenAppend(plan.out, plan.prog.GoodBytes)
	if err != nil {
		return sum, err
	}
	defer out.Close()
	errPath := batchio.ErrorPath(plan.out)
	if err := os.Remove(errPath); err != nil && !errors.Is(err, os.ErrNotExist) { // last run's failures: this run retries those lines
		return sum, err
	}
	var errs *batchio.Writer // created on the first failure, so a clean run leaves no empty file
	defer func() {
		if errs != nil {
			errs.Close()
		}
	}()

	var (
		mu      sync.Mutex
		current string
		since   time.Time
	)
	stopBeat := make(chan struct{})
	var beat sync.WaitGroup
	beat.Add(1)
	go func() { // a line can take minutes on a big model; say it is alive rather than leave the log silent
		defer beat.Done()
		t := time.NewTicker(60 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-stopBeat:
				return
			case <-t.C:
				mu.Lock()
				if current != "" {
					fmt.Fprintf(log, "  … %s still running (%s)\n", current, time.Since(since).Round(time.Second))
				}
				mu.Unlock()
			}
		}
	}()
	defer func() { close(stopBeat); beat.Wait() }()

	start := time.Now()
	total := len(plan.lines)
	for n, idx := range plan.todo {
		if ctx.Err() != nil {
			break
		}
		l := plan.lines[idx]
		mu.Lock()
		current, since = l.CustomID, time.Now()
		mu.Unlock()
		res := run(ctx, l)
		mu.Lock()
		lineDur := time.Since(since)
		current = ""
		mu.Unlock()
		if ctx.Err() != nil {
			break // interrupted mid-line: drop it, it runs again on resume
		}
		id := "batch_req_" + newReqID()
		if res.errType == "" && res.body != nil {
			if err := out.Append(batchio.OKLine(id, l.CustomID, res.status, res.body)); err != nil {
				return sum, fmt.Errorf("writing %s: %w", plan.out, err)
			}
			sum.OK++
		} else {
			if errs == nil {
				if errs, err = batchio.OpenAppend(errPath, 0); err != nil {
					return sum, err
				}
			}
			if err := errs.Append(batchio.ErrLine(id, l.CustomID, res.errType, res.errMsg)); err != nil {
				return sum, fmt.Errorf("writing %s: %w", errPath, err)
			}
			sum.Failed++
		}
		fmt.Fprintln(log, progressLine(n+1, len(plan.todo), sum.Already, total, l.CustomID, res, lineDur, time.Since(start)))
	}
	sum.Remaining = len(plan.todo) - sum.OK - sum.Failed
	sum.Interrupted = ctx.Err() != nil
	return sum, nil
}

// progressLine is one line's report: position, id, outcome, speed, elapsed. The estimate appears only once ten lines of this run
// have finished, and is a mean over them — lines differ in cost, so it is a guess and is marked as one.
func progressLine(n, ofTodo, already, total int, id string, r lineResult, dur, elapsed time.Duration) string {
	status := "ok"
	if r.errType != "" {
		status = "FAILED " + r.errType
	}
	speed := ""
	if r.nTok > 0 && dur > 0 {
		speed = fmt.Sprintf("  %d tok  %.1f tok/s", r.nTok, float64(r.nTok)/dur.Seconds())
	}
	eta := ""
	if n >= 10 && n < ofTodo {
		eta = fmt.Sprintf("  eta ~%s", (elapsed / time.Duration(n) * time.Duration(ofTodo-n)).Round(time.Second))
	}
	return fmt.Sprintf("[%*d/%d] %s %s%s  elapsed %s%s", len(fmt.Sprint(total)), already+n, total, id, status, speed, elapsed.Round(time.Second), eta)
}

func newReqID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// ---- one line through the model ----

// batchDefaults are the settings of a line that does not state them. They are serve's API defaults, NOT this binary's
// interactive ones (temperature 1.0, no top-k/top-p, a random seed, 512 tokens, no system prompt), so a file means the same
// thing here as posted to serve — except where the user passed the flag explicitly, which is an instruction about this run.
type batchDefaults struct {
	sp      decoder.SamplingParams
	seedSet bool // --seed was passed: a line without a seed uses it; otherwise each gets a fresh random one, as serve does
	maxTok  int
	system  string
}

func newBatchDefaults(cf *chatFlags, explicit map[string]bool) batchDefaults {
	bd := batchDefaults{sp: decoder.SamplingParams{Temperature: 1.0}, maxTok: apiDefaultMaxTokens}
	if explicit["temp"] {
		bd.sp.Temperature = *cf.temp
	}
	if explicit["top-k"] {
		bd.sp.TopK = *cf.topK
	}
	if explicit["top-p"] {
		bd.sp.TopP = *cf.topP
	}
	if explicit["min-p"] {
		bd.sp.MinP = *cf.minP
	}
	if explicit["repeat-penalty"] {
		bd.sp.RepeatPenalty = *cf.repPen
	}
	if explicit["presence-penalty"] {
		bd.sp.PresencePenalty = *cf.presPen
	}
	if explicit["frequency-penalty"] {
		bd.sp.FrequencyPenalty = *cf.freqPen
	}
	if explicit["repeat-last-n"] {
		bd.sp.RepeatLastN = *cf.repLastN
	}
	if explicit["seed"] {
		bd.sp.Seed, bd.seedSet = *cf.seed, true
	}
	if explicit["max"] {
		bd.maxTok = *cf.maxTok
	}
	if explicit["system"] {
		bd.system = *cf.system
	}
	return bd
}

// batchChatReq is the subset of an OpenAI chat request a batch line may carry. Unknown keys are ignored, as serve ignores them;
// the ones below that this runner cannot honour are refused by name in batchLine.
type batchChatReq struct {
	Model    string `json:"model"`
	Messages []struct {
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		ToolCalls        json.RawMessage `json:"tool_calls"`
		ReasoningContent string          `json:"reasoning_content"`
		Reasoning        string          `json:"reasoning"`
	} `json:"messages"`
	Tools      json.RawMessage `json:"tools"`
	ToolChoice json.RawMessage `json:"tool_choice"`
	N          *int            `json:"n"`

	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Seed                *int64          `json:"seed"`
	FrequencyPenalty    *float64        `json:"frequency_penalty"`
	PresencePenalty     *float64        `json:"presence_penalty"`
	Stop                json.RawMessage `json:"stop"`
	Logprobs            bool            `json:"logprobs"`
	TopLogprobs         *int            `json:"top_logprobs"`
	Confidence          bool            `json:"goinfer_confidence"`
	ResponseFormat      *struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`

	ChatTemplateKwargs  map[string]json.RawMessage `json:"chat_template_kwargs"`
	ReasoningEffort     string                     `json:"reasoning_effort"`
	ReasoningFormat     string                     `json:"reasoning_format"`
	ThinkingTokenBudget *int                       `json:"thinking_token_budget"`
}

// contentText is a message's text: a plain string, or the concatenated text parts of a content array. An image part is
// reported so the line can be refused; serve's batch refuses images the same way.
func contentText(raw json.RawMessage) (text string, hasImage bool) {
	if len(raw) == 0 {
		return "", false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, false
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return "", false
	}
	var b strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text":
			b.WriteString(p.Text)
		case "image_url", "input_image", "image":
			hasImage = true
		}
	}
	return b.String(), hasImage
}

// stopAt finds the earliest stop string in text, looking only from `from` (earlier text was already searched).
func stopAt(text string, from int, stops []string) (cut int, which string, hit bool) {
	cut = -1
	for _, st := range stops {
		if st == "" {
			continue
		}
		if i := strings.Index(text[from:], st); i >= 0 && (cut < 0 || from+i < cut) {
			cut, which = from+i, st
		}
	}
	return cut, which, cut >= 0
}

func parseStopField(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		if one == "" {
			return nil
		}
		return []string{one}
	}
	var many, kept []string
	_ = json.Unmarshal(raw, &many)
	for _, st := range many {
		if st != "" {
			kept = append(kept, st)
		}
	}
	return kept
}

// batchLine runs one batch line through the loaded model and returns its chat.completion body — the same shape serve's batch
// puts in an output line — or the error that line earns. ctx is the run's (Ctrl-C); a stop string ends only this line.
func (s *session) batchLine(ctx context.Context, bd batchDefaults, modelName string, l batchio.Line) lineResult {
	var req batchChatReq
	if err := json.Unmarshal(l.Body, &req); err != nil {
		return lineErr(400, "invalid_request_error", "body is not a chat completion request: "+err.Error())
	}
	if len(req.Messages) == 0 {
		return lineErr(400, "invalid_request_error", "messages is required and must contain at least one message")
	}
	// Refused by name rather than dropped: an unrecognised key is silently ignored by design, so a request for something
	// this runner cannot do must be an error or the answer would read as if it had been done.
	if len(req.Tools) > 0 && strings.TrimSpace(string(req.ToolChoice)) != `"none"` {
		return lineErr(400, "invalid_request_error", "tools are not supported by --batch (text chat only); use goinfer-serve")
	}
	if req.Logprobs || req.TopLogprobs != nil {
		return lineErr(400, "invalid_request_error", "logprobs are not supported by --batch; use goinfer-serve")
	}
	if req.Confidence {
		return lineErr(400, "invalid_request_error", "goinfer_confidence is not supported by --batch; use goinfer-serve")
	}
	if req.N != nil && *req.N != 1 {
		return lineErr(400, "invalid_request_error", fmt.Sprintf("n must be 1 (got %d): one reply per line", *req.N))
	}

	system := bd.system
	var turns []chat.Turn
	for _, m := range req.Messages {
		text, img := contentText(m.Content)
		if img {
			return lineErr(400, "invalid_request_error", "image inputs are not supported by --batch; use goinfer-serve")
		}
		switch m.Role {
		case "system", "developer":
			system = text
		case "tool":
			return lineErr(400, "invalid_request_error", "tool messages are not supported by --batch; use goinfer-serve")
		case "assistant":
			if len(m.ToolCalls) > 0 && string(m.ToolCalls) != "null" {
				return lineErr(400, "invalid_request_error", "assistant tool_calls are not supported by --batch; use goinfer-serve")
			}
			r := m.ReasoningContent
			if r == "" {
				r = m.Reasoning
			}
			turns = append(turns, chat.Turn{Role: "assistant", Content: text, Reasoning: r})
		default:
			turns = append(turns, chat.Turn{Role: "user", Content: text})
		}
	}
	system = strings.TrimSpace(system)

	// Thinking: the session's mode (--thinking), moved by the request the way serve moves it (think.go resolveThink) — an
	// explicit enable_thinking, or reasoning_effort "none" to turn it off (any other effort cannot turn it on).
	mode, explicitMode := s.think, false
	if raw, ok := req.ChatTemplateKwargs["enable_thinking"]; ok {
		var on bool
		if err := json.Unmarshal(raw, &on); err != nil {
			return lineErr(400, "invalid_request_error", "chat_template_kwargs.enable_thinking must be a boolean")
		}
		mode, explicitMode = chat.ThinkOff, true
		if on {
			mode = chat.ThinkOn
		}
	} else if strings.EqualFold(strings.TrimSpace(req.ReasoningEffort), "none") {
		mode, explicitMode = chat.ThinkOff, true
	}
	if req.ThinkingTokenBudget != nil && *req.ThinkingTokenBudget < 0 {
		return lineErr(400, "invalid_request_error", fmt.Sprintf("the thinking token budget must be positive, got %d", *req.ThinkingTokenBudget))
	}
	splitReasoning := true
	switch req.ReasoningFormat {
	case "", "deepseek":
	case "none":
		splitReasoning = false // the reasoning stays in the content, as serve's -reasoning-format none leaves it
	default:
		return lineErr(400, "invalid_request_error", fmt.Sprintf("reasoning_format %q is not supported by --batch: want deepseek or none", req.ReasoningFormat))
	}

	// Output constraint.
	var masker func([]int, []float32)
	constrained := false
	if rf := req.ResponseFormat; rf != nil {
		switch rf.Type {
		case "", "text":
		case "json_object":
			constrained, masker = true, s.grammarMasker(constrain.JSON())
		case "json_schema":
			if rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
				return lineErr(400, "invalid_request_error", "response_format json_schema requires a schema")
			}
			g, err := constrain.JSONSchema(rf.JSONSchema.Schema)
			if err != nil {
				return lineErr(400, "invalid_request_error", err.Error())
			}
			constrained, masker = true, s.grammarMasker(g)
		default:
			return lineErr(400, "invalid_request_error", fmt.Sprintf("unsupported response_format type %q", rf.Type))
		}
	}

	// Sampling, validated as serve's prepare validates it.
	sp := bd.sp
	if req.Temperature != nil {
		if *req.Temperature < 0 {
			return lineErr(400, "invalid_request_error", fmt.Sprintf("temperature must be >= 0 (got %v); 0 selects greedy/deterministic decoding", *req.Temperature))
		}
		sp.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		switch p := *req.TopP; {
		case p < 0 || p > 1:
			return lineErr(400, "invalid_request_error", fmt.Sprintf("top_p must be in [0,1] (got %v)", p))
		case p == 0:
			sp.Temperature = 0 // the tightest nucleus is the single likeliest token
		case p < 1:
			sp.TopP = p
		default:
			sp.TopP = 0 // 1 is "no nucleus", even over an explicit --top-p
		}
	}
	if req.TopK != nil {
		sp.TopK = *req.TopK
	}
	if req.FrequencyPenalty != nil {
		sp.FrequencyPenalty = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		sp.PresencePenalty = *req.PresencePenalty
	}
	switch {
	case req.Seed != nil:
		sp.Seed = *req.Seed
	case !bd.seedSet:
		sp.Seed = mathrand.Int63() // an omitted seed varies run to run, as in serve; an explicit one (even 0) is honoured
	}
	maxTok := bd.maxTok
	mt := req.MaxTokens
	if req.MaxCompletionTokens != nil { // OpenAI's newer field wins over max_tokens
		mt = req.MaxCompletionTokens
	}
	if mt != nil {
		if *mt < 1 {
			return lineErr(400, "invalid_request_error", fmt.Sprintf("max_tokens must be >= 1 (got %d)", *mt))
		}
		if *mt > apiMaxOutputTokens {
			return lineErr(400, "invalid_request_error", fmt.Sprintf("max_tokens %d exceeds the ceiling of %d", *mt, apiMaxOutputTokens))
		}
		maxTok = *mt
	}
	stops := parseStopField(req.Stop)

	// The prompt: the model's own template in the request's thinking mode; a grammar governs the first token, so a prompt that
	// would end inside an open think block is rendered thinking-off.
	tm := s.batchTemplate(mode, explicitMode, req.ReasoningEffort, constrained)
	var ids []int
	var err error
	if tm != nil {
		// EncodeSegments, not Encode: text in a user message that spells a special token stays text, never a forged turn boundary.
		ids, err = s.tk.EncodeSegments(tm.RenderSegments(system, turns), false)
	} else {
		ids, err = s.tk.Encode(rawPrompt(system, turns), true)
	}
	if err != nil {
		return lineErr(500, "api_error", "encode: "+err.Error())
	}
	window := s.model.Config().MaxPositions
	if rc := s.model.ResidentContextCap(); rc > 0 && (window <= 0 || rc < window) {
		window = rc
	}
	if window > 0 {
		if len(ids) >= window {
			return lineErr(400, "invalid_request_error", fmt.Sprintf("prompt is %d tokens but the model's context window is %d (context_length_exceeded)", len(ids), window))
		}
		maxTok = min(maxTok, window-len(ids))
	}

	sp.StopIDs = s.stopIDs
	if masker != nil {
		sp.LogitProcessor = masker
	}
	explicitBudget := 0
	if req.ThinkingTokenBudget != nil {
		explicitBudget = *req.ThinkingTokenBudget
	}
	s.applyBudgetFor(&sp, tm, turns, maxTok, constrained, explicitBudget)

	gctx, cancel := context.WithCancel(ctx) // a stop string ends this generation only
	defer cancel()
	stream, gen := s.startGen(gctx, ids, maxTok, sp)

	var rs *chat.ReplySplitter
	if splitReasoning {
		rs = tm.NewReplySplitter(turns) // nil for a model with no recognised thinking control
	}
	var answer, reasoning strings.Builder
	maxStop := 0
	for _, st := range stops {
		maxStop = max(maxStop, len(st))
	}
	stopHit := ""
	// addAnswer appends to the answer and applies the stop strings, which are matched on the ANSWER only.
	addAnswer := func(piece string) {
		from := max(0, answer.Len()-(maxStop-1)) // a stop may straddle the previous piece
		answer.WriteString(piece)
		if len(stops) == 0 {
			return
		}
		if cut, which, hit := stopAt(answer.String(), from, stops); hit {
			text := answer.String()[:cut]
			answer.Reset()
			answer.WriteString(text)
			stopHit = which
			cancel()
		}
	}
	nTok := 0
	for id := range stream {
		if stopHit != "" {
			continue // drain so the generation goroutine exits
		}
		nTok++
		piece, _ := s.tk.DecodePiece(id)
		if rs != nil {
			r, a := rs.Push(piece)
			reasoning.WriteString(r)
			piece = a
		}
		addAnswer(piece)
	}
	if rs != nil {
		r, a, _ := rs.Finish()
		reasoning.WriteString(r)
		if stopHit == "" {
			addAnswer(a)
		}
	}
	if ctx.Err() != nil {
		return lineErr(499, "cancelled", "interrupted") // the caller drops it; never recorded
	}
	if gerr := gen.Err(); gerr != nil && !errors.Is(gerr, context.Canceled) {
		return lineErr(500, "api_error", "generation failed: "+gerr.Error())
	}

	finish := "stop"
	if stopHit == "" {
		budget := maxTok
		if gen.BudgetClamped { // the resident context cap ended the turn short of the request: that is truncation
			budget = gen.Budget
		}
		if nTok >= budget {
			finish = "length"
		}
	}
	msg := map[string]any{"role": "assistant", "content": answer.String()}
	if reasoning.Len() > 0 {
		msg["reasoning_content"] = reasoning.String()
	}
	usage := map[string]any{
		"prompt_tokens": len(ids), "completion_tokens": nTok, "total_tokens": len(ids) + nTok,
		"prefill_reused_tokens": gen.PrefillReused,
	}
	body := map[string]any{
		"id": "batch_req_" + newReqID(), "object": "chat.completion", "created": time.Now().Unix(), "model": modelName,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}},
		"usage":   usage,
	}
	return lineResult{body: body, status: 200, nTok: nTok}
}

// modelNameOf is the name a batch response reports for the model: the file's base name without .gguf, as serve names a model it
// loads from a path.
func modelNameOf(spec string) string {
	if spec == "" {
		return "goinfer-chat"
	}
	return strings.TrimSuffix(filepath.Base(spec), ".gguf")
}

// runBatch runs the plan against the loaded model and returns the process exit code: 0 when every line produced a response,
// 1 when some failed (the output is still complete for the rest), 130 when interrupted.
func (s *session) runBatch(plan *batchPlan, bd batchDefaults, modelName string) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	run := func(ctx context.Context, l batchio.Line) lineResult { return s.batchLine(ctx, bd, modelName, l) }
	sum, err := runBatchLines(ctx, plan, run, os.Stderr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return 1
	}
	fmt.Fprintf(os.Stderr, "%s\n", sum.report(plan))
	switch {
	case sum.Interrupted:
		return 130
	case sum.Failed > 0:
		return 1
	}
	return 0
}

// report is the closing line: what this run did and what a rerun would do.
func (b batchSummary) report(plan *batchPlan) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "this run: %d ok, %d failed; %d already done before it", b.OK, b.Failed, b.Already)
	if b.Interrupted {
		fmt.Fprintf(&sb, "; interrupted with %d not finished — rerun the same command to continue from there", b.Remaining)
	} else if b.Failed > 0 {
		fmt.Fprintf(&sb, "; the %d failed line(s) are in %s, and a rerun retries them", b.Failed, batchio.ErrorPath(plan.out))
	}
	return sb.String()
}

// batchTemplate is the chat template one batch line is rendered with: the session's thinking mode (--thinking), moved by the line's
// own enable_thinking / reasoning_effort "none" (explicitMode), the line's reasoning_effort (low | medium | high) over the session's
// (--reasoning-effort) for gpt-oss, and thinking-off when a grammar governs the first token and the prompt would otherwise end
// inside an open think block. A pure function of the session's template, so it is tested without a model.
func (s *session) batchTemplate(mode chat.ThinkMode, explicitMode bool, lineEffort string, constrained bool) *chat.Template {
	tm := s.tmpl.WithThinking(s.think)
	if explicitMode {
		tm = s.tmpl.WithThinking(mode)
	}
	effort := s.effort
	if e, ok := chat.NormalizeReasoningEffort(lineEffort); ok {
		effort = e
	}
	tm = tm.WithReasoningEffort(effort)
	if constrained && tm.PromptOpensThink() {
		tm = tm.WithThinking(chat.ThinkOff)
	}
	return tm
}
