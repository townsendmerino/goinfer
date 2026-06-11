// Package agent is the shared core of the stdlib-agent demos: a fully-local
// RAG loop where goinfer runs the model, ken runs the retrieval, and aikit
// underlies both.
//
// Each Turn runs two phases:
//
//  1. DECIDE — goinfer's constrained decoding (constrain.JSONSchema) forces
//     the model to emit a syntactically guaranteed-valid tool decision:
//     {"action":"search","query":"..."} or {"action":"answer","query":""}.
//     The model cannot emit malformed JSON, so the tool call never fails to
//     parse — this is the goinfer party trick.
//  2. ANSWER — if the model chose search, the query is forwarded verbatim as
//     an MCP tools/call to a ken server subprocess (the go-stdlib demo binary
//     with the pre-built index + Model2Vec model baked in). The returned
//     chunks (file:line cited) are spliced into the prompt and the model
//     streams a grounded answer.
//
// The package is UI-agnostic: cmd/stdlib-agent wraps it in a terminal REPL,
// cmd/agent-web in a browser chat. Callers observe progress via Events.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
	"github.com/townsendmerino/goinfer/vision"
)

// answerSystem steers the grounded-answer phase.
const answerSystem = "You are a concise Go expert answering questions about the Go standard library. " +
	"Ground every claim in the search results you are given and cite them as file:line. " +
	"If the results do not contain the answer, say so instead of guessing."

// visionSystem steers an image turn — a plain visual assistant, NOT the Go-stdlib
// RAG persona (which derails into "I searched the stdlib…" on an image).
const visionSystem = "You are a helpful assistant. Look at the image and answer the user's " +
	"question about it directly and concisely. Describe only what you actually see."

// decideSystem steers the tool-decision phase. Kept short and imperative —
// small Instruct models follow concrete rules better than prose.
const decideSystem = "You can search the Go standard library source code. " +
	"Decide whether the user's message needs a code search.\n" +
	"Reply with ONLY a JSON object:\n" +
	`  {"action":"search","query":"<short code-search query>"}` + "\n" +
	`  {"action":"answer","query":""}` + "\n" +
	"Choose \"search\" for any question about how the stdlib works or where something is implemented. " +
	"Choose \"answer\" only for greetings or follow-ups fully covered by earlier results."

// decisionSchema is the JSON Schema the DECIDE phase is constrained to.
// constrain.JSONSchema compiles it into a token-level grammar mask: the
// model is physically unable to emit anything that does not conform.
// NOTE: goinfer's constrain subset enforces structure/enum/required, not string
// length — a maxLength here would be silently ignored, so it's omitted. The
// DECIDE generation is bounded by maxTok=96 + StopWhenComplete (stops at the
// first complete object), which keeps the query short in practice.
const decisionSchema = `{
  "type": "object",
  "properties": {
    "action": {"type": "string", "enum": ["search", "answer"]},
    "query":  {"type": "string"}
  },
  "required": ["action", "query"],
  "additionalProperties": false
}`

// decision is the parsed phase-1 output.
type decision struct {
	Action string `json:"action"`
	Query  string `json:"query"`
}

// msg is one conversation turn kept in history.
type msg struct{ role, content string }

// Options configures New. Exactly one of ModelPath / ModelBytes must be set.
type Options struct {
	ModelPath  string // path to a .gguf file or HF checkpoint dir
	ModelBytes []byte // in-memory GGUF (the -tags embed path)
	Quant      string // "" | int8 | int8int8 | int4
	Vision     string // optional vision-tower dir (Gemma 3 VL); defaults to ModelPath when it carries one. Enables TurnImage.

	KenBin  string // path to a ken MCP server binary (e.g. ken-demo-go-stdlib)
	KenTopK int    // chunks per search (default 5)

	MaxTokens   int     // max tokens per answer (default 512)
	Temperature float64 // answer phase; decide phase is always greedy
	TopK        int
	TopP        float64
	// Repetition penalties for the answer phase (OpenAI semantics). Small models
	// fall into list/loop repetition easily; a mild frequency penalty is the
	// standard cure. Applied only to ANSWER (DECIDE stays greedy + constrained).
	FrequencyPenalty float64
	PresencePenalty  float64
}

// Events lets a UI observe a Turn as it happens. All callbacks are optional
// and are invoked synchronously from the Turn goroutine.
type Events struct {
	// Decision fires after the constrained DECIDE phase.
	Decision func(action, query string)
	// Search fires after the ken tools/call returns (err != nil on failure;
	// the turn continues unaided).
	Search func(query, results string, err error)
	// Token fires for each newly-completed span of answer text (UTF-8 safe).
	Token func(text string)
}

// Session holds the loaded model, the ken MCP client, and the chat history.
// Turn/Reset are safe for concurrent use; turns are serialized internally.
type Session struct {
	tk      *tokenizer.Tokenizer
	model   *decoder.Model
	special tokenizer.SpecialTokens
	tmpl    *chat.Template
	stopIDs []int
	vocab   int

	// Vision tower (Gemma 3 VL) — nil unless Options.Vision is set / auto-discovered.
	// When present, TurnImage runs an image through preprocess → encoder → projector
	// and answers it via decoder.GenerateVL.
	venc    *vision.Encoder
	vproj   *vision.Projector
	vcfg    vision.Config
	vimgTok int

	ken *kenClient

	opts Options

	mu      sync.Mutex
	history []msg

	// LoadSummary is a one-line description of the loaded model, for logs.
	LoadSummary string
}

// New loads the model, spawns the ken MCP server subprocess, and performs
// the MCP handshake.
func New(ctx context.Context, o Options) (*Session, error) {
	if o.KenTopK <= 0 {
		o.KenTopK = 4
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 512
	}
	dopts := decoder.Options{Backend: "cpu", Quant: o.Quant}
	// Tune matmul parallelism for batch=1 decode — same call as demo/chat.
	decoder.SetDecodeParallelThreshold(decoder.DefaultDecodeParallelThreshold)

	var s *Session
	var err error
	switch {
	case o.ModelPath != "":
		s, err = loadFromPath(o.ModelPath, dopts)
	case o.ModelBytes != nil:
		s, err = loadFromBytes(o.ModelBytes, dopts)
	default:
		return nil, fmt.Errorf("agent: Options.ModelPath or Options.ModelBytes required")
	}
	if err != nil {
		return nil, err
	}
	s.opts = o

	// Vision tower (optional): -vision dir, else the ModelPath dir when it carries
	// a Gemma 3 VL tower (auto-discover). Enables TurnImage.
	if vdir := o.Vision; vdir != "" || o.ModelPath != "" {
		if vdir == "" {
			if _, derr := vision.LoadProjector(o.ModelPath); derr == nil {
				vdir = o.ModelPath // the model dir carries a projector → it's a VL checkpoint
			}
		}
		if vdir != "" {
			if err := s.loadVision(vdir); err != nil {
				return nil, err
			}
		}
	}

	kc, err := dialKen(ctx, o.KenBin, o.KenTopK)
	if err != nil {
		return nil, fmt.Errorf("start ken MCP server %q: %w", o.KenBin, err)
	}
	s.ken = kc
	return s, nil
}

// Close shuts down the ken subprocess.
func (s *Session) Close() {
	if s.ken != nil {
		s.ken.close()
	}
}

// Tools lists the tool names the ken server advertised at handshake.
func (s *Session) Tools() []string { return s.ken.tools }

// Reset clears the conversation history.
func (s *Session) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = nil
}

// Turn runs the two-phase DECIDE → (search?) → ANSWER pipeline for one user
// message and returns the assistant reply. Cancel ctx to abort generation.
func (s *Session) Turn(ctx context.Context, user string, ev Events) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Phase 1: constrained tool decision (greedy, deterministic).
	d := s.decide(ctx, user)
	if ev.Decision != nil {
		ev.Decision(d.Action, d.Query)
	}

	// Phase 2: optional retrieval, then a grounded streamed answer.
	answerTurns := append([]msg(nil), s.history...)
	if d.Action == "search" && d.Query != "" {
		results, err := s.ken.search(ctx, d.Query)
		if ev.Search != nil {
			ev.Search(d.Query, results, err)
		}
		if err == nil {
			// Spliced as a user-role context turn: universally supported by
			// chat templates (no tool role needed for a 0.5B model).
			answerTurns = append(answerTurns, msg{"user",
				"Search results from the Go stdlib index for \"" + d.Query + "\":\n\n" + results +
					"\n\nUse only these results to answer, citing file:line."})
		}
	}
	answerTurns = append(answerTurns, msg{"user", user})

	sp := decoder.SamplingParams{
		Temperature:      s.opts.Temperature,
		TopK:             s.opts.TopK,
		TopP:             s.opts.TopP,
		FrequencyPenalty: s.opts.FrequencyPenalty,
		PresencePenalty:  s.opts.PresencePenalty,
	}
	reply, err := s.generate(ctx, answerSystem, answerTurns, sp, s.opts.MaxTokens, ev.Token)

	// Only the clean user/assistant exchange enters long-term history — the
	// bulky search results stay out so context doesn't balloon across turns.
	s.history = append(s.history, msg{"user", user})
	if reply != "" {
		s.history = append(s.history, msg{"assistant", reply})
	}
	return reply, err
}

// HasVision reports whether a vision tower is loaded (TurnImage usable).
func (s *Session) HasVision() bool { return s.venc != nil && s.vproj != nil }

// loadVision attaches a Gemma 3 SigLIP encoder + projector from dir and resolves
// the image-soft-token id (the placeholder the embed-by-vector seam overrides).
func (s *Session) loadVision(dir string) error {
	enc, err := vision.LoadEncoder(dir)
	if err != nil {
		return fmt.Errorf("load vision encoder (%s): %w", dir, err)
	}
	proj, err := vision.LoadProjector(dir)
	if err != nil {
		return fmt.Errorf("load vision projector (%s): %w", dir, err)
	}
	id, ok := s.tk.TokenID(vision.ImageSoftToken)
	if !ok {
		return fmt.Errorf("vision: tokenizer has no %q token", vision.ImageSoftToken)
	}
	s.venc, s.vproj, s.vcfg, s.vimgTok = enc, proj, vision.Gemma3(), id
	return nil
}

// TurnImage answers a question about an image: it runs the image through the
// vision tower (preprocess → SigLIP encoder → projector), assembles the Gemma 3
// image block into the user turn, and generates a grounded answer via
// decoder.GenerateVL. An image turn skips the RAG search — the image is the
// context, not the Go-stdlib index. NOTE: the SigLIP prefill is heavy on CPU
// (minutes per image); show a "processing…" affordance.
func (s *Session) TurnImage(ctx context.Context, user string, image []byte, ev Events) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.HasVision() {
		return "", fmt.Errorf("agent: no vision tower loaded (set Options.Vision)")
	}
	pv, err := vision.Preprocess(image, s.vcfg)
	if err != nil {
		return "", err
	}
	hidden, err := s.venc.Forward(pv.Data)
	if err != nil {
		return "", fmt.Errorf("vision encoder: %w", err)
	}
	feats, err := s.vproj.Forward(hidden)
	if err != nil {
		return "", fmt.Errorf("vision projector: %w", err)
	}
	n := s.vproj.MMTokens()

	turns := append([]msg(nil), s.history...)
	turns = append(turns, msg{"user", vision.Gemma3ImageBlock(n) + "\n" + user})
	ids, err := s.tk.Encode(s.buildPrompt(visionSystem, turns), s.tmpl == nil)
	if err != nil {
		return "", fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := vision.FindImageRun(ids, s.vimgTok)
	if imgLen != n {
		return "", fmt.Errorf("image placeholder run = %d, want %d (template/tokenizer mismatch)", imgLen, n)
	}

	sp := decoder.SamplingParams{
		Temperature:      s.opts.Temperature,
		TopK:             s.opts.TopK,
		TopP:             s.opts.TopP,
		FrequencyPenalty: s.opts.FrequencyPenalty,
		PresencePenalty:  s.opts.PresencePenalty,
		StopIDs:          s.stopIDs,
	}
	tokens, gen := s.model.GenerateVL(ctx, ids, feats, imgPos, imgLen, s.opts.MaxTokens, sp)
	reply, err := s.streamGen(ctx, tokens, gen, ev.Token)

	// History keeps the user text + reply (the image bytes are not retained).
	s.history = append(s.history, msg{"user", user})
	if reply != "" {
		s.history = append(s.history, msg{"assistant", reply})
	}
	return reply, err
}

// decide runs the constrained phase-1 generation and parses its JSON. The
// grammar mask guarantees conformance, so a parse failure would indicate a
// bug, not bad model output — we still fail soft (search with the raw user
// text) to keep the demo resilient.
func (s *Session) decide(ctx context.Context, user string) decision {
	turns := append(append([]msg(nil), s.history...), msg{"user", user})
	sp := decoder.SamplingParams{Temperature: 0}
	sp.LogitProcessor = s.schemaMasker([]byte(decisionSchema))
	raw, _ := s.generate(ctx, decideSystem, turns, sp, 96, nil)

	var d decision
	if err := json.Unmarshal([]byte(raw), &d); err != nil {
		return decision{Action: "search", Query: user}
	}
	return d
}

// generate renders system+turns through the chat template, runs one
// generation, emitting completed text spans to onToken (if non-nil), and
// returns the full text.
func (s *Session) generate(ctx context.Context, system string, turns []msg, sp decoder.SamplingParams, maxTok int, onToken func(string)) (string, error) {
	prompt := s.buildPrompt(system, turns)
	ids, err := s.tk.Encode(prompt, s.tmpl == nil /* addBOS */)
	if err != nil {
		return "", fmt.Errorf("encode: %w", err)
	}
	sp.StopIDs = s.stopIDs
	tokens, gen := s.model.Generate(ctx, ids, maxTok, sp)
	return s.streamGen(ctx, tokens, gen, onToken)
}

// streamGen drains a token channel into text with UTF-8 holdback, emitting each
// newly-completed span to onToken. Shared by the text path (generate) and the
// vision path (generateImage).
func (s *Session) streamGen(ctx context.Context, tokens <-chan int, gen *decoder.Generation, onToken func(string)) (string, error) {
	// Decode the whole generated slice each step and emit only newly-completed
	// bytes (a byte-fallback token may be a partial rune).
	var out []int
	emitted := 0
	flush := func(final bool) {
		if onToken == nil {
			return
		}
		text, derr := s.tk.Decode(out)
		if derr != nil {
			return
		}
		b := []byte(text)
		end := len(b)
		if !final {
			end = completeUTF8Len(b)
		}
		if end > emitted {
			onToken(string(b[emitted:end]))
			emitted = end
		}
	}
	for id := range tokens {
		out = append(out, id)
		flush(false)
	}
	flush(true)

	if err := gen.Err(); err != nil && ctx.Err() == nil {
		return "", fmt.Errorf("generation: %w", err)
	}
	text, _ := s.tk.Decode(out)
	return strings.TrimSpace(text), nil
}

// buildPrompt renders system + turns via the detected chat template, falling
// back to plain concatenation for unrecognized templates.
func (s *Session) buildPrompt(system string, turns []msg) string {
	ct := make([]chat.Turn, len(turns))
	for i, m := range turns {
		ct[i] = chat.Turn{Role: m.role, Content: m.content}
	}
	if s.tmpl != nil {
		return s.tmpl.Render(system, ct)
	}
	var b strings.Builder
	if system != "" {
		b.WriteString(system + "\n\n")
	}
	for _, t := range ct {
		b.WriteString(t.Content + "\n")
	}
	return b.String()
}

// schemaMasker compiles a JSON Schema into goinfer's logit-masking grammar:
// the model cannot emit non-conforming output, and generation stops at the
// first complete document.
func (s *Session) schemaMasker(schema []byte) func(generated []int, logits []float32) {
	g, err := constrain.JSONSchema(schema)
	if err != nil { // compile-time constant schema: this is a programmer error
		panic(fmt.Sprintf("agent: bad decision schema: %v", err))
	}
	var eos []int
	for _, id := range []int{s.special.EOS, s.special.EndOfTurn} {
		if id >= 0 {
			eos = append(eos, id)
		}
	}
	return constrain.NewMasker(g, constrain.TokenBytes(s.vocab, s.tk.TokenText), eos).StopWhenComplete().Process
}

// loadFromPath loads tokenizer + model from a .gguf file or HF checkpoint dir.
func loadFromPath(path string, opts decoder.Options) (*Session, error) {
	loadTok := tokenizer.Load
	if strings.HasSuffix(path, ".gguf") {
		loadTok = tokenizer.LoadGGUF
	}
	t0 := time.Now()
	tk, err := loadTok(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	model, err := decoder.Load(path, opts)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return newSession(tk, model, time.Since(t0)), nil
}

// loadFromBytes loads tokenizer + model from an in-memory GGUF slice (the
// -tags embed path).
func loadFromBytes(raw []byte, opts decoder.Options) (*Session, error) {
	t0 := time.Now()
	tk, err := tokenizer.LoadGGUFBytes(raw)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	model, err := decoder.LoadGGUFBytes(raw, opts)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return newSession(tk, model, time.Since(t0)), nil
}

func newSession(tk *tokenizer.Tokenizer, model *decoder.Model, dt time.Duration) *Session {
	cfg := model.Config()
	s := &Session{
		tk: tk, model: model, special: tk.Special(), vocab: cfg.VocabSize,
		LoadSummary: fmt.Sprintf("loaded %d-layer model (hidden %d, vocab %d) in %s [quant=%s]",
			cfg.NumLayers, cfg.HiddenDim, cfg.VocabSize, dt.Round(time.Millisecond), model.Quant()),
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		s.LoadSummary += " (no recognized chat template; raw completions)"
		return s
	}
	s.tmpl = tmpl
	for _, str := range tmpl.Stops().Strings {
		if id, ok := tk.TokenID(str); ok {
			s.stopIDs = append(s.stopIDs, id)
		}
	}
	return s
}

func completeUTF8Len(b []byte) int {
	i := 0
	for i < len(b) {
		if b[i] < utf8.RuneSelf {
			i++
			continue
		}
		if !utf8.FullRune(b[i:]) {
			break
		}
		_, size := utf8.DecodeRune(b[i:])
		i += size
	}
	return i
}
