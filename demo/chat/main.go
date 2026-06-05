// Command chat is goinfer's interactive demo: a terminal REPL around a local
// decoder-only LLM running on the pure-Go decoder — no cgo, no Python, no model
// download step (in the embed build the model ships *inside* the binary).
//
// It is built for prompt iteration. Type a message and the model streams a
// reply; slash-commands tune the system prompt and sampling live so you can
// feel out what makes a small Instruct model behave:
//
//	/system <text>   set the system prompt (steers tone/role) and reset history
//	/temp <f>        sampling temperature (0 = greedy/deterministic)
//	/topk <n>        top-k filter (0 = off)        /topp <f>  nucleus (0 = off)
//	/max <n>         max tokens per reply           /seed <n> RNG seed
//	/json            toggle JSON-constrained output (model cannot emit bad JSON)
//	/reset           clear the conversation history
//	/params          show current settings          /help     list commands
//	/quit            exit (or Ctrl-D)
//
// During a reply, Ctrl-C cancels just that generation and returns to the prompt.
//
// Usage:
//
//	go run ./demo/chat --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
//	go run ./demo/chat --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	    --system "You are a terse Go expert. Answer with code, no prose."
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// defaultSystem is a code-flavored system prompt — on-brand for the Coder model
// and a sane starting point you can override with --system or /system.
const defaultSystem = "You are a helpful, concise coding assistant. Prefer correct, runnable code and short explanations."

// msg is one conversation turn kept in history.
type msg struct{ role, content string }

// session holds the loaded model + the live, user-tunable settings.
type session struct {
	tk      *tokenizer.Tokenizer
	model   *decoder.Model
	special tokenizer.SpecialTokens
	style   tokenizer.ChatStyle
	vocab   int

	system  string
	history []msg
	sp      decoder.SamplingParams
	maxTok  int
	jsonOut bool
}

func main() {
	var (
		model    = flag.String("model", "", "path to a .gguf file or HF checkpoint dir (omit in the -tags embed build to use the baked-in model)")
		system   = flag.String("system", defaultSystem, "system prompt that steers the model")
		backend  = flag.String("backend", "cpu", "compute backend: cpu | webgpu")
		quant    = flag.String("quant", "int8int8", "weight quant: \"\" (native) | int8 | int8int8 | int4. int8int8 (W8A8) uses the native int8×int8 SDOT kernel — much faster than int4's per-token nibble unpack")
		maxTok   = flag.Int("max", 512, "max tokens per reply")
		temp     = flag.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
		topK     = flag.Int("top-k", 20, "top-k filter (0 = off)")
		topP     = flag.Float64("top-p", 0.8, "top-p / nucleus (0 = off)")
		seed     = flag.Int64("seed", 0, "sampling RNG seed")
		modelTmp = flag.Bool("model-tmp", false, "embed build: stream the baked-in model to a temp file + mmap instead of loading it into memory. Lower peak RAM for big models, but needs a writable temp dir. Also via GOINFER_MODEL_TMP=1. (If your temp dir is a tmpfs / RAM-backed, this saves no RAM.)")
	)
	flag.Parse()

	opts := decoder.Options{Backend: *backend, Quant: *quant}
	useTmp := *modelTmp || os.Getenv("GOINFER_MODEL_TMP") != ""

	var s *session
	var err error
	switch {
	case *model != "":
		// Explicit checkpoint (a .gguf file or an HF dir) — load from disk.
		s, err = loadFromPath(*model, opts)
	case hasEmbeddedModel:
		// Baked-in model (-tags embed): in-memory by default — no temp file, so
		// the binary runs on a read-only filesystem. --model-tmp opts into the
		// lower-peak-RAM disk path.
		s, err = loadEmbedded(useTmp, opts)
	default:
		fmt.Fprintln(os.Stderr, "error: --model is required (path to a .gguf file or HF checkpoint dir)")
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	s.system = *system
	s.maxTok = *maxTok
	s.sp = decoder.SamplingParams{Temperature: *temp, TopK: *topK, TopP: *topP, Seed: *seed}

	s.repl()
}

// loadEmbedded loads the baked-in model. Its implementation is build-tag
// specific: -tags prequant maps a serialized weight bundle (zero-copy); -tags
// embed loads the embedded GGUF; the default build has no embedded model.

// loadFromPath loads the tokenizer + model from a path. A bare .gguf carries its
// tokenizer in metadata (LoadGGUF); an HF dir has a tokenizer.json (Load).
func loadFromPath(path string, opts decoder.Options) (*session, error) {
	loadTok := tokenizer.Load
	if strings.HasSuffix(path, ".gguf") {
		loadTok = tokenizer.LoadGGUF
	}
	t0 := time.Now()
	tk, err := loadTok(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	progress("loading + quantizing…")
	model, err := decoder.Load(path, opts)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return newSession(tk, model, opts, time.Since(t0)), nil
}

// loadFromBytes loads the tokenizer + model from an in-memory GGUF slice — the
// no-filesystem path used by the embedded binary.
func loadFromBytes(raw []byte, opts decoder.Options) (*session, error) {
	t0 := time.Now()
	tk, err := tokenizer.LoadGGUFBytes(raw)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	progress("loading + quantizing…")
	model, err := decoder.LoadGGUFBytes(raw, opts)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return newSession(tk, model, opts, time.Since(t0)), nil
}

// newSession assembles the REPL session and prints the load summary to stderr.
func newSession(tk *tokenizer.Tokenizer, model *decoder.Model, opts decoder.Options, dt time.Duration) *session {
	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "loaded %d-layer model (hidden %d, vocab %d) in %s [backend=%s quant=%s]\n",
		cfg.NumLayers, cfg.HiddenDim, cfg.VocabSize, dt.Round(time.Millisecond), opts.Backend, model.Quant())
	style := tk.ChatStyle()
	if style == tokenizer.ChatStyleNone {
		fmt.Fprintln(os.Stderr, "note: this checkpoint has no chat template; replies may be raw completions")
	}
	return &session{tk: tk, model: model, special: tk.Special(), style: style, vocab: cfg.VocabSize}
}

// progress prints a one-line status to stderr (stdout stays clean for piping).
func progress(msg string) { fmt.Fprintln(os.Stderr, msg) }

// repl is the read–eval–print loop. Lines starting with '/' are commands; every
// other line is a user turn that gets a streamed reply.
func (s *session) repl() {
	fmt.Printf("\ngoinfer chat — %d msgs of history, system prompt steers it. /help for commands, /quit to exit.\n", 0)
	fmt.Printf("system: %s\n\n", short(s.system))
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 1<<20), 1<<20) // allow long pasted prompts
	for {
		fmt.Print("\033[1myou>\033[0m ")
		if !in.Scan() {
			fmt.Println("\nbye")
			return
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "/") {
			if s.command(line) {
				return // /quit
			}
			continue
		}
		s.history = append(s.history, msg{"user", line})
		reply := s.generate()
		if reply != "" {
			s.history = append(s.history, msg{"assistant", reply})
		}
	}
}

// generate builds the templated prompt from system + history, streams the
// reply to stdout, and returns the assistant text (for history). Ctrl-C during
// generation cancels just this turn.
func (s *session) generate() string {
	prompt := s.buildPrompt()
	ids, err := s.tk.Encode(prompt, true /* addBOS */)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		return ""
	}

	sp := s.sp
	if s.special.EndOfTurn >= 0 {
		sp.StopIDs = []int{s.special.EndOfTurn} // stop at <|im_end|>/<end_of_turn>
	}
	if s.jsonOut {
		sp.LogitProcessor = s.jsonMasker()
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	stream, gen := s.model.Generate(ctx, ids, s.maxTok, sp)

	// Stream with UTF-8 holdback: decode the whole generated slice each step and
	// print only newly-completed bytes (a byte-fallback token may be a partial
	// rune).
	fmt.Print("\033[36m") // cyan reply
	var out []int
	printed := 0
	start := time.Now()
	flush := func(final bool) {
		text, derr := s.tk.Decode(out)
		if derr != nil {
			return
		}
		b := []byte(text)
		end := len(b)
		if !final {
			end = completeUTF8Len(b)
		}
		if end > printed {
			os.Stdout.Write(b[printed:end])
			printed = end
		}
	}
	for id := range stream {
		out = append(out, id)
		flush(false)
	}
	flush(true)
	fmt.Print("\033[0m\n")

	if err := gen.Err(); err != nil && ctx.Err() == nil {
		fmt.Fprintf(os.Stderr, "(generation error: %v)\n", err)
	}
	if elapsed := time.Since(start); elapsed > 0 && len(out) > 0 {
		fmt.Fprintf(os.Stderr, "\033[2m[%d tok, %.1f tok/s]\033[0m\n", len(out), float64(len(out))/elapsed.Seconds())
	}
	text, _ := s.tk.Decode(out)
	return strings.TrimSpace(text)
}

// command handles a /slash line; returns true to quit.
func (s *session) command(line string) bool {
	cmd, arg, _ := strings.Cut(line, " ")
	arg = strings.TrimSpace(arg)
	switch cmd {
	case "/quit", "/exit":
		fmt.Println("bye")
		return true
	case "/help":
		fmt.Print(helpText)
	case "/reset":
		s.history = nil
		fmt.Println("(history cleared)")
	case "/system":
		s.system = arg
		s.history = nil
		fmt.Printf("(system set; history cleared)\nsystem: %s\n", short(s.system))
	case "/temp":
		s.sp.Temperature = parseF(arg, s.sp.Temperature)
	case "/topk":
		s.sp.TopK = parseI(arg, s.sp.TopK)
	case "/topp":
		s.sp.TopP = parseF(arg, s.sp.TopP)
	case "/seed":
		s.sp.Seed = int64(parseI(arg, int(s.sp.Seed)))
	case "/max":
		s.maxTok = parseI(arg, s.maxTok)
	case "/json":
		s.jsonOut = !s.jsonOut
		fmt.Printf("(json-constrained output: %v)\n", s.jsonOut)
	case "/params":
		s.printParams()
	case "/demos", "/examples":
		fmt.Println("canned demo prompts — run with /demo <n|name>:")
		for i, d := range demos {
			tag := ""
			if d.tier == "big" {
				tag += "  \033[2m[1.5B]\033[0m"
			}
			if d.json {
				tag += "  \033[2m[json]\033[0m"
			}
			fmt.Printf("  %d  %-8s %s%s\n", i+1, d.name, firstLine(d.prompt), tag)
		}
	case "/demo", "/ex":
		d, ok := pickDemo(arg)
		if !ok {
			fmt.Printf("(no such demo %q — /demos to list)\n", arg)
			break
		}
		if d.system != "" {
			s.system = d.system
			s.history = nil
			fmt.Printf("\033[2m(system: %s)\033[0m\n", short(s.system))
		}
		prevJSON := s.jsonOut
		s.jsonOut = d.json
		fmt.Printf("\033[1myou>\033[0m %s\n", d.prompt) // echo so viewers see the prompt
		s.history = append(s.history, msg{"user", d.prompt})
		if reply := s.generate(); reply != "" {
			s.history = append(s.history, msg{"assistant", reply})
		}
		s.jsonOut = prevJSON // restore; the demo's json setting was one-shot
	default:
		fmt.Printf("unknown command %q — /help for the list\n", cmd)
	}
	return false
}

func (s *session) printParams() {
	fmt.Printf("temp=%.2f topK=%d topP=%.2f seed=%d max=%d json=%v history=%d turns\nsystem: %s\n",
		s.sp.Temperature, s.sp.TopK, s.sp.TopP, s.sp.Seed, s.maxTok, s.jsonOut, len(s.history), short(s.system))
}

// buildPrompt renders system + history into the model's chat template.
func (s *session) buildPrompt() string {
	if s.style == tokenizer.ChatStyleGemma {
		return s.buildGemma()
	}
	return s.buildChatML() // ChatML default; also a reasonable raw fallback
}

func (s *session) buildChatML() string {
	var b strings.Builder
	turn := func(role, content string) {
		b.WriteString("<|im_start|>")
		b.WriteString(role)
		b.WriteByte('\n')
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}
	if sys := strings.TrimSpace(s.system); sys != "" {
		turn("system", sys)
	}
	for _, m := range s.history {
		turn(m.role, m.content)
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

func (s *session) buildGemma() string {
	var b strings.Builder
	sys := strings.TrimSpace(s.system)
	firstUser := true
	for _, m := range s.history {
		role := m.role
		if role == "assistant" {
			role = "model"
		}
		content := m.content
		if role == "user" && firstUser {
			firstUser = false
			if sys != "" {
				content = sys + "\n\n" + content
			}
		}
		b.WriteString("<start_of_turn>")
		b.WriteString(role)
		b.WriteByte('\n')
		b.WriteString(content)
		b.WriteString("<end_of_turn>\n")
	}
	b.WriteString("<start_of_turn>model\n")
	return b.String()
}

// jsonMasker constrains output to valid JSON via logit masking; EOS/end-of-turn
// are gated until the document is complete.
func (s *session) jsonMasker() func(generated []int, logits []float32) {
	var eos []int
	for _, id := range []int{s.special.EOS, s.special.EndOfTurn} {
		if id >= 0 {
			eos = append(eos, id)
		}
	}
	m := constrain.NewMasker(constrain.JSON(), constrain.TokenBytes(s.vocab, s.tk.TokenText), eos).StopWhenComplete()
	return m.Process
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

func parseF(s string, def float64) float64 {
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v
	}
	fmt.Printf("(bad number %q, keeping %v)\n", s, def)
	return def
}

func parseI(s string, def int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	fmt.Printf("(bad integer %q, keeping %v)\n", s, def)
	return def
}

func short(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "(none)"
	}
	if len(s) > 100 {
		return s[:97] + "..."
	}
	return s
}

// demo is a canned prompt so a live or recorded session can trigger a
// known-good result with a couple of keystrokes instead of typing a long
// prompt (and risking typos on camera). Run with /demo <n|name>.
type demo struct {
	name, system, prompt string
	json                 bool   // one-shot JSON-constrained output for this prompt
	tier                 string // "" / "fast" run great on the 0.5B; "big" flatter the 1.5B
}

var demos = []demo{
	{
		name:   "bug",
		prompt: "What's the bug, and what's the fix?\n\nfunc Sum(xs []int) int {\n\ttotal := 0\n\tfor i := 1; i <= len(xs); i++ {\n\t\ttotal += xs[i]\n\t}\n\treturn total\n}",
	},
	{
		name:   "dedup",
		prompt: "Complete this function:\n\n// Dedup returns xs with duplicates removed, preserving first-seen order.\nfunc Dedup[T comparable](xs []T) []T {",
	},
	{
		name:   "mutex",
		prompt: "Write a thread-safe counter in Go using sync.Mutex. Code only, then one sentence.",
	},
	{
		name:   "reverse",
		prompt: "Write a Go function that reverses a slice of ints in place.",
	},
	{
		name:   "fim",
		prompt: "Fill in the body. Return only the completed function.\n\n// Clamp returns v limited to the range [lo, hi].\nfunc Clamp(v, lo, hi int) int {\n\t// <fill in>\n}",
	},
	{
		name:   "range",
		prompt: "Rewrite this loop to use range, keeping behavior identical:\n\nfor i := 0; i < len(items); i++ {\n\tfmt.Println(i, items[i])\n}",
	},
	{
		name:   "json",
		json:   true,
		prompt: "Extract the name, version, and language as a JSON object from:\nken v0.4.0 — a fast Go code-search tool for agents",
	},
	// Bigger-model tier: multi-step bugs, small algorithms, concurrency, a
	// conceptual answer, richer extraction — these flatter the 1.5B (and still run
	// on the 0.5B). Kept brief: at ~20 tok/s a 400-token answer is ~20 s on screen.
	{name: "race", tier: "big", prompt: "What's the bug and the fix?\n\nfunc main() {\n\tfor i := 0; i < 3; i++ {\n\t\tgo func() { fmt.Println(i) }()\n\t}\n\ttime.Sleep(time.Second)\n}"},
	{name: "lru", tier: "big", prompt: "Implement an LRU cache in Go with O(1) Get and Put. Code only, brief."},
	{name: "pool", tier: "big", prompt: "Write a worker pool in Go: N goroutines consume a jobs channel and send results on another channel, coordinated with sync.WaitGroup. Concise."},
	{name: "test", tier: "big", prompt: "Write IsBalanced(s string) bool that checks balanced (), [], and {}, plus a table-driven test for it. Concise."},
	{name: "niltl", tier: "big", prompt: "Explain the difference between a nil slice and an empty slice in Go, with a one-line example of each. Two sentences."},
	{name: "wrap", tier: "big", prompt: "Write a Go function that opens a file and returns a wrapped error with %w, then show a caller using errors.Is. Concise."},
	{name: "extract", json: true, prompt: "Extract repo, version, language, and license as a JSON object from:\nken v0.4.0 is an MIT-licensed Go code-search tool."},
}

// pickDemo resolves a /demo argument by 1-based index or by name.
func pickDemo(arg string) (demo, bool) {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		return demo{}, false
	}
	if n, err := strconv.Atoi(arg); err == nil {
		if n >= 1 && n <= len(demos) {
			return demos[n-1], true
		}
		return demo{}, false
	}
	for _, d := range demos {
		if strings.EqualFold(d.name, arg) {
			return d, true
		}
	}
	return demo{}, false
}

// firstLine renders a one-line preview of a (possibly multi-line) prompt.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i] + " …"
	}
	if len(s) > 60 {
		s = s[:57] + "..."
	}
	return s
}

const helpText = `commands:
  /system <text>  set system prompt (steers the model) and reset history
  /temp <f>       temperature (0 = greedy)      /topk <n>  top-k (0=off)
  /topp <f>       nucleus (0=off)               /seed <n>  RNG seed
  /max <n>        max tokens per reply          /json      toggle JSON-only output
  /reset          clear conversation history    /params    show settings
  /demos          list canned demo prompts      /demo <n>  run demo n (or by name)
  /help           this list                     /quit      exit
`
