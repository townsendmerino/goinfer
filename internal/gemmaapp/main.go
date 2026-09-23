// Command gemma is a demo CLI that runs a local decoder-only LLM through
// aikit's pure-Go decoder and streams the completion to stdout. Despite the
// name it is multi-model: any family the decoder supports (Gemma 3, Qwen,
// Llama, Mistral, GPT-2, Mixtral) loads from an HF checkpoint dir, and a bare
// quantized .gguf file loads with no sidecar config or tokenizer.
//
// It loads the weights, tokenizes the prompt, and streams a completion —
// greedy by default, or temperature / top-k / top-p sampling via the flags.
//
// Usage:
//
//	go run ./demo/gemma --model ~/models/gemma-3-270m --prompt "Hello, world"
//	go run ./demo/gemma --model ~/models/Qwen2.5-1.5B --prompt "..." --max 128 --temp 0.7
//	go run ./demo/gemma --model ~/models/tinyllama-1.1b-chat.Q4_K_M.gguf --prompt "The capital of France is"
//
// Get a checkpoint (HF layout: config.json + model.safetensors +
// tokenizer.json) or a single .gguf:
//
//	huggingface-cli download google/gemma-3-270m --local-dir ~/models/gemma-3-270m
//	huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ~/models
package gemmaapp

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func Main() {
	var (
		modelDir = flag.String("model", "", "path to a checkpoint dir (config.json + model.safetensors + tokenizer) or a bare .gguf file")
		prompt   = flag.String("prompt", "Hello, world", "prompt text")
		maxTok   = flag.Int("max", 128, "max tokens to generate")
		temp     = flag.Float64("temp", 0.0, "sampling temperature (0 = greedy)")
		topK     = flag.Int("top-k", 0, "top-k filter (0 = off)")
		topP     = flag.Float64("top-p", 0.0, "top-p / nucleus (0 = off)")
		seed     = flag.Int64("seed", 0, "sampling RNG seed")
		backend  = flag.String("backend", "cpu", "compute backend: cpu | webgpu | metal (metal needs -tags metal + --quant int8int8, darwin)")
		quant    = flag.String("quant", "", "weight quantization: \"\" (f32) | int8 | int8int8 | int4")
		jsonMode = flag.Bool("json", false, "constrain output to valid JSON (logit masking via constrain.JSON)")
		// M-26 (audit-2026-09-10): the same --exact-prefill goinfer-chat and serve have, wired
		// to the decoder.Options.ExactPrefill field that used to exist only in a task doc's claim.
		exactPrefill = flag.Bool("exact-prefill", false, "force BIT-EXACT prompt ingestion on every backend that has a faster, non-exact default (CPU f32-attention, Metal's f16-MMA batched prefill, CUDA's tensor-core batched prefill — all default ON above their own length thresholds)")
	)
	flag.Parse()

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required (path to a Gemma 3 checkpoint dir)")
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*modelDir, *prompt, *maxTok, *backend, *quant, *jsonMode, *exactPrefill, decoder.SamplingParams{
		Temperature: *temp,
		TopK:        *topK,
		TopP:        *topP,
		Seed:        *seed,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "\n%v\n", err)
		// A scaffold/NYI error is expected today — exit 1, not a crash.
		os.Exit(1)
	}
}

func run(modelDir, prompt string, maxTok int, backend, quant string, jsonMode, exactPrefill bool, sp decoder.SamplingParams) error {
	// 1) Tokenizer. A bare .gguf carries its tokenizer in metadata
	// (tokenizer.LoadGGUF); an HF checkpoint dir has a tokenizer.json
	// (tokenizer.Load).
	loadTok := tokenizer.Load
	if strings.HasSuffix(modelDir, ".gguf") {
		loadTok = tokenizer.LoadGGUF
	}
	tk, err := loadTok(modelDir)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}

	// 2) Model + backend.
	t0 := time.Now()
	model, err := decoder.Load(modelDir, decoder.Options{Backend: backend, Quant: quant, ExactPrefill: exactPrefill})
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	cfg := model.Config()
	// M-25 (audit-2026-09-10): DecodePath(), not the raw requested -quant flag — on Metal,
	// int8/int8int8/int4mix all silently re-quantize to int4 at resident-build time (no int8
	// GEMV kernel on that backend), so printing the flag verbatim named a precision the GPU never
	// actually runs. DecodePath() reports the corrected label (decoder/residency.go's
	// residentQuantLabel).
	fmt.Fprintf(os.Stderr, "loaded %d-layer model (hidden %d, vocab %d) in %s [backend=%s decode=%s]\n",
		cfg.NumLayers, cfg.HiddenDim, cfg.VocabSize, time.Since(t0).Round(time.Millisecond), model.BackendReport(), model.DecodePath())

	// 2b) Constrained decoding: mask every step's logits to the tokens a JSON
	// grammar permits, so the model physically cannot emit malformed JSON. The
	// vocab→bytes map comes from the tokenizer; EOS is gated until the document
	// is complete.
	if jsonMode {
		sp.LogitProcessor = jsonMasker(tk, cfg.VocabSize)
		fmt.Fprintln(os.Stderr, "constrained to valid JSON (constrain.JSON)")
	}

	// 3) Encode prompt.
	ids, err := tk.Encode(prompt, true /* addBOS */)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}

	// 4) Generate + stream. Print only the newly-completed bytes each step,
	// holding back any trailing incomplete UTF-8 — a byte-fallback token is a
	// single (possibly partial) byte, so a naive per-token DecodePiece would
	// emit broken multibyte characters mid-stream without the holdback.
	//
	// R-08 (audit-2026-09-02 P-17, task-recompute-audit.md): this used to
	// re-Decode the WHOLE running sequence (prompt + every generated token so
	// far) on every single token, an O(n^2) pattern in output length — the
	// same defect already fixed in internal/serveapp/openai.go's streamTokens
	// (2026-09-03) and this repo's other two demo CLIs, chatapp/agent.go
	// (2026-09-11, c0ab6ed3) — deferred here at the time because this loop
	// ALSO decodes the prompt through the same call, and needs the
	// SentencePiece leading-space strip to land once, at the sequence's true
	// start, which a per-token DecodePiece never applies (chatapp/agent avoid
	// this because they only ever decode the GENERATED continuation, never the
	// prompt, through their own streamGen). The fix: decode the prompt once,
	// with the one-time whole-sequence Decode this always needed anyway (the
	// strip lands correctly there, and it costs nothing extra — it already ran
	// once per request, not once per token), then accumulate every GENERATED
	// token's own DecodePiece onto a strings.Builder seeded with that text.
	// Concatenation is associative regardless of where the chunk boundary
	// falls (TestDecodeContinuation_isIncrementallyAssociative, tokenizer/),
	// so appending the prompt's own decode followed by each token's DecodePiece
	// is byte-identical to re-decoding the whole growing sequence every time —
	// strings.Builder, not `text += piece`: Go strings are immutable, so naive
	// concatenation is itself O(n) per append and would silently reintroduce
	// the O(n^2) this removes.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Seed the display sequence with the prompt ids, minus a leading BOS (which
	// would render as "<s>"/"<bos>"). The prompt then renders from the tokens,
	// so what's shown is exactly what the model sees.
	promptSeq := append([]int(nil), ids...)
	if len(promptSeq) > 0 && promptSeq[0] == tk.Special().BOS {
		promptSeq = promptSeq[1:]
	}

	stream, gen := model.Generate(ctx, ids, maxTok, sp)
	if _, _, err := streamGen(tk, promptSeq, stream, func(chunk string) {
		os.Stdout.Write([]byte(chunk))
	}); err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	fmt.Println()
	return gen.Err()
}

// streamGen decodes promptIDs once (applying the SentencePiece leading-space strip at the true
// sequence start — see run's own R-08 comment above for why that one-time call stays a whole-
// sequence Decode) then drains tokens into text incrementally, calling onChunk with each newly-
// completed span exactly as run's inline flush did before this was extracted. Returns the full
// text (prompt + generated) and the count of GENERATED tokens (promptIDs itself is not counted).
func streamGen(tk *tokenizer.Tokenizer, promptIDs []int, tokens <-chan int, onChunk func(string)) (text string, nTok int, err error) {
	promptText, derr := tk.Decode(promptIDs)
	if derr != nil {
		return "", 0, derr
	}
	var sb strings.Builder
	sb.WriteString(promptText)
	printed := 0
	flush := func(final bool) {
		txt := sb.String() // O(1): a view over the Builder's buffer, not a copy
		tail := txt[printed:]
		end := len(tail) + printed
		if !final {
			end = completeUTF8Len([]byte(tail)) + printed
		}
		if end > printed {
			if onChunk != nil {
				onChunk(txt[printed:end])
			}
			printed = end
		}
	}
	flush(false) // render the prompt
	for id := range tokens {
		nTok++
		// DecodePiece, not Decode: these ids CONTINUE the prompt already
		// rendered above (no sequence-level dummy-prefix strip — M-25), and
		// appending each token's own piece is exactly what a whole-sequence
		// decode does internally, one token at a time.
		piece, _ := tk.DecodePiece(id)
		sb.WriteString(piece)
		flush(false)
	}
	flush(true)
	return sb.String(), nTok, nil
}

// jsonMasker builds the JSON-constraining LogitProcessor: the vocab's surface
// bytes (tk.TokenText) feed a constrain.Masker over a JSON grammar, with the
// tokenizer's EOS / end-of-turn ids gated until the document is complete.
func jsonMasker(tk *tokenizer.Tokenizer, vocab int) func(generated []int, logits []float32) {
	var eos []int
	for _, id := range []int{tk.Special().EOS, tk.Special().EndOfTurn} {
		if id >= 0 {
			eos = append(eos, id)
		}
	}
	m := constrain.NewMasker(constrain.JSON(), constrain.TokenBytes(vocab, tk.TokenText), eos).StopWhenComplete()
	return m.Process
}

// completeUTF8Len returns the length of the longest prefix of b that ends on a
// complete UTF-8 rune boundary, so a partial trailing byte-fallback sequence is
// held back until the bytes that finish the character arrive.
func completeUTF8Len(b []byte) int {
	i := 0
	for i < len(b) {
		if b[i] < utf8.RuneSelf {
			i++
			continue
		}
		if !utf8.FullRune(b[i:]) {
			break // incomplete trailing sequence — hold back from here
		}
		_, size := utf8.DecodeRune(b[i:])
		i += size
	}
	return i
}
