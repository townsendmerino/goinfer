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
	)
	flag.Parse()

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required (path to a Gemma 3 checkpoint dir)")
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*modelDir, *prompt, *maxTok, *backend, *quant, *jsonMode, decoder.SamplingParams{
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

func run(modelDir, prompt string, maxTok int, backend, quant string, jsonMode bool, sp decoder.SamplingParams) error {
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
	model, err := decoder.Load(modelDir, decoder.Options{Backend: backend, Quant: quant})
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	cfg := model.Config()
	q := quant
	if q == "" {
		q = "f32"
	}
	fmt.Fprintf(os.Stderr, "loaded %d-layer model (hidden %d, vocab %d) in %s [backend=%s quant=%s]\n",
		cfg.NumLayers, cfg.HiddenDim, cfg.VocabSize, time.Since(t0).Round(time.Millisecond), model.BackendReport(), q)

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

	// 4) Generate + stream. Decode the whole running sequence (prompt +
	// generated) each step and print only the newly-completed bytes, holding
	// back any trailing incomplete UTF-8 — a byte-fallback token is a single
	// (possibly partial) byte, so per-token DecodePiece would emit broken
	// multibyte characters. Decoding the full sequence (not just the generated
	// tail) is what makes the SentencePiece leading-space strip land once at the
	// true start, so the space before the first generated word is preserved.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Seed the display sequence with the prompt ids, minus a leading BOS (which
	// would render as "<s>"/"<bos>"). The prompt then renders from the tokens,
	// so what's shown is exactly what the model sees.
	seq := append([]int(nil), ids...)
	if len(seq) > 0 && seq[0] == tk.Special().BOS {
		seq = seq[1:]
	}
	printed := 0
	flush := func(final bool) error {
		text, derr := tk.Decode(seq)
		if derr != nil {
			return derr
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
		return nil
	}
	if err := flush(false); err != nil { // render the prompt
		return fmt.Errorf("decode: %w", err)
	}

	stream, gen := model.Generate(ctx, ids, maxTok, sp)
	for id := range stream {
		seq = append(seq, id)
		if err := flush(false); err != nil {
			return fmt.Errorf("decode: %w", err)
		}
	}
	if err := flush(true); err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	fmt.Println()
	return gen.Err()
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
