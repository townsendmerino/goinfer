package serveapp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/codec"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
	"github.com/townsendmerino/goinfer/internal/runtime/scheduler"
	"github.com/townsendmerino/goinfer/pull"
)

// loadAdapters registers each --adapter against its base --model: it shares the base's resident decoder.Model (the RAM
// win: only the low-rank A/B bytes are new) but gets its own served name, KV-session LRU, decode mutex, and queue. A
// request routes to the fine-tune via the OpenAI `model` field; the per-LRU adapter binding makes each session project
// through it. The shared all-resident weights are read-only during a forward (per-stream scratch + KV), so the base and
// its adapters run as independent decode workers safely; --stream-weights (mutable per-layer paging on the shared model)
// is therefore rejected.
func (s *server) loadAdapters(cfg config) error {
	for _, spec := range cfg.adapters {
		if cfg.load.StreamWeights {
			return fmt.Errorf("--adapter %q: compute-time LoRA is incompatible with --stream-weights", spec.name)
		}
		base, ok := s.models[spec.base]
		if !ok {
			return fmt.Errorf("--adapter %q: base model %q not loaded (declare it with --model %s=…)", spec.name, spec.base, spec.base)
		}
		if base.adapter != "" {
			return fmt.Errorf("--adapter %q: base %q is itself an adapter — attach to the underlying --model", spec.name, spec.base)
		}
		if _, dup := s.models[spec.name]; dup {
			return fmt.Errorf("--adapter %q: name collides with a loaded model", spec.name)
		}
		t0 := time.Now()
		if err := base.model.LoadAdapter(spec.name, spec.dir); err != nil {
			return fmt.Errorf("--adapter %q: %w", spec.name, err)
		}
		fp := base.fp + "+adapter:" + spec.name // distinct snapshot namespace from the base + sibling adapters
		lm := &loadedModel{
			tk: base.tk, model: base.model, closer: base.closer, tmpl: base.tmpl, stopIDs: base.stopIDs,
			eosIDs: base.eosIDs, vocab: base.vocab, name: spec.name, fp: fp, adapter: spec.name, sampling: base.sampling,
			spec:         cfg.spec == "ngram",
			specAdaptive: cfg.specAdaptive,
			sessions:     newSessionLRU(base.model, cfg.kvSessions, 0, fp),
			caches:       modelCaches{tokenBytesFrom: base},
		}
		lm.sessions.adapter = spec.name
		if cfg.maxQueue > 0 {
			lm.queue = scheduler.NewQueue(1, cfg.maxQueue)
		}
		s.models[spec.name] = lm
		s.retainLocked(lm.livenessKey()) // adapter shares base.model → same liveness entry, refs++
		fmt.Fprintf(os.Stderr, "loaded adapter %q on base %q in %s\n", spec.name, spec.base, time.Since(t0).Round(time.Millisecond))
	}
	return nil
}

// splitShardSuffix is a split GGUF's first-shard suffix, "-00001-of-00004.gguf".
var splitShardSuffix = regexp.MustCompile(`(?i)-00001-of-\d{5}\.gguf$`)

// servedNameFor is the default served name for a model path: its base name without ".gguf", and for a split GGUF
// (named by its first shard) without the "-00001-of-NNNNN" too, so a split model is served as the model it is.
// Only ".gguf" is cut: a checkpoint directory named "Qwen2.5-0.5B-Instruct" has no extension, and filepath.Ext would
// take ".5B-Instruct" for one.
func servedNameFor(path string) string {
	name := filepath.Base(path)
	if loc := splitShardSuffix.FindStringIndex(name); loc != nil {
		return name[:loc[0]]
	}
	if strings.EqualFold(filepath.Ext(name), ".gguf") {
		name = name[:len(name)-len(".gguf")]
	}
	return name
}

// loadDecoder loads one generative model + tokenizer, resolves its chat template,
// and returns it as a *loadedModel. The served name is the spec's name=, else (a
// single unnamed --model) --served-model-name, else the file/dir basename.
func loadDecoder(ctx context.Context, spec modelSpec, cfg config) (*loadedModel, error) {
	// Resolve this model's knobs (per-model overrides over server-global defaults)
	// and reject invalid enums up front.
	opts := spec.options(&cfg)
	head, err := specHead(spec, &opts)
	if err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}
	// A model directory that carries joint_head.safetensors is a Clef decision model (Route C): its backbone is a plain qwen3_5 with the adapters merged, and
	// the head is loaded and checked here so a bad one stops startup, not the first request.
	var clefHead *clef.Head
	if isClefDir(spec.path) {
		if spec.head != nil {
			return nil, fmt.Errorf("--model %q carries joint_head.safetensors (a Clef model) and also head=%s (a JEV head): an entry is one or the other", spec.path, *spec.head)
		}
		if err := clefQuantRefusal(opts.Quant); err != nil {
			return nil, fmt.Errorf("--model %q: %w", spec.path, err)
		}
		if clefHead, err = clef.LoadHead(spec.path); err != nil {
			return nil, fmt.Errorf("--model %q: %w", spec.path, err)
		}
	}
	if err := opts.Validate(); err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}

	// --drafter attaches AFTER this model's own residency is built (attachBlockDrafter), but the elastic terms BuildResident
	// sizes against live free VRAM (CUDA's expert-cache slots, its unpinned ctx-by-default) cannot know it is coming unless
	// something prices it first. So the drafter is opened HERE, before decoder.Load, and Options.ExtraResidentBytes carries
	// its size into BuildResident: without that a 26B auto-sizes its expert cache to every free byte and the later attach
	// fails with nowhere left to go. Opened, not loaded (OpenDFlashDrafter): its header
	// prices it, and the attach streams its matrices from the file, so the host never holds the drafter beside the target's
	// load. Opened once and reused at the attach call site, so pricing and the actual attach see the same weights.
	var drafter *decoder.DFlashDrafter
	if cfg.drafter != "" {
		var derr error
		if drafter, derr = decoder.OpenDFlashDrafter(cfg.drafter); derr != nil {
			return nil, fmt.Errorf("--drafter %q: load drafter: %w", cfg.drafter, derr)
		}
		opts.ExtraResidentBytes = decoder.DrafterResidentBytesEstimate(drafter)
		// The drafter's own device K/V scales with whatever resident context the target ends up choosing, which is not known yet:
		// see Options.ExtraResidentKVPerPosition for why this is a rate, not a total, and who multiplies it by what.
		opts.ExtraResidentKVPerPosition = decoder.DrafterKVBytesPerPosition(drafter)
		opts.Drafter = true // decoder.ResolveKV: auto is f32 on CUDA beside a drafter, and an explicit f16 refuses
	}
	// A CUDA vision tower loads after this model and claims VRAM too (tower_reserve.go): price it the same way, so the KV plan leaves room for it.
	opts.ExtraResidentBytes += towerReserve(cfg, spec.path)

	// Resolve, sidecar / streaming transcode, tokenizer, swap-guarded load, the one automatic
	// streaming retry on a dense fit decline, and the .giw quant check: the path chat and fit share
	// (internal/modelload). The served name and fingerprint derive from the resolved SOURCE, never
	// the cache it may have loaded through.
	res, err := modelload.Load(ctx, modelload.Request{
		Spec: spec.path, Opts: opts, DirectLoad: cfg.load.DirectLoad,
		ExplicitQuant: spec.explicitQuant(&cfg), GuardAdvice: "use -stream-weights or a smaller quant",
	})
	if err != nil {
		return nil, err
	}
	spec.path, opts = res.Source, res.Opts
	tk, model := res.Tokenizer, res.Model
	t0 := time.Now().Add(-res.LoadTime) // the banner's "in X": the model load plus the setup below, as before
	mcfg := model.Config()
	name := spec.name
	if name == "" {
		if len(cfg.models) == 1 && cfg.name != "" {
			name = cfg.name
		} else {
			name = strings.TrimSuffix(filepath.Base(spec.path), ".gguf")
			if loc := splitShardSuffix.FindStringIndex(filepath.Base(spec.path)); loc != nil {
				name = filepath.Base(spec.path)[:loc[0]] // a split model is served as the model, not as its first shard
			}
		}
	}
	fp := modelFingerprint(spec.path, model.Quant())
	lm := &loadedModel{
		tk: tk, model: model, vocab: mcfg.VocabSize, eosIDs: mcfg.EOSIDs(), name: name, fp: fp, decisions: decisionState{head: head}, source: spec.path,
		sampling:     mcfg.Sampling,
		spec:         cfg.spec == "ngram",
		specAdaptive: cfg.specAdaptive,
		// capHint 0: KV grows on demand. The fingerprint binds disk snapshots to
		// this exact model+quant so a -session-dir reused across models is rejected.
		sessions: newSessionLRU(model, cfg.kvSessions, 0, fp),
	}
	// --drafter: attach a pretrained block drafter ONCE, here, on the BASE model only. Adapter-bearing requests route down
	// the session path, where the block-spec branch does not run, so attaching one there would upload ~500 MB and never be
	// used.
	// It fails startup rather than degrading silently: an operator who passed --drafter wants block drafting, and a wrong
	// pairing or an incapable backend should be one startup error they see, not a fleet quietly serving at 1x.
	if drafter != nil {
		if err := attachBlockDrafter(lm, drafter, cfg.drafterWidth); err != nil {
			return nil, fmt.Errorf("--drafter %q: %w", cfg.drafter, err)
		}
	}
	// --spec ngram: the same startup-refusal rule, for the same reason. The resident branch in
	// openai.go treats a spec error as "fall back to plain Generate" PER REQUEST, which is right for
	// a sampler the spec path does not support and wrong for a load-time property: it would leave an
	// operator who asked for speculation serving every request at 1x with no signal. --drafter needs
	// no separate check here — NewBlockSpec refuses inside attachBlockDrafter above.
	if lm.spec {
		if err := model.SpecDecodeConflict(); err != nil {
			return nil, fmt.Errorf("--spec ngram: %w", err)
		}
	}
	if clefHead != nil {
		if tk == nil {
			return nil, fmt.Errorf("--model %q is a Clef model but has no tokenizer.json: its record encoder tokenizes the schema fragment by fragment", spec.path)
		}
		if lm.decisions.clef, err = clef.NewModel(model, clefHead, clef.TokenizerFunc(tk), 0); err != nil {
			return nil, fmt.Errorf("--model %q: %w", spec.path, err)
		}
	}
	if cfg.maxQueue > 0 {
		lm.queue = scheduler.NewQueue(1, cfg.maxQueue)
	}
	if tmpl, derr := codec.Template(tk); derr == nil {
		lm.tmpl = cfg.tuneTemplate(tmpl)
		for _, str := range tmpl.Stops().Strings {
			if id, ok := tk.TokenID(str); ok {
				lm.stopIDs = append(lm.stopIDs, id)
			}
		}
	}
	warnThinkingTemplate(lm)
	fmt.Fprintf(os.Stderr, "loaded %q: %d-layer model (vocab %d) in %s [chat: %s]\n",
		name, mcfg.NumLayers, mcfg.VocabSize, time.Since(t0).Round(time.Millisecond), templateName(lm.tmpl))
	// State the RESOLVED paths, not the requested ones. The resident decode path and the batched prefill are optional
	// capabilities that fall back silently: a model can load clean, report a GPU backend, and still take one forward per
	// prompt token. Printing them at load makes that visible; -require-backend turns a decline into a startup failure.
	batched, why := lm.model.PrefillPath()
	// The rest of the resolved state — context cap, KV precision, session reuse (and WHY when
	// it is off), and the features a harness asks about — comes from modelBanner so a test can
	// hold it to the runtime's own state rather than trusting a run of Fprintf calls.
	bannerCfg := cfg
	bannerCfg.load.Ctx = opts.ResidentContext // the -ctx this model actually asked for (a per-model ctx= wins)
	for _, line := range modelBanner(lm, bannerCfg) {
		fmt.Fprintf(os.Stderr, "  %s\n", line)
	}
	if lm.decisions.clef != nil {
		fmt.Fprintf(os.Stderr, "  decisions: route C (clef), joint head %d-wide (CPU, f32), backbone through PromptHiddenAll: on the device when the model is resident on a backend that implements it (CUDA), the CPU otherwise, POST /v1/systemone\n", clefHead.Cfg.Width)
	}
	if h := lm.decisions.head; h != nil {
		how := "weights merged"
		if h.AdapterDir() != "" {
			how = "adapter merged at load; this entry's text generation is the decision model's, not the base's"
		}
		fmt.Fprintf(os.Stderr, "  decisions: route B, head %s %s (%s), POST /v1/systemone\n", h.Name, h.Version, how)
	}
	// The same reasoning applied to the TOKENIZER: a pre-tokenizer this build does not walk produces a different id stream
	// from HF and from llama.cpp with no error anywhere (count_tokens and usage drift by the same amount). Say it at load,
	// where the other silent fallbacks are already said.
	if lm.tk != nil {
		if d := lm.tk.PreTokenizerDecline(); d != "" {
			fmt.Fprintf(os.Stderr, "  !! tokenizer: %s\n", d)
		}
	}
	if cfg.requireBE {
		if err := requireFastPaths(name, cfg, lm, batched, why); err != nil {
			return nil, err
		}
	}
	return lm, nil
}

// requireFastPaths implements -require-backend for one loaded model: it fails the LOAD (so serve
// exits non-zero before binding a port) when the model didn't get the requested backend's fast
// paths. Two independent declines are checked, because either one is a large, silent regression:
// a GPU backend that produced no resident decode path (the whole forward is staged/CPU), and a
// prefill that declined to the sequential per-token loop. The CPU backend has no resident path by
// definition, so only the prefill half applies there.
func requireFastPaths(name string, cfg config, lm *loadedModel, batched bool, why string) error {
	if cfg.load.Backend != "cpu" && !lm.model.ResidentActive() {
		// Covers the whole ladder, not just an ineligible arch: an untagged build (`--backend cuda`
		// without `-tags cuda` resolves to the CPU backend), a box with no usable device (the cuda
		// factory always succeeds — the driver is only touched at BuildResident), and a model shape
		// the runner refuses. All three otherwise present as a healthy server that is silently on CPU.
		reason := lm.model.ResidentDecline()
		if reason == "" {
			reason = "no reason recorded"
		}
		return fmt.Errorf("--require-backend: model %q did not build a resident decode path on backend %q: %s (resolved: %s)",
			name, cfg.load.Backend, reason, lm.model.DecodePath())
	}
	if !batched {
		return fmt.Errorf("--require-backend: model %q declined the batched prefill: %s", name, why)
	}
	return nil
}

// requireAutoBackend is -require-backend's check on -backend auto, made at startup before any model loads. When auto
// passed over a GPU backend this binary links (no device answered, it is untested on this machine, or it is webgpu), it
// would run on the CPU, which is the silent fallback strict mode exists to stop, so it refuses and says how to choose.
// A binary with no GPU backend has only the CPU, so auto there is the CPU, and the per-model checks run as for
// -backend cpu.
func requireAutoBackend(cfg config) error {
	if !cfg.requireBE || cfg.load.Auto == nil || !cfg.load.Auto.Skipped {
		return nil
	}
	return fmt.Errorf("--require-backend: -backend auto would run on the CPU (%s); pass -backend cpu to run strict on the CPU, or name a GPU backend to require it",
		cfg.load.Auto.Reason)
}

// embedEncoderLoads is the plan check for an -embed-model safetensors checkpoint. loadEncoder's directory path is aikit's
// encoder.Load, which loads a NomicBert (CodeRankEmbed, nomic-embed-text) and checks no model_type itself, so anything
// else would fail late, or load a foreign checkpoint's tensors under the wrong architecture. It is refused here instead,
// after config.json and before any weight. A decoder used as an embedder comes as a GGUF (hf:<repo>:<quant>), which is
// not a checkpoint plan at all.
func embedEncoderLoads(modelType string) (string, error) {
	if fam, err := pull.EncoderLoads(modelType); err == nil {
		return fam, nil
	}
	return "", fmt.Errorf("model_type %q, which -embed-model's encoders do not load (they load nomic_bert: CodeRankEmbed, nomic-embed-text; and embedding_gemma2: google/embeddinggemma-2; a decoder used as an embedder comes as a GGUF, hf:<owner>/<repo>:<quant>)", modelType)
}

// resolveEmbedModel turns -embed-model into a local path: a plain path unchanged, hf:<repo>:safetensors fetched as an
// encoder checkpoint under embedEncoderLoads, any other hf: reference resolved as a GGUF. A seam, so a test can stand in
// the fetch.
var resolveEmbedModel = func(ctx context.Context, spec string) (string, error) {
	return pull.ResolveCheckpointFor(ctx, spec, embedEncoderLoads)
}

// loadEncoder loads the embedding model (f32 or int8) plus its tokenizer (used
// only to count tokens for usage.prompt_tokens — counting needs no forward pass,
// so it is cheaper than running EncodeTokensWithIDs, and works for both
// precisions, whereas that method is f32-only).
func (s *server) loadEncoder(cfg config) error {
	// A causal decoder used as an embedder (qwen3-embedding / embeddinggemma) arrives as a .gguf
	// FILE; the aikit encoder path takes an HF directory. Dispatch on that rather than adding
	// another flag —
	if fi, statErr := os.Stat(cfg.embedPath); statErr == nil && !fi.IsDir() {
		return s.loadDecoderEmbedder(cfg)
	}
	// EmbeddingGemma 2 is a directory too, but aikit's encoder.Load reads it as a NomicBert and would fail late; it
	// has its own loader (embeddinggemma2_embedder.go).
	if isEmbeddingGemma2(cfg.embedPath) {
		return s.loadEmbeddingGemma2(cfg)
	}
	t0 := time.Now()
	var (
		enc  encoder.Encoder
		err  error
		prec string
	)
	switch strings.ToLower(cfg.embedQuant) {
	case "q8", "int8":
		enc, err = encoder.LoadQ8(cfg.embedPath)
		prec = "int8"
	case "", "f32":
		enc, err = encoder.Load(cfg.embedPath)
		prec = "f32"
	default:
		return fmt.Errorf("invalid -embed-quant %q (want f32 | q8)", cfg.embedQuant)
	}
	if err != nil {
		return fmt.Errorf("load embedding model: %w", err)
	}
	tok, err := embed.LoadTokenizer(filepath.Join(cfg.embedPath, "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("load embedding tokenizer: %w", err)
	}
	name := cfg.embedName
	if name == "" {
		name = filepath.Base(strings.TrimRight(cfg.embedPath, "/"))
	}
	s.embed, s.embedTok, s.embedID, s.embedDim = enc, tok, name, enc.HiddenDim()
	// Matryoshka floor from aikit's exported registry — the same source of truth behind its
	// published Truncatable column. Unknown/non-MRL models get 0, i.e. `dimensions` is refused
	// rather than honored into a silently worse-retrieving vector. Keyed off the model PATH (the
	// directory name is the HF model name); embedName is an operator-chosen alias, so it must not
	// decide this.
	s.embedMRLMin, _ = encoder.MatryoshkaFloor(cfg.embedPath)
	trunc := "not truncatable"
	if s.embedMRLMin > 0 {
		trunc = fmt.Sprintf("truncatable to %d", s.embedMRLMin)
	}
	fmt.Fprintf(os.Stderr, "loaded embedding model %q (dim %d, %s, %s) in %s\n",
		name, s.embedDim, prec, trunc, time.Since(t0).Round(time.Millisecond))
	return nil
}

// sessionSubdir gives a model its own --session-dir folder so warm-KV snapshots
// from different models don't collide (the dir name is a short hash of the
// fingerprint; the snapshot's own identity guard still rejects a mismatch).
func sessionSubdir(base, fp string) string {
	h := sha256.Sum256([]byte(fp))
	return filepath.Join(base, hex.EncodeToString(h[:8]))
}

// modelFingerprint identifies the loaded model for binding KV snapshots to it:
// the checkpoint's basename + size + mtime + resident quant. Two different
// models (or the same weights at a different quant — whose KV is incompatible)
// produce different fingerprints, so a -session-dir reused across them is
// rejected on load rather than fed stale KV. A missing/unstattable path degrades
// to name+quant (still distinguishes different files by name).
func modelFingerprint(path, quant string) string {
	base := filepath.Base(path)
	if fi, err := os.Stat(path); err == nil {
		return fmt.Sprintf("%s|%d|%d|%s", base, fi.Size(), fi.ModTime().UnixNano(), quant)
	}
	return fmt.Sprintf("%s|%s", base, quant)
}

func templateName(t *chat.Template) string {
	if t == nil {
		return "raw (no template)"
	}
	return t.Name() + thinkingNote(t) + toolFormatNote(t)
}

// toolFormatNote says, for a model whose template declares a native tool form, which prompt serve renders tools with and how to get the
// other — the same reason thinkingNote exists: an operator should not have to find out from a model's behaviour.
func toolFormatNote(t *chat.Template) string {
	if !t.DeclaresNativeTools() {
		return ""
	}
	if t.UsesNativeTools() {
		return ", tools: the model's own template form (-tool-format hermes for goinfer's)"
	}
	return ", tools: goinfer's own form (-tool-format template for the model's own)"
}

// thinkingNote is the load log's statement of what serve will do about thinking for this model: the checkpoint's own
// default (read from its template), the mode -thinking selected, or "unmanaged" — a ChatML/Gemma template whose thinking
// control was not recognised is served exactly as before, and saying so beats leaving an operator to find out.
func thinkingNote(t *chat.Template) string {
	r := t.Reasoning()
	if r == nil {
		if t.Name() == "harmony" { // gpt-oss: always reasons (no off form in its prompt), and the reply is channel messages
			return ", reasoning: analysis channel split from the answer (gpt-oss always reasons; -thinking cannot turn it off)"
		}
		if t.Name() == "chatml" || t.Name() == "gemma4" {
			return ", thinking: unmanaged"
		}
		return ""
	}
	def := map[bool]string{true: "on", false: "off"}[r.DefaultOn()]
	return fmt.Sprintf(", thinking: template default %s, serving %s", def, t.ThinkMode())
}
