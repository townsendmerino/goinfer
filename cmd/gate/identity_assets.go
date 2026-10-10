package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// `gate identity`'s ASSETS: which checkpoint and which prompt each parity-manifest family is run on.
//
// TINY (the default, by day) is the fixture the family's forward golden already loads, with that
// golden's own prompt ids — the family's parity prompt — so "identical here" is a statement about
// exactly the input the family's tiny validation used. REAL (-assets real) is a small table of
// checkpoints under ~/models with two fixed prose prompts (L1's), tokenized once by this build so both
// revisions see the same ids. The table is explicit on purpose: a family with no tiny fixture is
// listed with the reason, and TestIdentityAssets_coverTheManifest fails when a manifest family is in
// neither list — a new family cannot silently fall out of the check.

// identityAsset is one checkpoint a family is run on.
type identityAsset struct {
	Family string
	Name   string // short, unique within the family (appears in cell ids)
	Path   string // tiny: repo-relative; real: relative to the models dir
	Golden string // tiny: repo-relative golden JSON (.json or .json.gz) holding the prompt ids
	Tok    string // real: tokenizer source relative to the models dir (.gguf, or a dir with tokenizer.json)
	// Backends limits a backend-specific file (a .giw sidecar is built for one backend); nil = any.
	Backends []string
	// Skip, when set, is why this asset is never run by default (reported NOT RUN with the reason).
	Skip string
}

// identityTiny: the forward goldens' fixtures (decoder/*_test.go: *_forwardParity / *_textParity).
var identityTiny = []identityAsset{
	{Family: "bailing_hybrid", Name: "tiny", Path: "testdata/bailing_hybrid-tiny", Golden: "testdata/bailing_hybrid_forward_golden.json"},
	{Family: "cohere", Name: "tiny", Path: "testdata/cohere-tiny", Golden: "testdata/cohere_tiny_golden.json"},
	{Family: "cohere2", Name: "tiny", Path: "testdata/cohere2-tiny", Golden: "testdata/cohere2_tiny_golden.json"},
	{Family: "deepseek_v3", Name: "tiny", Path: "testdata/deepseek-tiny", Golden: "testdata/deepseek_tiny_text_golden.json"},
	{Family: "gemma3", Name: "vl-tiny-text", Path: "testdata/gemma3-vl-tiny", Golden: "testdata/gemma3_vl_tiny_text_golden.json"},
	{Family: "gemma4", Name: "moe-tiny", Path: "testdata/gemma4-moe-tiny", Golden: "testdata/gemma4_moe_forward_golden.json"},
	{Family: "gemma4", Name: "moe-unified-tiny", Path: "testdata/gemma4-moe-unified-tiny", Golden: "testdata/gemma4_moe_forward_golden.json"},
	{Family: "gemma4", Name: "dense-twogeom-tiny", Path: "testdata/gemma4-dense-twogeom-tiny", Golden: "testdata/gemma4_dense_twogeom_golden.json"},
	{Family: "gemma4", Name: "dense-scaled", Path: "testdata/gemma4-dense-scaled", Golden: "testdata/gemma4_dense_scaled_golden.json"},
	{Family: "gemma4", Name: "moe-kv-tiny", Path: "testdata/gemma4-moe-kv-tiny", Golden: "testdata/gemma4_moe_kv_forward_golden.json"},
	{Family: "glm4_moe", Name: "tiny", Path: "testdata/glm-tiny", Golden: "testdata/glm_tiny_text_golden.json"},
	{Family: "glm4_moe", Name: "tiny-bias", Path: "testdata/glm-tiny-bias", Golden: "testdata/glm_tiny_bias_text_golden.json"},
	{Family: "glm_ocr", Name: "tiny", Path: "testdata/glm-ocr-tiny", Golden: "testdata/glm_ocr_tiny_golden.json"},
	{Family: "gpt-oss", Name: "tiny-gguf", Path: "decoder/testdata/gptoss_tiny.gguf", Golden: "decoder/testdata/gptoss_tiny_golden.json"},
	{Family: "gpt2", Name: "124m", Path: "testdata/gpt2", Golden: "testdata/gpt2_forward_golden.json"},
	{Family: "granite", Name: "dense-tiny", Path: "testdata/granite-dense-tiny", Golden: "testdata/granite_dense_forward_golden.json"},
	{Family: "granitemoehybrid", Name: "tiny", Path: "testdata/granite-tiny", Golden: "testdata/granite_tiny_text_golden.json"},
	{Family: "internlm2", Name: "tiny", Path: "decoder/testdata/internlm2-tiny", Golden: "decoder/testdata/internlm2_tiny_text_golden.json"},
	{Family: "kimi_k2", Name: "tiny", Path: "testdata/kimi-tiny", Golden: "testdata/kimi_tiny_text_golden.json"},
	{Family: "laguna", Name: "m1-tiny", Path: "decoder/testdata/laguna-m1-tiny", Golden: "decoder/testdata/laguna_m1_tiny_text_golden.json"},
	{Family: "laguna", Name: "xs2-tiny", Path: "decoder/testdata/laguna-xs2-tiny", Golden: "decoder/testdata/laguna_xs2_tiny_text_golden.json"},
	{Family: "laguna", Name: "xs21-tiny", Path: "decoder/testdata/laguna-xs21-tiny", Golden: "decoder/testdata/laguna_xs21_tiny_text_golden.json"},
	{Family: "lfm2", Name: "tiny", Path: "testdata/lfm2-tiny", Golden: "testdata/lfm2_tiny_text_golden.json"},
	{Family: "llama", Name: "tiny", Path: "testdata/llama-tiny", Golden: "testdata/llama_tiny_text_golden.json"},
	{Family: "llama4_text", Name: "tiny", Path: "testdata/llama4-tiny", Golden: "testdata/llama4_tiny_text_golden.json"},
	{Family: "mellum", Name: "mellum2-slice", Path: "decoder/testdata/mellum-mellum2-slice", Golden: "decoder/testdata/mellum_mellum2_slice_golden.json",
		Skip: "its fixture is a 4.2 GB slice of the real model, not a tiny checkpoint"},
	{Family: "mistral", Name: "tiny-window", Path: "testdata/mistral-tiny-window", Golden: "testdata/mistral_tiny_window_golden.json"},
	{Family: "mistral3", Name: "ministral3-tiny", Path: "testdata/ministral3-tiny", Golden: "testdata/ministral3_forward_golden.json"},
	{Family: "mixtral", Name: "tiny", Path: "testdata/mixtral-tiny", Golden: "testdata/mixtral_forward_golden.json"},
	{Family: "nemotron_h", Name: "tiny", Path: "testdata/nemotron-tiny", Golden: "testdata/nemotron_tiny_text_golden.json"},
	{Family: "nemotron_h", Name: "3nano-tiny", Path: "testdata/nemotron3nano-tiny", Golden: "testdata/nemotron3nano_tiny_text_golden.json"},
	{Family: "olmo3", Name: "tiny", Path: "testdata/olmo3-tiny", Golden: "testdata/olmo3_forward_golden.json"},
	{Family: "gemma", Name: "gemma1-tiny", Path: "testdata/gemma1-tiny", Golden: "testdata/gemma1_forward_golden.json"},
	{Family: "gemma2", Name: "tiny", Path: "testdata/gemma2-tiny", Golden: "testdata/gemma2_forward_golden.json"},
	{Family: "olmo_hybrid", Name: "tiny", Path: "testdata/olmo_hybrid-tiny", Golden: "testdata/olmo_hybrid_forward_golden.json"},
	{Family: "phi3", Name: "tiny", Path: "testdata/phi3-tiny", Golden: "testdata/phi3_tiny_text_golden.json"},
	{Family: "qwen2", Name: "0.5b", Path: "testdata/qwen2.5-0.5b", Golden: "testdata/qwen2_forward_golden.json"},
	{Family: "qwen2_5_vl", Name: "tiny-text", Path: "testdata/qwen25vl-tiny", Golden: "testdata/qwen25vl_tiny_text_golden.json"},
	{Family: "qwen2_moe", Name: "tiny", Path: "testdata/tiny-qwen2-moe", Golden: "testdata/qwen2moe_forward_golden.json"},
	{Family: "qwen3", Name: "1.7b", Path: "testdata/qwen3-1.7b", Golden: "testdata/qwen3_forward_golden.json"},
	{Family: "qwen3_5", Name: "tiny", Path: "decoder/testdata/qwen3_5-tiny", Golden: "decoder/testdata/qwen3_5_tiny_text_golden.json"},
	{Family: "qwen3_5_moe", Name: "tiny", Path: "decoder/testdata/qwen3_5_moe-tiny", Golden: "decoder/testdata/qwen3_5_moe_tiny_text_golden.json"},
	{Family: "qwen3_5_moe", Name: "qwen35-tiny", Path: "testdata/qwen35-tiny", Golden: "testdata/qwen35_forward_golden.json"},
	{Family: "qwen3_moe", Name: "tiny", Path: "testdata/qwen3moe-tiny", Golden: "testdata/qwen3moe_forward_golden.json"},
	{Family: "qwen3_moe", Name: "tiny-k3", Path: "testdata/qwen3moe-tiny-k3", Golden: "testdata/qwen3moe_k3_forward_golden.json"},
	{Family: "qwen3_next", Name: "tiny", Path: "testdata/qwen3next-tiny", Golden: "testdata/qwen3next_tiny_text_golden.json"},
	{Family: "qwen3_vl", Name: "tiny-text", Path: "testdata/qwen3vl-tiny", Golden: "testdata/qwen3vl_tiny_text_golden.json"},
	{Family: "qwen3_vl_moe", Name: "tiny", Path: "testdata/qwen3vlmoe-tiny", Golden: "testdata/qwen3vlmoe_tiny_golden.json"},
	{Family: "smollm3", Name: "tiny", Path: "testdata/smollm3-tiny", Golden: "testdata/smollm3_forward_golden.json"},
	{Family: "spark2_5", Name: "tiny", Path: "decoder/testdata/spark2-5-tiny", Golden: "decoder/testdata/spark2_5_tiny_text_golden.json"},
}

// identityNoTiny: manifest families with no tiny fixture of their own, and why.
var identityNoTiny = map[string]string{
	"deepseek_v2": "no deepseek_v2 tiny fixture (deepseek-tiny is model_type deepseek_v3; v2's gate is the real V2-Lite)",
	"qwen3_asr":   "its tiny checkpoint (testdata/qwen3asr-tiny) keeps its golden as golden.zip, the text path AND the audio composition, not the JSON text golden this tool reads; its gates are decoder TestQwen3ASR_tiny*",
	"voxtral":     "its tiny checkpoint (testdata/voxtral-tiny) keeps its golden as golden.zip, the text path AND the audio composition, not the JSON text golden this tool reads; its gates are decoder TestVoxtral_tiny*",
}

// identityReal: small real checkpoints under the models dir (~/models). Never /Volumes or /srv/models:
// see CLAUDE.md § Models — and identityPathAllowed, which refuses them.
var identityReal = []identityAsset{
	{Family: "qwen2", Name: "coder-0.5b-q4km", Path: "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", Tok: "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
	{Family: "qwen2", Name: "coder-1.5b-q4km", Path: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", Tok: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
	{Family: "qwen2", Name: "7b-cpu-giw", Path: "qwen2.5-7b-instruct-q4_k_m.int4.cpu-arm64.giw", Tok: "qwen2.5-7b-instruct-q4_k_m.gguf", Backends: []string{"cpu"}},
	{Family: "qwen2", Name: "7b-metal-giw", Path: "qwen2.5-7b-instruct-q4_k_m.int4.metal.giw", Tok: "qwen2.5-7b-instruct-q4_k_m.gguf", Backends: []string{"metal"}},
	{Family: "qwen3", Name: "0.6b-bf16", Path: "qwen3-0.6b-bf16", Tok: "qwen3-0.6b-bf16"},
	{Family: "lfm2", Name: "2.6b", Path: "lfm2.5-2.6b", Tok: "lfm2.5-2.6b"},
	{Family: "internlm2", Name: "1.8b", Path: "internlm2-1_8b", Tok: "internlm2-1_8b"},
	{Family: "spark2_5", Name: "1.7b", Path: "spark25-1.7b", Tok: "spark25-1.7b"},
}

// identityRealPrompts are L1's two prose prompts (docs/measurements/cpu-decode-peer-gap-2026-09-27/
// f16xbuild-mac.go.txt): 145 and 621 tokens on the qwen2.5 tokenizer.
func identityRealPrompts() []string {
	base := "Rivers shape the land they cross. Over thousands of years a river carves valleys, deposits silt on its floodplain, and builds deltas where it meets the sea. "
	var out []string
	for _, reps := range []int{4, 18} {
		out = append(out, strings.Repeat(base, reps)+"Explain how rivers influenced early civilizations.")
	}
	return out
}

// manifestFamilies reads the family names and the fields the report prints from the parity manifest.
type manifestFamily struct {
	Status      string          `json:"status"`
	Method      string          `json:"method"`
	ValidatedAt string          `json:"validated_at"`
	Machine     string          `json:"machine"`
	Reference   string          `json:"reference"`
	Metrics     json.RawMessage `json:"metrics"`
}

func readManifestFamilies(root string) (map[string]manifestFamily, error) {
	b, err := os.ReadFile(filepath.Join(root, "testdata", "parity_manifest.json"))
	if err != nil {
		return nil, err
	}
	var m struct {
		Families map[string]manifestFamily `json:"families"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("parity manifest: %w", err)
	}
	return m.Families, nil
}

// readGoldenPrompt returns a golden's prompt ids. The goldens name them three ways: prompt_ids (the
// tiny text goldens), ids (the HF forward goldens), input_ids (gpt-oss). A .gz suffix is gunzipped.
func readGoldenPrompt(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		r = gz
	}
	var g struct {
		PromptIDs []int `json:"prompt_ids"`
		IDs       []int `json:"ids"`
		InputIDs  []int `json:"input_ids"`
	}
	if err := json.NewDecoder(r).Decode(&g); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for _, ids := range [][]int{g.PromptIDs, g.IDs, g.InputIDs} {
		if len(ids) > 0 {
			return ids, nil
		}
	}
	return nil, fmt.Errorf("%s: no prompt_ids / ids / input_ids", path)
}

// maxPositions reads a checkpoint dir's context length from config.json (0 = unknown).
func maxPositions(dir string) int {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return 0
	}
	var c struct {
		MaxPos     int `json:"max_position_embeddings"`
		NPositions int `json:"n_positions"`
		Text       struct {
			MaxPos int `json:"max_position_embeddings"`
		} `json:"text_config"`
	}
	if json.Unmarshal(b, &c) != nil {
		return 0
	}
	for _, v := range []int{c.MaxPos, c.Text.MaxPos, c.NPositions} {
		if v > 0 {
			return v
		}
	}
	return 0
}

// longPrompt cycles the parity prompt out to a longer one, so a prompt past the batched-prefill
// threshold is covered as well as the golden's short one (L1's dumps used 145 and 621 tokens). It
// stays inside the checkpoint's context with room for the decode steps; 0 when there is no room.
func longPrompt(ids []int, maxPos, steps int) []int {
	const want = 160
	n := want
	if maxPos > 0 && maxPos-steps-2 < n {
		n = maxPos - steps - 2
	}
	if n < len(ids)+16 {
		return nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = ids[i%len(ids)]
	}
	return out
}

// assetPresent reports why an asset cannot be loaded here, or "" when it can.
func assetPresent(p string) string {
	fi, err := os.Stat(p)
	if err != nil {
		return "asset missing: " + p
	}
	if !fi.IsDir() {
		return ""
	}
	if _, err := os.Stat(filepath.Join(p, "config.json")); err != nil {
		return "asset incomplete (no config.json): " + p
	}
	for _, w := range []string{"model.safetensors", "model.safetensors.index.json"} {
		if _, err := os.Stat(filepath.Join(p, w)); err == nil {
			return ""
		}
	}
	return "asset incomplete (no model.safetensors): " + p
}

// identityPathAllowed refuses the model archive as a read path: CLAUDE.md § Models — /srv/models (the
// nobara SMR disk) and /Volumes/ (its SMB mount here) are storage, never a read path. A symlink into
// either is followed and refused too.
func identityPathAllowed(p string) error {
	real := p
	if r, err := filepath.EvalSymlinks(p); err == nil {
		real = r
	}
	for _, bad := range []string{"/Volumes/", "/srv/models"} {
		if strings.HasPrefix(p, bad) || strings.HasPrefix(real, bad) {
			return fmt.Errorf("%s resolves under %s — the model archive is not a read path (CLAUDE.md § Models); models-pull it into ~/models", p, bad)
		}
	}
	return nil
}

// identityCell is one (asset, quant) run: the unit that is dumped and compared.
type identityCell struct {
	ID      string // file-safe, unique
	Family  string
	Asset   string
	Path    string // absolute
	Quant   string
	Prompts [][]int
	NotRun  string // set: not run, and why
}

// planIdentityCells builds the cells for the selected families. Families not in the manifest are an
// error (a typo must not read as "nothing to check").
func planIdentityCells(o *identityOpts, fams map[string]manifestFamily) ([]*identityCell, []string, error) {
	var selected []string
	if len(o.Families) > 0 {
		for _, f := range o.Families {
			if _, ok := fams[f]; !ok {
				return nil, nil, fmt.Errorf("family %q is not in testdata/parity_manifest.json", f)
			}
			selected = append(selected, f)
		}
	} else {
		for f := range fams {
			selected = append(selected, f)
		}
	}
	sort.Strings(selected)
	want := map[string]bool{}
	for _, f := range selected {
		want[f] = true
	}
	table := identityTiny
	if o.Assets == "real" {
		table = identityReal
	}
	quants := o.Quants
	if len(quants) == 0 {
		quants = defaultIdentityQuants(o.Backend, o.Assets)
	}
	var prompts []string
	var cells []*identityCell
	has := map[string]bool{}
	for _, a := range table {
		if !want[a.Family] {
			continue
		}
		if len(o.Only) > 0 && !slicesContains(o.Only, a.Name) {
			continue
		}
		if len(a.Backends) > 0 && !slicesContains(a.Backends, o.Backend) {
			continue
		}
		has[a.Family] = true
		base := &identityCell{Family: a.Family, Asset: a.Name}
		var ids [][]int
		switch {
		case a.Skip != "":
			base.NotRun = a.Skip
		case o.Assets == "real":
			base.Path = filepath.Join(o.ModelsDir, a.Path)
			if err := identityPathAllowed(base.Path); err != nil {
				return nil, nil, err
			}
			if why := assetPresent(base.Path); why != "" {
				base.NotRun = why
				break
			}
			if prompts == nil {
				prompts = identityRealPrompts()
			}
			tk, err := loadIdentityTokenizer(filepath.Join(o.ModelsDir, a.Tok))
			if err != nil {
				base.NotRun = "tokenizer: " + err.Error()
				break
			}
			for _, p := range prompts {
				t, err := tk.Encode(p, true)
				if err != nil {
					base.NotRun = "tokenize: " + err.Error()
					break
				}
				ids = append(ids, t)
			}
		default:
			base.Path = filepath.Join(o.FixtureRoot, a.Path)
			if err := identityPathAllowed(base.Path); err != nil {
				return nil, nil, err
			}
			if why := assetPresent(base.Path); why != "" {
				base.NotRun = why
				break
			}
			p, err := readGoldenPrompt(filepath.Join(o.FixtureRoot, a.Golden))
			if err != nil {
				base.NotRun = "parity prompt: " + err.Error()
				break
			}
			ids = [][]int{p}
			if lp := longPrompt(p, maxPositions(base.Path), o.Steps); lp != nil {
				ids = append(ids, lp)
			}
		}
		for _, q := range quants {
			c := *base
			c.Quant = q
			c.Prompts = ids
			c.ID = sanitize(c.Family + "__" + c.Asset + "__" + q)
			cells = append(cells, &c)
		}
	}
	for _, f := range selected {
		if has[f] {
			continue
		}
		why := identityNoTiny[f]
		if o.Assets == "real" {
			why = "no real checkpoint for this family in the -assets real table"
		} else if why == "" {
			why = "no asset for this family (and no reason recorded in identityNoTiny)"
		}
		cells = append(cells, &identityCell{ID: sanitize(f + "__none"), Family: f, Asset: "-", Quant: "-", NotRun: why})
	}
	return cells, selected, nil
}

func loadIdentityTokenizer(p string) (*tokenizer.Tokenizer, error) {
	if err := identityPathAllowed(p); err != nil {
		return nil, err
	}
	if strings.HasSuffix(p, ".gguf") {
		return tokenizer.LoadGGUF(p)
	}
	return tokenizer.Load(p)
}

// defaultIdentityQuants: on CPU the tiny fixtures run at f32 (what their goldens load) AND at the two
// shipped quantized paths, because a numerically-neutral claim about a W8A8 or int4 kernel is not
// exercised by an f32 run at all. A GPU backend declines f32 residency, so it runs the quantized paths
// only: both on Metal, int4 alone on WebGPU, where most tiny families run staged (one dispatch per
// matmul) and both quants made the run too slow for the day loop; -quant int4,int8int8 restores the
// second. Real checkpoints run int4, the served quant.
func defaultIdentityQuants(backend, assets string) []string {
	switch {
	case assets == "real":
		return []string{"int4"}
	case backend == "cpu":
		return []string{"f32", "int8int8", "int4"}
	case backend == "webgpu":
		return []string{"int4"}
	}
	return []string{"int4", "int8int8"}
}
