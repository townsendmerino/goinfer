package pull

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// A safetensors checkpoint as a pull target (docs/tasks/task-checkpoint-fetch-2026-09.md, P1-P3 and P5). GGUF is one
// file; a safetensors checkpoint is a SET: config.json, the tokenizer files, and either model.safetensors or an index
// plus the shards it names. decoder.Load already opens such a directory, so this only gets the set onto disk, with the
// guarantees a single file has: every file digest-verified, resumable, and nothing at the final path until the whole set
// is there. Anonymous only (the doc's §2, option (c), owner 2026-10-03): a gated original is declined by CheckAccess
// before any of this runs.

// CheckpointSelector is the selector that names a repo's safetensors checkpoint ("owner/repo:safetensors"). No GGUF
// quant is called that, so it cannot shadow a quant selector.
const CheckpointSelector = "safetensors"

// Plan is what a safetensors checkpoint pull would fetch, decided before any weight byte moves.
type Plan struct {
	Repo      string
	Files     []File // config, tokenizer and processor files, then the weights; sorted by path
	Bytes     int64  // the sum of Files' sizes: what the transfer costs in disk and bandwidth
	ModelType string // config.json's model_type
	Family    string // the capability-matrix family that loads it
	Dtype     string // config.json's torch_dtype (or dtype), "" if absent
}

// checkpointExtras are the non-weight files a checkpoint plan takes when the repo has them: the tokenizer in its several
// forms, the chat template, generation defaults, the image/video processor configs a vision-language family needs, and
// the sentence-transformers module files an embedding model's pooling is read from (aikit's encoder reads
// 1_Pooling/config.json; without it, pooling falls back to its default).
var checkpointExtras = map[string]bool{
	"config.json": true, "generation_config.json": true,
	"tokenizer.json": true, "tokenizer_config.json": true, "tokenizer.model": true, "special_tokens_map.json": true,
	"added_tokens.json": true, "vocab.json": true, "merges.txt": true,
	"chat_template.jinja": true, "chat_template.json": true,
	"preprocessor_config.json": true, "processor_config.json": true, "video_preprocessor_config.json": true,
	"modules.json": true, "config_sentence_transformers.json": true, "sentence_bert_config.json": true, "1_Pooling/config.json": true,
}

// generativeLoads is PlanCheckpoint's model_type check: a family in the capability matrix with a safetensors loader.
func generativeLoads(modelType string) (string, error) {
	if fam, ok := FamilyForModelType(modelType); ok {
		return fam, nil
	}
	return "", fmt.Errorf("model_type %q, which this build cannot load from safetensors (docs/capability-matrix.json)", modelType)
}

// EncoderFamily is the family a checkpoint loads as when it is the embedding encoder's (EncoderLoads), not a generative
// model's: it opens with serve --embed-model, not --model.
const EncoderFamily = "nomic_bert encoder"

// EmbeddingGemma2Family is the family an EmbeddingGemma 2 checkpoint loads as (serve --embed-model, through
// goinfer's own embeddinggemma2 package rather than aikit's encoder).
const EmbeddingGemma2Family = "embedding_gemma2 encoder"

// EncoderLoads is the model_type check of goinfer's embedding encoders (serve --embed-model): aikit's encoder.Load,
// which loads a NomicBert (CodeRankEmbed, nomic-embed-text) and checks no model_type itself, and the embeddinggemma2
// package. The one list of what they take: serve's own check and the pull CLI both read it.
func EncoderLoads(modelType string) (string, error) {
	switch modelType {
	case "nomic_bert":
		return EncoderFamily, nil
	case "embedding_gemma2":
		return EmbeddingGemma2Family, nil
	}
	return "", fmt.Errorf("model_type %q, which the embedding encoder does not load (it loads nomic_bert and embedding_gemma2)", modelType)
}

// AnyLoads is the pull CLI's model_type check: a generative family this build loads from safetensors, else the
// embedding encoder. The CLI fetches for whatever will open the files; --model and the web UI, which load generative
// models only, keep PlanCheckpoint's check (task-checkpoint-fetch P7: `pull <encoder>:safetensors` used to decline what
// serve --embed-model fetches itself).
func AnyLoads(modelType string) (string, error) {
	if fam, err := generativeLoads(modelType); err == nil {
		return fam, nil
	}
	if fam, err := EncoderLoads(modelType); err == nil {
		return fam, nil
	}
	return "", fmt.Errorf("model_type %q, which this build loads neither as a generative model from safetensors (docs/capability-matrix.json) nor as an embedding encoder (nomic_bert, embedding_gemma2)", modelType)
}

// safeRepoPath reports whether a repo file path is safe to join under a local directory: relative, no ".." or empty
// segment, no backslash. A tree listing is remote input.
func safeRepoPath(p string) bool {
	if p == "" || strings.HasPrefix(p, "/") || strings.Contains(p, "\\") {
		return false
	}
	for _, seg := range strings.Split(p, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
	}
	return true
}

// listTree returns every file in the repo's main branch (recursively), with LFS digests.
func listTree(ctx context.Context, repo string) ([]File, error) {
	resp, err := get(ctx, hfAPI+"/"+repo+"/tree/main?recursive=true")
	if err != nil {
		return nil, fmt.Errorf("listing %s: %w", repo, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("listing %s: HuggingFace returned %s", repo, resp.Status)
	}
	var entries []treeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, fmt.Errorf("parsing file list: %w", err)
	}
	var out []File
	for _, e := range entries {
		if e.Type != "file" {
			continue
		}
		f := File{Path: e.Path, Size: e.Size}
		if e.LFS != nil {
			f.SHA256 = e.LFS.OID
		}
		out = append(out, f)
	}
	return out, nil
}

// fetchSmall reads a small repo file (a config or an index) into memory, refusing anything over max bytes.
func fetchSmall(ctx context.Context, repo, p string, max int64) ([]byte, error) {
	resp, err := get(ctx, hfCDN+"/"+repo+"/resolve/main/"+p)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", p, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: HuggingFace returned %s", p, resp.Status)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, max+1))
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", p, err)
	}
	if int64(len(b)) > max {
		return nil, fmt.Errorf("%s is larger than %d bytes; not a config", p, max)
	}
	return b, nil
}

// matrixFamily is the subset of a capability-matrix row P5 reads.
type matrixFamily struct {
	Name       string   `json:"name"`
	ModelTypes []string `json:"model_types"`
	Loaders    string   `json:"loaders"`
	Modality   string   `json:"modality"`
}

// FamilyForModelType finds the capability-matrix family whose model_types include mt and that loads from safetensors
// (P5: "will this load?" from data already in the tree). ok is false when this build has no such family.
func FamilyForModelType(mt string) (family string, ok bool) {
	var rows []matrixFamily
	if json.Unmarshal(capabilityMatrixJSON, &rows) != nil {
		return "", false
	}
	for _, r := range rows {
		for _, t := range r.ModelTypes {
			if t == mt && strings.Contains(strings.ToLower(r.Loaders), "safetensors") {
				return r.Name, true
			}
		}
	}
	return "", false
}

// PlanCheckpoint decides what a safetensors checkpoint pull of repo would fetch, before any weight moves. It reads the
// tree and two small files (config.json, and the shard index when there is one), and refuses:
//   - a repo with no config.json or no safetensors weights;
//   - a model_type this build cannot load (P5's decline, before the first weight byte);
//   - a shard index that names a file the repo does not have, or an unsafe path.
//
// The caller runs CheckAccess first, as for a GGUF pull.
func PlanCheckpoint(ctx context.Context, repo string) (Plan, error) {
	return PlanCheckpointFor(ctx, repo, generativeLoads)
}

// PlanCheckpointFor is PlanCheckpoint with the caller's model_type check. loads names the family that will load the
// checkpoint, or returns why it cannot, and a refusal still comes after reading config.json and before any weight file.
// serve's -embed-model uses it with its encoder's own check, because an embedding encoder is not a generative family in
// the capability matrix (task-checkpoint-fetch P7).
func PlanCheckpointFor(ctx context.Context, repo string, loads func(modelType string) (family string, err error)) (Plan, error) {
	files, err := listTree(ctx, repo)
	if err != nil {
		return Plan{}, err
	}
	byPath := map[string]File{}
	for _, f := range files {
		byPath[f.Path] = f
	}
	if _, ok := byPath["config.json"]; !ok {
		return Plan{}, fmt.Errorf("repo %s has no config.json: not a safetensors (transformers) checkpoint", repo)
	}
	cfgRaw, err := fetchSmall(ctx, repo, "config.json", 4<<20)
	if err != nil {
		return Plan{}, err
	}
	var cfg struct {
		ModelType  string `json:"model_type"`
		TorchDtype string `json:"torch_dtype"`
		Dtype      string `json:"dtype"`
	}
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return Plan{}, fmt.Errorf("parsing %s's config.json: %w", repo, err)
	}
	if cfg.ModelType == "" {
		return Plan{}, fmt.Errorf("repo %s's config.json has no model_type, so goinfer cannot tell what it is", repo)
	}
	fam, err := loads(cfg.ModelType)
	if err != nil {
		return Plan{}, fmt.Errorf("repo %s is %w; nothing was downloaded", repo, err)
	}
	p := Plan{Repo: repo, ModelType: cfg.ModelType, Family: fam, Dtype: cfg.TorchDtype}
	if p.Dtype == "" {
		p.Dtype = cfg.Dtype
	}
	want := map[string]bool{}
	for name := range checkpointExtras {
		if _, ok := byPath[name]; ok {
			want[name] = true
		}
	}
	if _, ok := byPath["model.safetensors.index.json"]; ok {
		idxRaw, err := fetchSmall(ctx, repo, "model.safetensors.index.json", 64<<20)
		if err != nil {
			return Plan{}, err
		}
		var idx struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(idxRaw, &idx); err != nil {
			return Plan{}, fmt.Errorf("parsing %s's shard index: %w", repo, err)
		}
		if len(idx.WeightMap) == 0 {
			return Plan{}, fmt.Errorf("repo %s's shard index names no weights", repo)
		}
		want["model.safetensors.index.json"] = true
		for _, shard := range idx.WeightMap {
			if _, ok := byPath[shard]; !ok {
				return Plan{}, fmt.Errorf("repo %s's shard index names %q, which the repo does not have", repo, shard)
			}
			want[shard] = true
		}
	} else if _, ok := byPath["model.safetensors"]; ok {
		want["model.safetensors"] = true
	} else {
		return Plan{}, fmt.Errorf("repo %s has a config.json but no model.safetensors or model.safetensors.index.json", repo)
	}
	for name := range want {
		if !safeRepoPath(name) {
			return Plan{}, fmt.Errorf("repo %s lists an unsafe path %q", repo, name)
		}
		f := byPath[name]
		p.Files = append(p.Files, f)
		p.Bytes += f.Size
	}
	sort.Slice(p.Files, func(i, j int) bool { return p.Files[i].Path < p.Files[j].Path })
	return p, nil
}

// TreeDigest is the digest a registry entry pins for a checkpoint directory: sha256 over one line per planned file, in path
// order, "path<TAB>size<TAB>sha256-or-dash". The weights are LFS files, so their lines carry the digest Hugging Face
// declares; a small non-LFS file (config, tokenizer) carries its size only, which is what the tree listing offers. It is the
// directory's counterpart of a GGUF entry's sha256: the same upstream-re-upload guard, for a set.
func (p Plan) TreeDigest() string {
	files := append([]File(nil), p.Files...)
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	h := sha256.New()
	for _, f := range files {
		sum := f.SHA256
		if sum == "" {
			sum = "-"
		}
		fmt.Fprintf(h, "%s\t%d\t%s\n", f.Path, f.Size, sum)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// VerifyPin is the check a registry entry's directory pull makes after planning and before any weight byte moves: the repo's
// files must hash to the tree digest this build pins. An empty pin (the explicit owner/repo:safetensors form) pins nothing.
func (p Plan) VerifyPin(pin string) error {
	if pin == "" {
		return nil
	}
	if got := p.TreeDigest(); got != pin {
		return fmt.Errorf("%s has changed since this build was cut (tree digest %s, this build pins %s); nothing was downloaded. "+
			"Fetch the new files with the explicit form owner/repo:%s, which pins nothing", p.Repo, got, pin, CheckpointSelector)
	}
	return nil
}

// SizeNote is the disk-and-bandwidth line a plan must show before the transfer (the doc's §3 rule 4): the full-precision
// original costs several times a GGUF of the same model, which goinfer would run at the same quant.
func (p Plan) SizeNote() string {
	s := fmt.Sprintf("%d files, %s to download", len(p.Files), humanBytes(p.Bytes))
	switch strings.ToLower(p.Dtype) {
	case "bfloat16", "float16", "float32":
		s += fmt.Sprintf(" (%s weights: the full-precision original; goinfer quantizes at load, so a GGUF q4 of the same model would download about a quarter of this)", p.Dtype)
	}
	return s
}

// checkpointMarker is the file a completed checkpoint directory carries: the plan it was built from, so a later pull or
// Resolve can tell a complete, goinfer-pulled checkpoint from anything else at that path and verify it offline.
const checkpointMarker = ".goinfer-checkpoint.json"

type markerDoc struct {
	Repo  string `json:"repo"`
	Files []File `json:"files"`
}

// checkpointComplete reports whether dir is a complete checkpoint for these files: the marker is there, names the same
// files, and every file verifies (size and digest, through the digest sidecar cache).
func checkpointComplete(dir string, files []File) bool {
	b, err := os.ReadFile(filepath.Join(dir, checkpointMarker))
	if err != nil {
		return false
	}
	var m markerDoc
	if json.Unmarshal(b, &m) != nil || len(m.Files) != len(files) {
		return false
	}
	for i := range files {
		if m.Files[i] != files[i] {
			return false
		}
	}
	for _, f := range files {
		if _, ok := cachedIntact(filepath.Join(dir, filepath.FromSlash(path.Dir(f.Path))), f); !ok {
			return false
		}
	}
	return true
}

// CachedCheckpoint is a complete checkpoint already in dir, verified offline from its marker (no network). ok is false
// when dir has none.
func CachedCheckpoint(dir string) (string, bool) {
	b, err := os.ReadFile(filepath.Join(dir, checkpointMarker))
	if err != nil {
		return "", false
	}
	var m markerDoc
	if json.Unmarshal(b, &m) != nil || len(m.Files) == 0 {
		return "", false
	}
	if !checkpointComplete(dir, m.Files) {
		return "", false
	}
	return dir, true
}

// DownloadCheckpoint fetches every file in p into dest and returns dest. The set is assembled in a staging directory
// beside dest (dest + ".partial"), each file through Download (its own .part, digest check and resume), and dest
// appears only when the whole set has verified: one rename, after a marker naming the set is written. So an interrupted
// pull leaves no loadable-looking directory at dest, and a re-run resumes in the staging directory, skipping files
// already verified there. A complete dest is a no-op. A dest that exists without a matching marker is refused rather than
// overwritten. progress reports bytes over the whole set and the file in flight; it may be nil.
func DownloadCheckpoint(ctx context.Context, p Plan, dest string, progress func(done, total int64, file string)) (string, error) {
	if len(p.Files) == 0 {
		return "", fmt.Errorf("checkpoint plan for %s has no files", p.Repo)
	}
	if checkpointComplete(dest, p.Files) {
		return dest, nil
	}
	if _, err := os.Stat(dest); err == nil {
		return "", fmt.Errorf("%s exists but is not a complete checkpoint of %s pulled by goinfer: move it aside, or pull with -o <dir>", dest, p.Repo)
	}
	staging := dest + ".partial"
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return "", err
	}
	var done int64
	for _, f := range p.Files {
		if !safeRepoPath(f.Path) {
			return "", fmt.Errorf("unsafe path %q in the plan for %s", f.Path, p.Repo)
		}
		dir := filepath.Join(staging, filepath.FromSlash(path.Dir(f.Path)))
		base := done
		if _, err := Download(ctx, p.Repo, f, dir, func(d, _ int64) {
			if progress != nil {
				progress(base+d, p.Bytes, f.Path)
			}
		}); err != nil {
			return "", fmt.Errorf("%s (file %s; re-run to resume, the files already verified are kept): %w", p.Repo, f.Path, err)
		}
		done += f.Size
		if progress != nil {
			progress(done, p.Bytes, f.Path)
		}
	}
	mb, err := json.MarshalIndent(markerDoc{Repo: p.Repo, Files: p.Files}, "", " ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(staging, checkpointMarker), mb, 0o644); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(staging, dest); err != nil {
		return "", fmt.Errorf("publishing %s: %w", dest, err)
	}
	return dest, nil
}
