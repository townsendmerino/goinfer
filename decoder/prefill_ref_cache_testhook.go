//go:build goinfer_testhooks

package decoder

// A content-keyed cache for the prefill fidelity gates' CPU f32-activation references (docs/tasks/task-test-efficiency-2026-09.md,
// TE6(a) and TE8). TestPrefillGateReference is the longest single run in the test census and all-or-nothing, and a stale
// reference cannot be spotted by hand (prefill-ref-identity-2026-09-26.md records a set scored against logits for different
// text). So each (model, K, prompt) reference is keyed by everything it depends on:
//
//   - the checkpoint's sha256 (cached by path + size + mtime, so a 4.7 GB file is hashed once);
//   - the prompt's own token ids at that K (sha256, the same encoding as internal/fidelity.PromptSetHash);
//   - the source the CPU reference path compiles from: every non-test .go file in decoder/, internal/giw/ and constrain/ (from
//     go list -deps ./decoder) plus go.mod, which pins aikit and golang.org/x;
//   - runtime.GOARCH (the CPU reference is bit-identical within an arch, not across), the weight quant, the continuation length,
//     and the forced exact attention.
//
// A hit is a lookup, not a judgement; a miss is computed and stored atomically, so an interrupted generator resumes where it
// stopped. The historical ~/goinfer-logs/prefill-ref[-<set>]/<model>-K<k>-p<i>.bin path is populated from the cache (a hard link
// where the filesystem allows, else a copy), with a <file>.key.json sidecar beside it, so every consumer keeps reading the path it
// reads today and can check the sidecar's prompt hash exactly.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// PrefillRefKeyParts is what a reference depends on. Key() is the sha256 of its canonical JSON.
type PrefillRefKeyParts struct {
	Version       int    `json:"version"`
	Checkpoint    string `json:"checkpoint_sha256"`
	CheckpointRef string `json:"checkpoint_path"` // informational, not part of the key's identity beyond the sha
	PromptSHA256  string `json:"prompt_sha256"`
	PromptTokens  int    `json:"prompt_tokens"`
	SourceSHA256  string `json:"source_sha256"`
	Arch          string `json:"arch"`
	Quant         string `json:"quant"`
	ContinuationN int    `json:"continuation_n"`
	ExactAttn     bool   `json:"exact_attention"`
}

// Key is the cache key: sha256 over the parts that determine the reference (the checkpoint path is excluded).
func (p PrefillRefKeyParts) Key() string {
	q := p
	q.CheckpointRef = ""
	b, _ := json.Marshal(q)
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// PromptIDsSHA256 hashes one prompt's ids with internal/fidelity.PromptSetHash's encoding (length, then each id, as
// little-endian uint64), so a sidecar's prompt hash equals PromptSetHash([][]int{ids}).
func PromptIDsSHA256(ids []int) string {
	h := sha256.New()
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(ids)))
	h.Write(b[:])
	for _, t := range ids {
		binary.LittleEndian.PutUint64(b[:], uint64(int64(t)))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// PrefillRefCacheDirForTest is ~/goinfer-logs/prefill-ref-cache, or GOINFER_PREFILL_REF_CACHE (a test hook) when set.
func PrefillRefCacheDirForTest() (string, error) {
	if d := os.Getenv("GOINFER_PREFILL_REF_CACHE"); d != "" {
		return d, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "goinfer-logs", "prefill-ref-cache"), nil
}

var (
	refSourceOnce sync.Once
	refSourceHash string
	refSourceErr  error
)

// prefillRefSourceSHA256 hashes the reference path's source: every non-test .go file in decoder/, internal/giw/ and
// constrain/, plus go.mod, in sorted relative-path order, each path and content length-prefixed.
func prefillRefSourceSHA256() (string, error) {
	refSourceOnce.Do(func() {
		_, self, _, ok := runtime.Caller(0)
		if !ok {
			refSourceErr = errors.New("prefill ref cache: cannot locate the source tree")
			return
		}
		root := filepath.Dir(filepath.Dir(self)) // decoder/ -> module root
		var files []string
		for _, dir := range []string{"decoder", filepath.Join("internal", "giw"), "constrain"} {
			ents, err := os.ReadDir(filepath.Join(root, dir))
			if err != nil {
				refSourceErr = err
				return
			}
			for _, e := range ents {
				n := e.Name()
				if e.IsDir() || !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
					continue
				}
				files = append(files, filepath.Join(dir, n))
			}
		}
		files = append(files, "go.mod")
		sort.Strings(files)
		h := sha256.New()
		for _, rel := range files {
			b, err := os.ReadFile(filepath.Join(root, rel))
			if err != nil {
				refSourceErr = err
				return
			}
			fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(rel), len(b))
			h.Write(b)
		}
		refSourceHash = hex.EncodeToString(h.Sum(nil))
	})
	return refSourceHash, refSourceErr
}

// checkpointSHA256 hashes a checkpoint file, caching the result by absolute path + size + mtime in the cache dir's
// filehash.json so each file is read in full once.
func checkpointSHA256(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		abs = real
	}
	st, err := os.Stat(abs)
	if err != nil {
		return "", err
	}
	if st.IsDir() {
		return checkpointDirSHA256(abs)
	}
	stamp := fmt.Sprintf("%s|%d|%d", abs, st.Size(), st.ModTime().UnixNano())
	dir, err := PrefillRefCacheDirForTest()
	if err != nil {
		return "", err
	}
	idxPath := filepath.Join(dir, "filehash.json")
	idx := map[string]string{}
	if b, err := os.ReadFile(idxPath); err == nil {
		_ = json.Unmarshal(b, &idx)
	}
	if h, ok := idx[stamp]; ok {
		return h, nil
	}
	f, err := os.Open(abs)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hh := sha256.New()
	if _, err := io.Copy(hh, f); err != nil {
		return "", err
	}
	sum := hex.EncodeToString(hh.Sum(nil))
	idx[stamp] = sum
	if err := os.MkdirAll(dir, 0o755); err == nil {
		if b, err := json.MarshalIndent(idx, "", " "); err == nil {
			_ = writeFileAtomic(idxPath, b)
		}
	}
	return sum, nil
}

// checkpointDirSHA256 is checkpointSHA256 for a safetensors checkpoint directory: every regular file under it, by relative path,
// each through checkpointSHA256's own cached file hash, folded into one digest in path order. Hidden files and goinfer's own
// sidecars (*.giw and their *.verified markers, written beside a checkpoint after the fact) are left out: they say nothing about
// the weights, and counting them would re-key every reference the first time a model is transcoded.
func checkpointDirSHA256(dir string) (string, error) {
	var rels []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if p != dir && strings.HasPrefix(name, ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() || !d.Type().IsRegular() || strings.HasSuffix(name, ".giw") || strings.HasSuffix(name, ".verified") {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		rels = append(rels, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		return "", err
	}
	if len(rels) == 0 {
		return "", fmt.Errorf("%s: no checkpoint files", dir)
	}
	sort.Strings(rels)
	hh := sha256.New()
	for _, rel := range rels {
		h, err := checkpointSHA256(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			return "", err
		}
		fmt.Fprintf(hh, "%s\x00%s\n", rel, h)
	}
	return "dir:" + hex.EncodeToString(hh.Sum(nil)), nil
}

// PrefillRefKeyForTest builds the key parts for one reference.
func PrefillRefKeyForTest(checkpointPath, quant string, ids []int, continuationN int) (PrefillRefKeyParts, error) {
	ck, err := checkpointSHA256(checkpointPath)
	if err != nil {
		return PrefillRefKeyParts{}, fmt.Errorf("checkpoint hash: %w", err)
	}
	src, err := prefillRefSourceSHA256()
	if err != nil {
		return PrefillRefKeyParts{}, fmt.Errorf("source hash: %w", err)
	}
	return PrefillRefKeyParts{
		Version: 1, Checkpoint: ck, CheckpointRef: checkpointPath, PromptSHA256: PromptIDsSHA256(ids),
		PromptTokens: len(ids), SourceSHA256: src, Arch: runtime.GOARCH, Quant: quant,
		ContinuationN: continuationN, ExactAttn: os.Getenv("GOINFER_CPU_FAST_ATTENTION") == "0",
	}, nil
}

func prefillRefCachePath(key string) (string, error) {
	dir, err := PrefillRefCacheDirForTest()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, key[:2], key+".bin"), nil
}

// LookupPrefillRefForTest returns the cached reference file for these parts, if one is complete.
func LookupPrefillRefForTest(parts PrefillRefKeyParts) (string, bool) {
	p, err := prefillRefCachePath(parts.Key())
	if err != nil {
		return "", false
	}
	if _, err := os.Stat(p); err != nil {
		return "", false
	}
	if _, err := os.Stat(p + ".key.json"); err != nil {
		return "", false // a .bin without its parts record is an interrupted store, not a hit
	}
	return p, true
}

// StorePrefillRefForTest writes a reference into the cache atomically (the .bin, then its parts record last).
func StorePrefillRefForTest(parts PrefillRefKeyParts, seedLogits []float32, refTokens []int, refLogits [][]float32) (string, error) {
	p, err := prefillRefCachePath(parts.Key())
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	tmp := p + ".tmp"
	if err := WritePrefillReferenceForTest(tmp, seedLogits, refTokens, refLogits); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	if err := os.Rename(tmp, p); err != nil {
		return "", err
	}
	b, _ := json.MarshalIndent(parts, "", " ")
	if err := writeFileAtomic(p+".key.json", b); err != nil {
		return "", err
	}
	return p, nil
}

// LinkPrefillRefForTest places a cached reference at a consumer's historical path (hard link, else copy), with the
// parts record beside it as <legacyPath>.key.json.
func LinkPrefillRefForTest(cachePath, legacyPath string, parts PrefillRefKeyParts) error {
	if err := os.MkdirAll(filepath.Dir(legacyPath), 0o755); err != nil {
		return err
	}
	tmp := legacyPath + ".tmp"
	_ = os.Remove(tmp)
	if err := os.Link(cachePath, tmp); err != nil {
		if err := copyFile(cachePath, tmp); err != nil {
			return err
		}
	}
	if err := os.Rename(tmp, legacyPath); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(parts, "", " ")
	return writeFileAtomic(legacyPath+".key.json", b)
}

// PrefillRefIdentity is the result of checking a reference file against the prompt it is about to be scored with.
type PrefillRefIdentity int

const (
	RefIdentityNoSidecar PrefillRefIdentity = iota // a legacy file: fall back to the seed-logit KL inference
	RefIdentityVerified                            // the sidecar's prompt hash equals the prompt's
	RefIdentityMismatch                            // the reference was built from different ids: VOID
)

// PrefillRefIdentityForTest checks refPath's sidecar against the ids the gate is about to score it with.
func PrefillRefIdentityForTest(refPath string, ids []int) (PrefillRefIdentity, string) {
	b, err := os.ReadFile(refPath + ".key.json")
	if err != nil {
		return RefIdentityNoSidecar, "no sidecar (a file from before the content-keyed cache)"
	}
	var parts PrefillRefKeyParts
	if err := json.Unmarshal(b, &parts); err != nil {
		return RefIdentityMismatch, fmt.Sprintf("unreadable sidecar: %v", err)
	}
	if parts.PromptSHA256 != PromptIDsSHA256(ids) || parts.PromptTokens != len(ids) {
		return RefIdentityMismatch, fmt.Sprintf("prompt mismatch: the reference was built from a %d-token prompt %s…, "+
			"the gate is scoring %d tokens %s…", parts.PromptTokens, parts.PromptSHA256[:12], len(ids), PromptIDsSHA256(ids)[:12])
	}
	return RefIdentityVerified, "prompt verified by sidecar hash"
}

func writeFileAtomic(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
