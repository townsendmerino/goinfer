// Package pull fetches a GGUF checkpoint from HuggingFace onto local disk.
//
// EXPERIMENTAL (docs/api-tiers.md): supported and used by every goinfer front end, but it may
// change in any release. It was `internal/modelpull` until 2026-09-02; it is exported because a
// library caller embedding goinfer needs the same first step the CLIs and the web UI take, and
// re-deriving it is the one part of "get a model" that has no other owner.
//
// It is deliberately the ONLY new capability in the model-pull work: goinfer already
// loads a .gguf or .giw (--model), already transcodes a bare .gguf to a sidecar .giw
// cache on first use, and already converts one offline (cmd/prequant). The single step
// that existed nowhere in the repo was getting the bytes onto disk — before this, a
// `grep` for outbound HTTP in Go code returned nothing, and the only fetch anywhere was
// a curl in the release workflow. So this package gets bytes onto disk and stops; it
// deliberately does NOT load, convert, quantize, or embed, because each of those already
// has a tested owner and a second copy would drift from it.
//
// No new module dependency: net/http, crypto/sha256 and encoding/json only, which keeps
// the cgo-free single-static-binary property the rest of the project is built around.
package pull

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// hfAPI and hfCDN are split because they are different services with different failure
// modes: the API answers metadata (and is what reports `gated`), the resolve host streams
// bytes through a redirect to a CDN.
// vars, not consts, solely so tests can point them at an httptest server. Nothing in the
// shipped paths reassigns them.
var (
	hfAPI = "https://huggingface.co/api/models"
	hfCDN = "https://huggingface.co"
)

// userAgent identifies goinfer to HF. Anonymous requests are rate-limited by IP; naming the
// client is the courteous minimum and makes our traffic attributable.
const userAgent = "goinfer-pull/1 (+https://github.com/townsendmerino/goinfer)"

// Ref is a parsed model reference: a HuggingFace repo plus either an exact filename or a
// quant selector to resolve against the repo's listing.
//
// The explicit form is the real interface. There is deliberately no name→repo table here
// beyond what the caller supplies: an opaque tag would hide which repo and which file the
// bytes came from, and this project's whole citation/provenance discipline runs the other
// way — "here is exactly what was pulled" is the property worth keeping.
type Ref struct {
	Repo  string // "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
	File  string // exact filename, when given after the colon
	Quant string // quant selector ("q4_k_m"), when the colon part is not a filename
	// Pin is a sha256 this ref REQUIRES, set only for curated demo: refs. It is checked
	// against what the API reports before anything downloads. That is the point of pinning:
	// `resolve/main` is a moving reference, so an upstream re-upload would otherwise change
	// the bytes under a name goinfer vouches for, and the API-declared digest would happily
	// verify the NEW file. Empty for user-supplied refs, which pin nothing by construction.
	Pin string
}

//go:embed curated.json
var curatedJSON []byte

// curated is the parsed curated.json, loaded once.
var curated = struct {
	once  sync.Once
	tiers map[string]CuratedTier
}{}

// CuratedTier is one vetted model: an exact repo + filename plus the digest and size this
// build pins for it.
type CuratedTier struct {
	Repo   string `json:"repo"`
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// Curated returns the vetted demo tiers keyed by short name ("0.5b").
func Curated() map[string]CuratedTier {
	curated.once.Do(func() {
		var doc struct {
			Tiers map[string]CuratedTier `json:"tiers"`
		}
		if err := json.Unmarshal(curatedJSON, &doc); err != nil {
			panic("pull: curated.json is malformed: " + err.Error()) // build-time asset; a parse failure is a bug, not input
		}
		curated.tiers = doc.Tiers
	})
	return curated.tiers
}

// CuratedNames returns the tier names, sorted, for help text and error messages.
func CuratedNames() []string {
	names := make([]string, 0, len(Curated()))
	for k := range Curated() {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// ParseRef accepts "owner/repo", "owner/repo:file.gguf" or "owner/repo:quant".
func ParseRef(s string) (Ref, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Ref{}, fmt.Errorf("empty model reference")
	}
	repo, sel, hasSel := strings.Cut(s, ":")
	// demo:<tier> — the only non-explicit form, and deliberately the only one. It resolves to
	// a concrete repo + filename + pinned digest from curated.json, so it is shorthand for an
	// exact reference rather than an opaque tag: what it pulled is still nameable afterwards.
	if repo == "demo" {
		t, ok := Curated()[sel]
		if !ok {
			return Ref{}, fmt.Errorf("unknown demo model %q; have: %s", sel, strings.Join(CuratedNames(), ", "))
		}
		return Ref{Repo: t.Repo, File: t.File, Pin: t.SHA256}, nil
	}
	if !validRepo(repo) {
		return Ref{}, fmt.Errorf("%q: want owner/repo[:quant|:file.gguf] (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m)", s)
	}
	r := Ref{Repo: repo}
	if hasSel {
		if sel == "" {
			return Ref{}, fmt.Errorf("%q: empty selector after ':'", s)
		}
		if strings.HasSuffix(strings.ToLower(sel), ".gguf") {
			r.File = sel
		} else {
			r.Quant = sel
		}
	}
	return r, nil
}

// repoSegment is HuggingFace's owner/name charset. Validating against an ALLOW-list rather
// than blocking bad characters is deliberate: the repo string is interpolated into a URL
// path AND joined into a filesystem path, so anything that slips through is wrong in two
// places at once. "../.." satisfies a naive "exactly one slash, no leading or trailing
// slash" check — which is what this replaced — and would then escape the cache directory
// under filepath.Join. Reachable remotely once the repo name can come from an HTTP request.
var repoSegment = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

// validRepo reports whether repo is exactly "owner/name" with both segments in the safe
// charset and neither being a relative-path element.
func validRepo(repo string) bool {
	owner, name, ok := strings.Cut(repo, "/")
	if !ok || strings.Contains(name, "/") {
		return false
	}
	for _, seg := range [2]string{owner, name} {
		if seg == "." || seg == ".." || !repoSegment.MatchString(seg) {
			return false
		}
	}
	return true
}

// File is one downloadable file in a repo.
type File struct {
	Path   string
	Size   int64
	SHA256 string // HF's LFS oid, which IS the file's sha256; empty for small non-LFS files
}

// repoInfo is the subset of the models API we read.
type repoInfo struct {
	Gated any  `json:"gated"` // false, or a string like "manual"/"auto" — NOT a bool in general
	Priv  bool `json:"private"`
}

// treeEntry is the subset of the tree API we read.
type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		OID string `json:"oid"`
	} `json:"lfs"`
}

// multiPart matches llama.cpp's split-GGUF naming ("…-00001-of-00003.gguf").
var multiPart = regexp.MustCompile(`-\d{5}-of-\d{5}\.gguf$`)

func get(ctx context.Context, url string) (*http.Response, error) {
	return getRange(ctx, url, 0)
}

// getRange issues the GET, asking to resume from `from` when it is non-zero.
func getRange(ctx context.Context, url string, from int64) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if from > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", from))
	}
	return http.DefaultClient.Do(req)
}

// CheckAccess reports whether the repo is reachable anonymously, naming the reason when it
// is not. Called BEFORE any download so a gated repo fails in a second with an actionable
// message instead of after a multi-GB 401.
//
// Worth knowing when reading a failure here: gating is far less of a wall for GGUF than it
// looks. Upstream originals are frequently gated (google/gemma-3-4b-it and
// meta-llama/Llama-3.2-3B-Instruct both report gated="manual"), but the community GGUF
// re-uploads that this command actually targets are not — bartowski/google_gemma-3-4b-it-GGUF
// and unsloth/Llama-3.2-3B-Instruct-GGUF both report gated=false. So the usual fix is to
// point at a GGUF repo, which is what you wanted anyway.
func CheckAccess(ctx context.Context, repo string) error {
	resp, err := get(ctx, hfAPI+"/"+repo)
	if err != nil {
		return fmt.Errorf("reaching HuggingFace: %w", err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return fmt.Errorf("repo %q not found (check the owner/name, and that it is a GGUF repo)", repo)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("repo %q needs authentication; goinfer pull is anonymous-only.\n"+
			"  Download it in a browser and use --model <path>, or pick a community GGUF re-upload (those are usually ungated)", repo)
	default:
		return fmt.Errorf("repo %q: HuggingFace returned %s", repo, resp.Status)
	}
	var info repoInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return fmt.Errorf("parsing repo metadata: %w", err)
	}
	// gated is `false` when open and a STRING ("manual"/"auto") when not, so it cannot be
	// decoded as a bool — doing so fails on exactly the repos this check exists to catch.
	if g, ok := info.Gated.(string); ok && g != "" {
		return fmt.Errorf("repo %q is gated (%s): accept its licence on huggingface.co, then download in a browser and use --model <path>.\n"+
			"  A community GGUF re-upload of the same model is usually ungated and works with pull", repo, g)
	}
	if info.Priv {
		return fmt.Errorf("repo %q is private", repo)
	}
	return nil
}

// List returns the repo's .gguf files, largest-name-sorted for stable output.
func List(ctx context.Context, repo string) ([]File, error) {
	resp, err := get(ctx, hfAPI+"/"+repo+"/tree/main")
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
		if e.Type != "file" || !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		f := File{Path: e.Path, Size: e.Size}
		if e.LFS != nil {
			f.SHA256 = e.LFS.OID
		}
		out = append(out, f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// Select resolves a Ref against a repo listing.
//
// Quant matching is on the FULL suffix "-<quant>.gguf", case-insensitively, and never on a
// substring. That is not fussiness: real repos ship both Q2_K.gguf and Q2_K_L.gguf, so a
// substring match on "q2_k" is ambiguous and would silently hand back whichever sorted first.
// Case-insensitivity is likewise required by the ecosystem, not a nicety — Qwen publishes
// "…-q4_k_m.gguf" and bartowski publishes "…-Q4_K_M.gguf".
func Select(files []File, ref Ref) (File, error) {
	if len(files) == 0 {
		return File{}, fmt.Errorf("repo %s has no .gguf files", ref.Repo)
	}
	var cands []File
	switch {
	case ref.File != "":
		for _, f := range files {
			if strings.EqualFold(f.Path, ref.File) {
				if err := checkPin(f, ref); err != nil {
					return File{}, err
				}
				return f, nil
			}
		}
		return File{}, fmt.Errorf("file %q not found in %s.\n%s", ref.File, ref.Repo, render(files))
	case ref.Quant != "":
		suffix := "-" + strings.ToLower(ref.Quant) + ".gguf"
		for _, f := range files {
			if strings.HasSuffix(strings.ToLower(f.Path), suffix) {
				cands = append(cands, f)
			}
		}
	default:
		return File{}, fmt.Errorf("%s: pick a quant or a file, e.g. %s:q4_k_m\n%s", ref.Repo, ref.Repo, render(files))
	}
	switch len(cands) {
	case 0:
		return File{}, fmt.Errorf("no file matching quant %q in %s.\n%s", ref.Quant, ref.Repo, render(files))
	case 1:
		f := cands[0]
		if multiPart.MatchString(f.Path) {
			return File{}, fmt.Errorf("%s is one shard of a split GGUF; pulling split checkpoints is not supported yet", f.Path)
		}
		return f, nil
	default:
		return File{}, fmt.Errorf("quant %q is ambiguous in %s — name the file exactly.\n%s", ref.Quant, ref.Repo, render(cands))
	}
}

// checkPin refuses a file whose upstream digest no longer matches what this build pinned.
// Fails BEFORE the download, not after: the pin exists precisely because `resolve/main` can
// move, and verifying only against the API-declared digest would cheerfully confirm the new
// file's own hash and report success.
func checkPin(f File, ref Ref) error {
	if ref.Pin == "" || f.SHA256 == "" || f.SHA256 == ref.Pin {
		return nil
	}
	return fmt.Errorf("%s in %s no longer matches the digest goinfer pinned for it:\n  pinned   %s\n  upstream %s\nthe file was re-uploaded; pull it explicitly by name if you still want it", f.Path, ref.Repo, ref.Pin, f.SHA256)
}

func render(files []File) string {
	var b strings.Builder
	b.WriteString("available:\n")
	for _, f := range files {
		b.WriteString(fmt.Sprintf("  %-52s %s\n", f.Path, humanBytes(f.Size)))
	}
	return b.String()
}

// CacheDir is where pulled models land: <user cache>/goinfer/models/<owner>/<repo>.
func CacheDir(repo string) (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "goinfer", "models", filepath.FromSlash(repo)), nil
}

// Download streams f from repo into dir, verifying the sha256 HF declared for it, and
// returns the final path.
//
// The digest is VERIFIED, not merely printed, because HF hands it over before the transfer:
// the tree API's LFS oid is the file's sha256 (confirmed against the digest
// .github/workflows/release-assets.yml already pins for the same file). A hash nobody
// compares is self-documentation, not a check.
//
// Writes to a .part file and renames only after the digest matches, so an interrupted or
// corrupted pull can never leave something at the final path that later looks loadable.
func Download(ctx context.Context, repo string, f File, dir string, progress func(done, total int64)) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	final := filepath.Join(dir, filepath.Base(f.Path))

	// Already present and intact ⇒ nothing to do. Re-verifies rather than trusting the
	// filename, so a truncated earlier attempt is re-fetched instead of silently served.
	if st, err := os.Stat(final); err == nil && st.Size() == f.Size {
		if f.SHA256 == "" {
			return final, nil
		}
		if sum, err := fileSHA256(final); err == nil && sum == f.SHA256 {
			return final, nil
		}
	}

	// Resume from a previous interrupted attempt when one is on disk. Worth doing precisely
	// because these files are multi-gigabyte: losing 4 GB to a dropped connection and starting
	// again is the difference between an annoyance and an unusable command on a flaky link.
	part := final + ".part"
	var resumeFrom int64
	h := sha256.New()
	if st, err := os.Stat(part); err == nil && st.Size() > 0 && (f.Size <= 0 || st.Size() < f.Size) {
		// The running digest has to cover the bytes already on disk, so re-read them through
		// the hash. That costs a local read of the partial — trivial beside re-fetching it,
		// and it keeps the end-to-end sha256 check honest rather than verifying only the tail.
		if n, err := hashPrefix(h, part); err == nil {
			resumeFrom = n
		} else {
			h.Reset()
		}
	}

	resp, err := getRange(ctx, hfCDN+"/"+repo+"/resolve/main/"+f.Path, resumeFrom)
	if err != nil {
		return "", fmt.Errorf("downloading %s: %w", f.Path, err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusPartialContent:
		// Server honoured the Range: append.
	case http.StatusOK:
		// Server ignored the Range (or we asked for none): this body is the WHOLE file, so any
		// partial on disk is now meaningless. Reset both the offset and the digest — appending
		// here would silently produce a corrupt file that fails the checksum with no clue why.
		resumeFrom = 0
		h.Reset()
	default:
		return "", fmt.Errorf("downloading %s: HuggingFace returned %s", f.Path, resp.Status)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if resumeFrom > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	out, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return "", err
	}
	total := f.Size
	if total <= 0 {
		total = resumeFrom + resp.ContentLength
	}
	pw := &progressWriter{w: io.MultiWriter(out, h), done: resumeFrom, total: total, report: progress, last: time.Now()}
	_, copyErr := io.Copy(pw, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		// Deliberately NOT removed: the bytes already fetched are the whole point of resume.
		// A digest mismatch below still deletes, because that partial is known-bad.
		return "", fmt.Errorf("downloading %s (%s fetched; re-run to resume): %w", f.Path, humanBytes(pw.done), copyErr)
	}
	if closeErr != nil {
		return "", closeErr
	}
	if f.SHA256 != "" {
		if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256 {
			os.Remove(part)
			return "", fmt.Errorf("checksum mismatch for %s:\n  want %s (declared by HuggingFace)\n  got  %s\nthe download was corrupted or the file changed upstream; nothing was written", f.Path, f.SHA256, got)
		}
	}
	if err := os.Rename(part, final); err != nil {
		return "", err
	}
	return final, nil
}

// hashPrefix feeds path's current contents into h and returns how many bytes it covered.
func hashPrefix(h io.Writer, path string) (int64, error) {
	fh, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer fh.Close()
	return io.Copy(h, fh)
}

func fileSHA256(path string) (string, error) {
	fh, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	h := sha256.New()
	if _, err := io.Copy(h, fh); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// progressWriter reports throughput on a TIME ticker rather than every N bytes, so the line
// updates about once a second whatever the transfer rate — the same reasoning behind this
// repo's long-running-test heartbeat rule: a multi-GB pull runs for minutes, and output that
// stalls must be distinguishable from a process that has.
type progressWriter struct {
	w      io.Writer
	done   int64
	total  int64
	report func(done, total int64)
	last   time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.report != nil && time.Since(p.last) >= time.Second {
		p.last = time.Now()
		p.report(p.done, p.total)
	}
	return n, err
}

func humanBytes(n int64) string {
	const u = 1024
	if n < u {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(u), 0
	for m := n / u; m >= u; m /= u {
		div *= u
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// HumanBytes is humanBytes for callers rendering their own progress line.
func HumanBytes(n int64) string { return humanBytes(n) }
