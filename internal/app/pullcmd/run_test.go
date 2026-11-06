package pullcmd

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// hubFile is one file the fake hub serves: its bytes, and the sha256 its tree listing declares ("" for none, as HF lists a
// small non-LFS file).
type hubFile struct {
	body   []byte
	sha256 string
}

// hub is a local stand-in for HuggingFace: per repo, its gating and its files. It records every path asked of it.
type hub struct {
	gated map[string]string // repo -> "manual" for a gated one
	repos map[string]map[string]hubFile

	mu   sync.Mutex
	asks []string
}

func (h *hub) asked(prefix string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, a := range h.asks {
		if strings.HasPrefix(a, prefix) {
			return true
		}
	}
	return false
}

func (h *hub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	h.asks = append(h.asks, r.URL.Path)
	h.mu.Unlock()
	if rest, ok := strings.CutPrefix(r.URL.Path, "/api/models/"); ok {
		repo, tree := strings.CutSuffix(rest, "/tree/main")
		files, known := h.repos[repo]
		if !known {
			http.NotFound(w, r)
			return
		}
		if !tree {
			gated := any(false)
			if g := h.gated[repo]; g != "" {
				gated = g
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"gated": gated, "private": false})
			return
		}
		var entries []map[string]any
		for path, f := range files {
			e := map[string]any{"type": "file", "path": path, "size": len(f.body)}
			if f.sha256 != "" {
				e["lfs"] = map[string]string{"oid": f.sha256}
			}
			entries = append(entries, e)
		}
		_ = json.NewEncoder(w).Encode(entries)
		return
	}
	for repo, files := range h.repos {
		if path, ok := strings.CutPrefix(r.URL.Path, "/"+repo+"/resolve/main/"); ok {
			if f, ok := files[path]; ok {
				_, _ = w.Write(f.body)
				return
			}
		}
	}
	http.NotFound(w, r)
}

// toHub sends every request to huggingface.co to h and fails the test on any other host, so nothing here reaches the network.
type toHub struct {
	t    *testing.T
	base string
}

func (rt toHub) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Host != "huggingface.co" {
		rt.t.Errorf("a request to %s, outside the test hub", r.URL)
		return nil, errors.New("outside the test hub")
	}
	r2 := r.Clone(r.Context())
	r2.URL.Scheme, r2.URL.Host, r2.Host = "http", rt.base, ""
	return http.DefaultTransport.RoundTrip(r2)
}

// withHub serves h for the test's HTTP and gives the user cache dir a temp home. Tests that call it are not parallel: it
// replaces http.DefaultClient's transport, which pull uses.
func withHub(t *testing.T, h *hub) {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	prev := http.DefaultClient.Transport
	http.DefaultClient.Transport = toHub{t: t, base: strings.TrimPrefix(srv.URL, "http://")}
	t.Cleanup(func() { http.DefaultClient.Transport = prev })
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	t.Setenv("LocalAppData", filepath.Join(home, "AppData", "Local"))
}

// run calls f with stdout and stderr captured, and returns its exit code and what each stream got.
func run(t *testing.T, f func([]string) int, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	dir := t.TempDir()
	out, err := os.Create(filepath.Join(dir, "stdout"))
	if err != nil {
		t.Fatal(err)
	}
	errf, err := os.Create(filepath.Join(dir, "stderr"))
	if err != nil {
		t.Fatal(err)
	}
	prevOut, prevErr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = out, errf
	code = f(args)
	os.Stdout, os.Stderr = prevOut, prevErr
	out.Close()
	errf.Close()
	o, _ := os.ReadFile(out.Name())
	e, _ := os.ReadFile(errf.Name())
	return code, string(o), string(e)
}

func digest(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// A repo with two quants and a README: the bare ref lists the two .gguf files, with how to fetch one, and exits 0.
func TestRun_listsARepo(t *testing.T) {
	q4, q8 := []byte("q4 weights"), []byte("q8 weights, longer")
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": {
		"m-Q4_K_M.gguf": {q4, digest(q4)}, "m-Q8_0.gguf": {q8, digest(q8)}, "README.md": {[]byte("hi"), ""},
	}}})
	code, stdout, _ := run(t, Run, "o/r")
	if code != 0 || !strings.Contains(stdout, "o/r — 2 GGUF file(s):") || !strings.Contains(stdout, "m-Q4_K_M.gguf") ||
		!strings.Contains(stdout, "m-Q8_0.gguf") || strings.Contains(stdout, "README") || !strings.Contains(stdout, "fetch one with:") {
		t.Errorf("exit %d, stdout:\n%s", code, stdout)
	}
}

// A quant selector fetches that file into -o, verified against the declared sha256: the bytes land under the file's name, no
// .part is left, and the summary says it was verified and how to run it.
func TestRun_fetchesAndVerifiesAQuant(t *testing.T) {
	q4 := []byte("q4 weights")
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": {"m-Q4_K_M.gguf": {q4, digest(q4)}, "m-Q8_0.gguf": {[]byte("q8"), digest([]byte("q8"))}}}})
	dir := t.TempDir()
	code, stdout, stderr := run(t, Run, "o/r:q4_k_m", "-o", dir)
	got, err := os.ReadFile(filepath.Join(dir, "m-Q4_K_M.gguf"))
	if code != 0 || err != nil || string(got) != string(q4) {
		t.Fatalf("exit %d, file %q (%v); stderr:\n%s", code, got, err, stderr)
	}
	if _, err := os.Stat(filepath.Join(dir, "m-Q4_K_M.gguf.part")); err == nil {
		t.Error("the .part file was left behind")
	}
	if _, err := os.Stat(filepath.Join(dir, "m-Q8_0.gguf")); err == nil {
		t.Error("the other quant was fetched too")
	}
	if !strings.Contains(stdout, "sha256 verified") || !strings.Contains(stdout, "--model "+filepath.Join(dir, "m-Q4_K_M.gguf")) {
		t.Errorf("stdout:\n%s", stdout)
	}
}

// A file whose bytes do not match the declared sha256 is refused, exit 1, and nothing is kept: no file, no .part.
func TestRun_refusesADigestMismatch(t *testing.T) {
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": {"m-Q4_K_M.gguf": {[]byte("served"), digest([]byte("declared"))}}}})
	dir := t.TempDir()
	code, _, stderr := run(t, Run, "o/r:q4_k_m", "-o", dir)
	if code != 1 || !strings.Contains(stderr, "checksum mismatch") {
		t.Fatalf("exit %d, stderr:\n%s", code, stderr)
	}
	for _, name := range []string{"m-Q4_K_M.gguf", "m-Q4_K_M.gguf.part"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			t.Errorf("%s was kept after the mismatch", name)
		}
	}
}

// A file HuggingFace declares no sha256 for is fetched, and the summary says it was NOT verified rather than implying it was.
func TestRun_saysWhenNothingWasVerified(t *testing.T) {
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": {"m-Q4_K_M.gguf": {[]byte("small"), ""}}}})
	code, stdout, _ := run(t, Run, "o/r:q4_k_m", "-o", t.TempDir())
	if code != 0 || !strings.Contains(stdout, "NOT verified") || strings.Contains(stdout, "sha256 verified") {
		t.Errorf("exit %d, stdout:\n%s", code, stdout)
	}
}

// A gated repo fails at once, exit 1, naming the gate, before the listing or any byte is asked for.
func TestRun_gatedRepoFailsBeforeAnyTransfer(t *testing.T) {
	h := &hub{gated: map[string]string{"o/r": "manual"}, repos: map[string]map[string]hubFile{"o/r": {"m-Q4_K_M.gguf": {[]byte("w"), digest([]byte("w"))}}}}
	withHub(t, h)
	code, _, stderr := run(t, Run, "o/r:q4_k_m", "-o", t.TempDir())
	if code != 1 || !strings.Contains(stderr, "gated (manual)") {
		t.Errorf("exit %d, stderr:\n%s", code, stderr)
	}
	if h.asked("/api/models/o/r/tree") || h.asked("/o/r/resolve/") {
		t.Errorf("a gated repo was listed or fetched: %v", h.asks)
	}
}

// -embed takes one GGUF file, so a split quant is refused before any shard is fetched. PATH is emptied so that, were the
// refusal missing, -embed would stop at finding no go toolchain rather than run the embed build.
func TestRun_embedRefusesASplitSetBeforeFetching(t *testing.T) {
	s1, s2 := []byte("shard one"), []byte("shard two")
	h := &hub{repos: map[string]map[string]hubFile{"o/r": {
		"Q8_0/m-Q8_0-00001-of-00002.gguf": {s1, digest(s1)}, "Q8_0/m-Q8_0-00002-of-00002.gguf": {s2, digest(s2)},
	}}}
	withHub(t, h)
	t.Setenv("PATH", t.TempDir())
	code, _, stderr := run(t, Run, "o/r:q8_0", "-embed", "-o", t.TempDir())
	if code != 1 || !strings.Contains(stderr, "split set") {
		t.Errorf("exit %d, stderr:\n%s", code, stderr)
	}
	if h.asked("/o/r/resolve/") {
		t.Errorf("a shard was fetched: %v", h.asks)
	}
}

// Malformed command lines exit 2 before any request: no ref, an os/arch target without -embed, and an unknown selector's
// listing is not reached either.
func TestRun_commandLineErrorsExit2WithoutARequest(t *testing.T) {
	h := &hub{repos: map[string]map[string]hubFile{}}
	withHub(t, h)
	for _, c := range []struct {
		args []string
		want string
	}{
		{nil, "pull <owner/repo>"},
		{[]string{"o/r:q4_k_m", "linux/amd64"}, "only meaningful with -embed"},
	} {
		code, _, stderr := run(t, Run, c.args...)
		if code != 2 || !strings.Contains(stderr, c.want) {
			t.Errorf("Run(%q): exit %d, stderr:\n%s", c.args, code, stderr)
		}
	}
	if len(h.asks) != 0 {
		t.Errorf("a malformed command line reached the hub: %v", h.asks)
	}
}

// A pull with no -o lands in the user cache, and the cache command then lists it, complete, with the path to pass to --model.
func TestRun_pullLandsInTheCacheThatCacheLists(t *testing.T) {
	q4 := []byte("q4 weights")
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": {"m-Q4_K_M.gguf": {q4, digest(q4)}}}})
	if code, stdout, _ := run(t, RunCache); code != 0 || !strings.Contains(stdout, "holds no models") {
		t.Fatalf("an empty cache: exit %d, stdout:\n%s", code, stdout)
	}
	if code, _, stderr := run(t, Run, "o/r:q4_k_m"); code != 0 {
		t.Fatalf("pull: exit %d, stderr:\n%s", code, stderr)
	}
	code, stdout, _ := run(t, RunCache)
	if code != 0 || !strings.Contains(stdout, "o/r") || !strings.Contains(stdout, "m-Q4_K_M.gguf") || strings.Contains(stdout, "INCOMPLETE") ||
		!strings.Contains(stdout, "1 model(s)") {
		t.Errorf("cache after the pull: exit %d, stdout:\n%s", code, stdout)
	}
	if code, _, stderr := run(t, RunCache, "extra"); code != 2 || !strings.Contains(stderr, "takes no arguments") {
		t.Errorf("cache with an argument: exit %d, stderr:\n%s", code, stderr)
	}
}

// ckptRepo is a llama safetensors checkpoint whose weights serve weight and declare declared.
func ckptRepo(weight []byte, declared string) map[string]hubFile {
	cfg := []byte(`{"model_type":"llama","torch_dtype":"bfloat16"}`)
	tok := []byte(`{"model":{}}`)
	return map[string]hubFile{
		"config.json": {cfg, ""}, "tokenizer.json": {tok, ""}, "model.safetensors": {weight, declared}, "README.md": {[]byte("hi"), ""},
	}
}

// :safetensors fetches the repo's checkpoint as one directory: config, tokenizer and weights land at -o with their bytes, the
// weights verified, the README left out, and no staging directory left behind.
func TestRun_fetchesASafetensorsCheckpointAsOneVerifiedSet(t *testing.T) {
	weight := []byte("safetensors weights")
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": ckptRepo(weight, digest(weight))}})
	dest := filepath.Join(t.TempDir(), "ckpt")
	code, stdout, stderr := run(t, Run, "o/r:safetensors", "-o", dest)
	if code != 0 || !strings.Contains(stdout, "every weight file sha256 verified") {
		t.Fatalf("exit %d, stdout:\n%s\nstderr:\n%s", code, stdout, stderr)
	}
	for name, want := range map[string]string{"model.safetensors": string(weight), "config.json": `{"model_type":"llama","torch_dtype":"bfloat16"}`, "tokenizer.json": `{"model":{}}`} {
		if got, err := os.ReadFile(filepath.Join(dest, name)); err != nil || string(got) != want {
			t.Errorf("%s: %q (%v), want %q", name, got, err, want)
		}
	}
	if _, err := os.Stat(filepath.Join(dest, "README.md")); err == nil {
		t.Error("the README was fetched into the checkpoint")
	}
	if _, err := os.Stat(dest + ".partial"); err == nil {
		t.Error("the staging directory was left behind")
	}
}

// A checkpoint whose weights fail their sha256 publishes nothing at -o, and exits 1.
func TestRun_checkpointWithABadWeightPublishesNothing(t *testing.T) {
	withHub(t, &hub{repos: map[string]map[string]hubFile{"o/r": ckptRepo([]byte("served"), digest([]byte("declared")))}})
	dest := filepath.Join(t.TempDir(), "ckpt")
	code, _, stderr := run(t, Run, "o/r:safetensors", "-o", dest)
	if code != 1 || !strings.Contains(stderr, "checksum mismatch") {
		t.Errorf("exit %d, stderr:\n%s", code, stderr)
	}
	if _, err := os.Stat(dest); err == nil {
		t.Error("a checkpoint whose weights failed verification was published at -o")
	}
}

// A checkpoint of a family this build does not load is refused from its config, exit 1, before any weight byte is asked for.
func TestRun_checkpointOfAnUnknownFamilyFetchesNoWeights(t *testing.T) {
	files := ckptRepo([]byte("w"), digest([]byte("w")))
	files["config.json"] = hubFile{[]byte(`{"model_type":"nonesuch"}`), ""}
	h := &hub{repos: map[string]map[string]hubFile{"o/r": files}}
	withHub(t, h)
	code, _, stderr := run(t, Run, "o/r:safetensors", "-o", filepath.Join(t.TempDir(), "ckpt"))
	if code != 1 || !strings.Contains(stderr, "nothing was downloaded") {
		t.Errorf("exit %d, stderr:\n%s", code, stderr)
	}
	if h.asked("/o/r/resolve/main/model.safetensors") {
		t.Errorf("the weights were asked for: %v", h.asks)
	}
}

// A repo with no .gguf lists its safetensors plan, with how to fetch it, and exits 0; -embed refuses a checkpoint, exit 2,
// since it bakes one GGUF file.
func TestRun_listsASafetensorsRepoAndEmbedRefusesIt(t *testing.T) {
	weight := []byte("w")
	h := &hub{repos: map[string]map[string]hubFile{"o/r": ckptRepo(weight, digest(weight))}}
	withHub(t, h)
	code, stdout, _ := run(t, Run, "o/r")
	if code != 0 || !strings.Contains(stdout, "safetensors checkpoint, model_type llama") || !strings.Contains(stdout, "model.safetensors") ||
		!strings.Contains(stdout, "o/r:safetensors") {
		t.Errorf("listing: exit %d, stdout:\n%s", code, stdout)
	}
	code, _, stderr := run(t, Run, "o/r:safetensors", "-embed")
	if code != 2 || !strings.Contains(stderr, "-embed bakes one GGUF file") {
		t.Errorf("-embed on a checkpoint: exit %d, stderr:\n%s", code, stderr)
	}
	if h.asked("/o/r/resolve/main/model.safetensors") {
		t.Errorf("the weights were asked for: %v", h.asks)
	}
}
