package serveapp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/pull"
)

// tinyCheckpointFiles is testdata/llama-tiny as a checkpoint set, with the file list a plan would carry: the weights
// with their sha256 (HF's LFS oid), the small files by size only. The fixture commits no tokenizer and serve's loader
// requires one, so the set gains tinyChatTokenizer's tokenizer.json, whose vocabulary sits inside the tiny model's 256 ids.
func tinyCheckpointFiles(t *testing.T) (content map[string][]byte, files []pull.File) {
	t.Helper()
	content = map[string][]byte{}
	src := filepath.Join("..", "..", "testdata", "llama-tiny")
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Skipf("no committed tiny fixture: %v", err)
		}
		content[name] = b
	}
	content["tokenizer.json"] = tinyChatTokenizer(t)
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors", "tokenizer.json"} {
		b := content[name]
		f := pull.File{Path: name, Size: int64(len(b))}
		if strings.HasSuffix(name, ".safetensors") {
			s := sha256.Sum256(b)
			f.SHA256 = hex.EncodeToString(s[:])
		}
		files = append(files, f)
	}
	return content, files
}

// writeCheckpoint lays down a complete checkpoint at dir the way pull.DownloadCheckpoint publishes one: the files, then
// the marker naming them. The marker's shape is pull's (.goinfer-checkpoint.json, {"repo","files"}); if it drifts, the
// load gates below refuse the directory and fail loudly, they do not pass on a stale copy.
func writeCheckpoint(t *testing.T, dir, repo string) {
	t.Helper()
	content, files := tinyCheckpointFiles(t)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(dir, f.Path), content[f.Path], 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mb, err := json.Marshal(map[string]any{"repo": repo, "files": files})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".goinfer-checkpoint.json"), mb, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestWebLoadPath_checkpointDir is P6's half of W5's confinement: a directory loads only when it is a complete
// checkpoint the pull flow published. Each refusal is a directory that could otherwise reach the loader looking like a
// model: one with no marker, one still being assembled, one whose weights no longer match the marker.
func TestWebLoadPath_checkpointDir(t *testing.T) {
	_, root := fakeCache(t)
	good := filepath.Join(root, "owner", "Qwen2.5-0.5B-Instruct")
	writeCheckpoint(t, good, "owner/Qwen2.5-0.5B-Instruct")
	bare := filepath.Join(root, "owner", "bare")
	if err := os.MkdirAll(bare, 0o755); err != nil {
		t.Fatal(err)
	}
	staging := filepath.Join(root, "owner", "staged.partial")
	writeCheckpoint(t, staging, "owner/staged") // the instant between pull's marker write and its rename
	damaged := filepath.Join(root, "owner", "damaged")
	writeCheckpoint(t, damaged, "owner/damaged")
	if err := os.Truncate(filepath.Join(damaged, "model.safetensors"), 100); err != nil {
		t.Fatal(err)
	}
	wantGood, _ := filepath.EvalSymlinks(good)

	for _, c := range []struct{ name, path, want string }{
		{"a complete pulled checkpoint", good, wantGood},
		{"a directory with no marker", bare, ""},
		{"a staging directory", staging, ""},
		{"a checkpoint whose weights no longer verify", damaged, ""},
		{"the cache root itself", root, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := webLoadPath(c.path)
			if c.want == "" {
				if err == nil {
					t.Fatalf("webLoadPath(%q) = %q, want a refusal", c.path, got)
				}
				return
			}
			if err != nil || got != c.want {
				t.Fatalf("webLoadPath(%q) = %q, %v; want %q", c.path, got, err, c.want)
			}
		})
	}
	if got := webServedName(wantGood); got != "Qwen2.5-0.5B-Instruct" {
		t.Errorf("served name of a checkpoint dir = %q, want the repo's name whole", got)
	}
	if got := webServedName("/c/o/r/model-Q4_K_M.GGUF"); got != "model-Q4_K_M" {
		t.Errorf("served name of a gguf = %q, want the extension cut", got)
	}
	if got := webServedName("/c/o/r/big-Q8_0-00001-of-00003.gguf"); got != "big-Q8_0" {
		t.Errorf("served name of a split set = %q, want the model's, without the shard suffix", got)
	}
}

// fakeCheckpointRepo stands in HuggingFace for one safetensors-only repo: access granted, no GGUF files, a plan of
// llama-tiny's files, and a fetch that publishes them at dest. It returns the plan calls and fetch calls it saw.
func fakeCheckpointRepo(t *testing.T, repo string, planErr error) (plans, fetches *int) {
	t.Helper()
	_, files := tinyCheckpointFiles(t)
	var total int64
	for _, f := range files {
		total += f.Size
	}
	plan := pull.Plan{Repo: repo, Files: files, Bytes: total, ModelType: "llama", Family: "llama", Dtype: "float32"}
	var np, nf int
	oa, ol, op, of := webCheckAccess, webListFiles, webPlanCheckpoint, webFetchCheckpoint
	t.Cleanup(func() { webCheckAccess, webListFiles, webPlanCheckpoint, webFetchCheckpoint = oa, ol, op, of })
	webCheckAccess = func(context.Context, string) error { return nil }
	webListFiles = func(context.Context, string) ([]pull.File, error) { return nil, nil }
	webPlanCheckpoint = func(_ context.Context, r string) (pull.Plan, error) {
		np++
		if planErr != nil {
			return pull.Plan{}, planErr
		}
		if r != repo {
			return pull.Plan{}, fmt.Errorf("planned %q, want %q", r, repo)
		}
		return plan, nil
	}
	webFetchCheckpoint = func(_ context.Context, p pull.Plan, dest string, progress func(done, total int64, file string)) (string, error) {
		nf++
		var done int64
		for _, f := range p.Files {
			done += f.Size
			progress(done, p.Bytes, f.Path)
		}
		writeCheckpoint(t, dest, p.Repo)
		return dest, nil
	}
	return &np, &nf
}

func postWeb(t *testing.T, h http.HandlerFunc, path string, req webPullReq) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(req)
	w := httptest.NewRecorder()
	h(w, httptest.NewRequest(http.MethodPost, path, strings.NewReader(string(b))))
	return w
}

// TestWebCheckpoint_listPullLoad is P6 end to end on the page's three routes: a repo with no GGUF lists as its
// checkpoint plan, the pull streams the set into the repo's cache directory, and the load takes that directory with the
// real loader, serves it under the repo's name, and the loaded model decodes.
func TestWebCheckpoint_listPullLoad(t *testing.T) {
	_, _ = fakeCache(t)
	const repo = "owner/tiny-repo"
	plans, fetches := fakeCheckpointRepo(t, repo, nil)
	s, err := newServer(config{web: true})
	if err != nil {
		t.Fatal(err)
	}

	w := postWeb(t, s.handleWebList, "/web/models/list", webPullReq{Repo: repo})
	if w.Code != http.StatusOK {
		t.Fatalf("list: %d %s", w.Code, w.Body.String())
	}
	var list struct {
		Files      []any          `json:"files"`
		Checkpoint map[string]any `json:"checkpoint"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Files) != 0 || list.Checkpoint == nil || list.Checkpoint["files"] != float64(4) || list.Checkpoint["family"] != "llama" {
		t.Fatalf("list of a safetensors-only repo = %s, want no files and a 4-file llama checkpoint", w.Body.String())
	}
	if note, _ := list.Checkpoint["note"].(string); !strings.Contains(note, "4 files") || !strings.Contains(note, "quarter") {
		t.Errorf("plan note = %q, want the file count and the full-precision size warning", note)
	}

	w = postWeb(t, s.handleWebPull, "/web/models/pull", webPullReq{Repo: repo, Checkpoint: true})
	evs := readLoadEvents(t, w.Body)
	if len(evs) < 3 || evs[0].name != "start" || evs[len(evs)-1].name != "done" {
		t.Fatalf("pull events: %d %+v", w.Code, evs)
	}
	if f, _ := evs[1].data["file"].(string); evs[1].name != "progress" || f == "" {
		t.Errorf("first progress event = %+v, want it to name the file in flight", evs[1])
	}
	done := evs[len(evs)-1].data
	dir, _ := pull.CacheDir(repo)
	if done["path"] != dir || done["files"] != float64(4) || done["sha256_files"] != float64(1) {
		t.Fatalf("pull done = %+v, want path %s, 4 files, 1 with a sha256", done, dir)
	}
	if *plans != 2 || *fetches != 1 {
		t.Errorf("plans %d, fetches %d; want 2 (list, pull) and 1", *plans, *fetches)
	}

	w = postLoad(s, context.Background(), dir)
	evs = readLoadEvents(t, w.Body)
	if len(evs) == 0 || evs[len(evs)-1].name != "done" {
		t.Fatalf("load of the pulled checkpoint: %d %+v", w.Code, evs)
	}
	if id := evs[len(evs)-1].data["id"]; id != "tiny-repo" {
		t.Fatalf("served as %v, want the repo's name", id)
	}
	s.regMu.RLock()
	lm := s.models["tiny-repo"]
	s.regMu.RUnlock()
	if lm == nil || lm.model == nil {
		t.Fatal("the loaded checkpoint is not in the registry")
	}
	t.Cleanup(func() { lm.model.Close(); lm.closeEntryNatives() })
	out, g := lm.model.Generate(context.Background(), []int{1, 2, 3}, 1, decoder.SamplingParams{})
	n := 0
	for range out {
		n++
	}
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("the web-loaded checkpoint decoded %d tokens, want 1", n)
	}
}

// TestWebCheckpoint_listOffersOnlyWhenNoGGUF: a GGUF repo lists exactly as before (no plan is made, so its listing pays no
// extra HuggingFace round trips), and a plan that declines puts its reason where the offer would be.
func TestWebCheckpoint_listOffersOnlyWhenNoGGUF(t *testing.T) {
	_, _ = fakeCache(t)
	s := &server{}
	s.cfg.web = true

	plans, _ := fakeCheckpointRepo(t, "owner/gguf-repo", nil)
	webListFiles = func(context.Context, string) ([]pull.File, error) {
		return []pull.File{{Path: "m-q4_k_m.gguf", Size: 10}}, nil
	}
	w := postWeb(t, s.handleWebList, "/web/models/list", webPullReq{Repo: "owner/gguf-repo"})
	if w.Code != http.StatusOK || strings.Contains(w.Body.String(), "checkpoint") || *plans != 0 {
		t.Fatalf("GGUF repo list = %d %s with %d plans, want its files only and no plan", w.Code, w.Body.String(), *plans)
	}

	fakeCheckpointRepo(t, "owner/odd", errors.New(`model_type "mamba9" is not one this build loads from safetensors`))
	w = postWeb(t, s.handleWebList, "/web/models/list", webPullReq{Repo: "owner/odd"})
	var list map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &list)
	if msg, _ := list["checkpoint_error"].(string); w.Code != http.StatusOK || list["checkpoint"] != nil || !strings.Contains(msg, "mamba9") {
		t.Fatalf("declined plan list = %d %s, want the decline reason and no offer", w.Code, w.Body.String())
	}
	w = postWeb(t, s.handleWebPull, "/web/models/pull", webPullReq{Repo: "owner/odd", Checkpoint: true})
	evs := readLoadEvents(t, w.Body)
	if len(evs) != 1 || evs[0].name != "error" {
		t.Fatalf("pull of a declined checkpoint: %+v, want one error and no start", evs)
	}
}

// TestWebLoad_checkpointFitDecline is P6's "check the fit guard's decline path renders as well for a directory": a
// won't-fit refusal on a checkpoint directory reaches the page as an error event with the guard's own message and a
// 400, the same shape a refused .gguf gets.
func TestWebLoad_checkpointFitDecline(t *testing.T) {
	_, root := fakeCache(t)
	dir := filepath.Join(root, "owner", "big")
	writeCheckpoint(t, dir, "owner/big")
	orig := webLoadDecoder
	t.Cleanup(func() { webLoadDecoder = orig })
	var got string
	webLoadDecoder = func(_ context.Context, spec modelSpec, _ config) (*loadedModel, error) {
		got = spec.path
		return nil, fmt.Errorf("decoder: needs 9.1 GB resident, 4.0 GB free: %w", decoder.ErrWontFitResident)
	}
	s := &server{models: map[string]*loadedModel{}}
	s.cfg.web = true
	evs := readLoadEvents(t, postLoad(s, context.Background(), dir).Body)
	if len(evs) != 2 || evs[1].name != "error" {
		t.Fatalf("events %+v, want start then error", evs)
	}
	msg, _ := evs[1].data["message"].(string)
	if want, _ := filepath.EvalSymlinks(dir); got != want || !strings.Contains(msg, "9.1 GB resident") || evs[1].data["status"] != float64(http.StatusBadRequest) {
		t.Fatalf("loader got %q (want %q); error %+v", got, want, evs[1].data)
	}
}
