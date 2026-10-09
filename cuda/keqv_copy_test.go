//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// R-23 (docs/tasks/task-recompute-audit.md §5): on a K=V layer (Gemma 4 global layers, attention_k_eq_v) the resident decode and prefill projected k TWICE, once into kB and
// once into vB, on identical inputs. They now project once and copy kB into vB on the same stream (a launch of the kv_store kernel at pos 0, so it is capturable under CUDA graphs like
// every other launch of the segment). Dropping a projection must change no bit, and this gate says so: for each fixture it hashes every decode logit and every greedy token, and the
// batched-prefill logits, with CUDA graphs off and on, and compares them with the hashes recorded from the code BEFORE the change (testdata/keqv_copy_baseline.json).
//
// The baseline is a device-specific bit pattern (the PTX is JIT-compiled to the card's own SASS), so it is compared only on the device and driver that recorded it, and the
// device-independent half always runs: graphs on equals graphs off, and every fixture must have a K=V layer, or the gate would pass without touching the branch it is about.
//
//	go test -tags 'cuda goinfer_testhooks' -run TestKEqVCopy -v ./cuda/
//	GOINFER_KEQV_RECORD=1 go test ...   # (re)writes the baseline: do it only from code whose K=V path is the reference
const keqvBaselinePath = "testdata/keqv_copy_baseline.json"

type keqvRun struct {
	Decode  string `json:"decode_logits_sha256"`  // every position's logits, then the greedy continuation's
	Tokens  []int  `json:"greedy_tokens"`         // the continuation, token by token
	Prefill string `json:"prefill_logits_sha256"` // PrefillLast over the prompt, then the same greedy continuation by decode
	PTokens []int  `json:"prefill_greedy_tokens"`
}

type keqvBaseline struct {
	Device  string                        `json:"device"`
	Driver  string                        `json:"driver"`
	Commit  string                        `json:"recorded_at_commit"`
	Note    string                        `json:"note"`
	Fixture map[string]map[string]keqvRun `json:"fixtures"` // fixture -> "graphs=off|on" -> run
}

func gpuIdentity(t *testing.T) (name, driver string) {
	t.Helper()
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader").Output()
	if err != nil {
		return "unknown", "unknown"
	}
	f := strings.SplitN(strings.TrimSpace(strings.Split(string(out), "\n")[0]), ",", 2)
	if len(f) != 2 {
		return "unknown", "unknown"
	}
	return strings.TrimSpace(f[0]), strings.TrimSpace(f[1])
}

func hashLogits(h interface{ Write([]byte) (int, error) }, l []float32) {
	var b [4]byte
	for _, v := range l {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
}

func keqvArgmax(l []float32) int {
	best := 0
	for i, v := range l {
		if v > l[best] {
			best = i
		}
	}
	return best
}

// keqvRunFixture loads dir resident on CUDA (graphs per the flag) and returns the hashes. ok=false means the fixture does not go resident here (reported, not a pass).
func keqvRunFixture(t *testing.T, dir string, graphs bool) (run keqvRun, kEqVLayers int, ok bool) {
	t.Helper()
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "")
	if graphs {
		t.Setenv("GOINFER_CUDA_GRAPHS", "1")
		t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1") // this box is in DEFAULT compute mode, where graphs are otherwise refused
	} else {
		t.Setenv("GOINFER_CUDA_GRAPHS", "")
		t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "")
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	t.Cleanup(func() { mc.Close() })
	rf, isCUDA := mc.ResidentForwardForTest().(*cudaResident)
	if !isCUDA {
		return keqvRun{}, 0, false
	}
	if graphs && !rf.graphs {
		t.Fatalf("%s: graphs were requested and not enabled, so the graphs-on arm would test nothing", dir)
	}
	if !graphs && rf.graphs {
		t.Fatalf("%s: graphs are on in the graphs-off arm", dir)
	}
	for l := range rf.layers {
		if rf.layers[l].kEqV {
			kEqVLayers++
		}
	}

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200, 4, 33, 91, 120} // 20 tokens, past the fixtures' sliding windows
	const gen = 8
	embed := func(id int) []float32 { return mc.EmbedResidentForTest(id) }

	// Run A: every prompt token through decode, then a greedy continuation through decode.
	rf.Reset()
	h := sha256.New()
	var logits []float32
	for i, tok := range prompt {
		if logits, err = rf.Forward(embed(tok), i); err != nil {
			t.Fatalf("%s decode pos %d: %v", dir, i, err)
		}
		hashLogits(h, logits)
	}
	pos := len(prompt)
	for range gen {
		tok := keqvArgmax(logits)
		run.Tokens = append(run.Tokens, tok)
		if logits, err = rf.Forward(embed(tok), pos); err != nil {
			t.Fatalf("%s generate pos %d: %v", dir, pos, err)
		}
		hashLogits(h, logits)
		pos++
	}
	run.Decode = hex.EncodeToString(h.Sum(nil))

	// Run B: the prompt through the BATCHED prefill (the other half of R-23), then the same greedy continuation through decode.
	if batched, why := rf.PrefillPath(); !batched {
		t.Logf("%s: batched prefill declines (%s); run B is not available here", dir, why)
		return run, kEqVLayers, true
	}
	embs := make([][]float32, len(prompt))
	for i, tok := range prompt {
		embs[i] = embed(tok)
	}
	rf.Reset()
	hp := sha256.New()
	if logits, err = rf.PrefillLast(context.Background(), embs, 0); err != nil {
		t.Fatalf("%s PrefillLast: %v", dir, err)
	}
	hashLogits(hp, logits)
	pos = len(prompt)
	for range gen {
		tok := keqvArgmax(logits)
		run.PTokens = append(run.PTokens, tok)
		if logits, err = rf.Forward(embed(tok), pos); err != nil {
			t.Fatalf("%s prefill-then-generate pos %d: %v", dir, pos, err)
		}
		hashLogits(hp, logits)
		pos++
	}
	run.Prefill = hex.EncodeToString(hp.Sum(nil))
	return run, kEqVLayers, true
}

func TestKEqVCopy_bitIdenticalToTheDoubleProjection(t *testing.T) {
	fixtures := []string{"gemma4-dense-scaled", "gemma4-moe-scaled", "gemma4-moe-kv-tiny"}
	dev, drv := gpuIdentity(t)
	record := os.Getenv("GOINFER_KEQV_RECORD") != ""
	var base keqvBaseline
	haveBase := false
	if b, err := os.ReadFile(keqvBaselinePath); err == nil && !record {
		if err := json.Unmarshal(b, &base); err != nil {
			t.Fatalf("parse %s: %v", keqvBaselinePath, err)
		}
		haveBase = true
	}
	out := keqvBaseline{Device: dev, Driver: drv, Fixture: map[string]map[string]keqvRun{},
		Note: "Hashes of every decode logit and greedy token, and of PrefillLast, on a CUDA resident with K=V layers, recorded BEFORE R-23 replaced the second k projection with a copy. Device-specific: compared only on the same device and driver."}
	if c, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output(); err == nil {
		out.Commit = strings.TrimSpace(string(c))
	}
	compared, ran := 0, 0
	for _, fx := range fixtures {
		results := map[string]keqvRun{}
		for _, graphs := range []bool{false, true} {
			name := map[bool]string{false: "graphs=off", true: "graphs=on"}[graphs]
			run, nK, ok := keqvRunFixture(t, "../testdata/"+fx, graphs)
			if !ok {
				t.Logf("%s (%s): NOT CUDA-resident here, so it cannot exercise the resident K=V branch and is not a gate", fx, name)
				continue
			}
			if nK == 0 {
				t.Fatalf("%s has no K=V layer on the resident: this gate would pass without touching the branch it is about", fx)
			}
			ran++
			t.Logf("%s %s: %d K=V layers; decode %s.. tokens %v; prefill %s.. tokens %v", fx, name, nK, run.Decode[:12], run.Tokens, run.Prefill[:min(12, len(run.Prefill))], run.PTokens)
			results[name] = run
		}
		if len(results) == 2 { // device-independent: replaying the captured graph must equal re-issuing the launches
			if a, b := results["graphs=off"], results["graphs=on"]; a.Decode != b.Decode || a.Prefill != b.Prefill || fmt.Sprint(a.Tokens) != fmt.Sprint(b.Tokens) {
				t.Errorf("%s: graphs on differs from graphs off (decode %s vs %s, prefill %s vs %s)", fx, a.Decode[:12], b.Decode[:12], a.Prefill, b.Prefill)
			}
		}
		out.Fixture[fx] = results
		if haveBase && base.Device == dev && base.Driver == drv {
			for name, got := range results {
				want, has := base.Fixture[fx][name]
				if !has {
					t.Errorf("%s %s: no baseline recorded", fx, name)
					continue
				}
				compared++
				if got.Decode != want.Decode || got.Prefill != want.Prefill || fmt.Sprint(got.Tokens) != fmt.Sprint(want.Tokens) || fmt.Sprint(got.PTokens) != fmt.Sprint(want.PTokens) {
					t.Errorf("%s %s DIFFERS from the pre-change baseline recorded at %s:\n  decode  %s vs %s\n  prefill %s vs %s\n  tokens  %v vs %v / %v vs %v",
						fx, name, base.Commit, got.Decode, want.Decode, got.Prefill, want.Prefill, got.Tokens, want.Tokens, got.PTokens, want.PTokens)
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no fixture ran resident: nothing was checked")
	}
	switch {
	case record:
		b, _ := json.MarshalIndent(out, "", " ")
		if err := os.WriteFile(keqvBaselinePath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("recorded %s on %s, driver %s", keqvBaselinePath, dev, drv)
	case !haveBase:
		t.Errorf("no %s: record it from the pre-change code with GOINFER_KEQV_RECORD=1", keqvBaselinePath)
	case base.Device != dev || base.Driver != drv:
		t.Logf("baseline is for %s / %s, this is %s / %s: the bit comparison was NOT made (only graphs on == graphs off was)", base.Device, base.Driver, dev, drv)
	case compared == 0:
		t.Fatal("the baseline matched this device but nothing was compared")
	default:
		t.Logf("compared %d runs with the baseline recorded at %s: bit-identical", compared, base.Commit)
	}
}
