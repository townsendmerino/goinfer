//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The batched prefill's tails (R-24, docs/tasks/task-recompute-audit.md §5): only ResidualAll reads the [M, hidden]
// residual back to the host; the argmax and all-logits heads read it on the device, and the last-row tails upload one
// row. This gate hashes the output of EVERY tail mode the resident exposes, per fixture, and compares it with the hashes
// recorded from the code before the residual download was dropped (testdata/prefill_tails_baseline.json). It is
// device-specific for the same reason keqv_copy_baseline.json is (the PTX is JIT-compiled to the card's own SASS), so the
// baseline is compared only on the device and driver that recorded it.
//
//	go test -tags 'cuda goinfer_testhooks' -run TestPrefillTails -v ./cuda/
//	GOINFER_TAILS_RECORD=1 go test ...   # (re)writes the baseline: only from code whose tails are the reference
const tailsBaselinePath = "testdata/prefill_tails_baseline.json"

type tailsBaseline struct {
	Device  string                       `json:"device"`
	Driver  string                       `json:"driver"`
	Commit  string                       `json:"recorded_at_commit"`
	Note    string                       `json:"note"`
	Fixture map[string]map[string]string `json:"fixtures"` // fixture -> tail -> sha256 of its output (and the greedy continuation where it has one)
}

// tailsRun returns tail -> hash for one fixture. ok=false: the fixture is not CUDA-resident or has no batched prefill here, so it is not a gate.
func tailsRun(t *testing.T, dir string, moeCache bool) (res map[string]string, ok bool) {
	t.Helper()
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	if moeCache {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
		t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3")
	} else {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "")
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	t.Cleanup(func() { mc.Close() })
	rf, isCUDA := mc.ResidentForwardForTest().(*cudaResident)
	if !isCUDA {
		return nil, false
	}
	if batched, why := rf.PrefillPath(); !batched {
		t.Logf("%s: batched prefill declines (%s)", dir, why)
		return nil, false
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200, 4, 33, 91, 120}
	embs := make([][]float32, len(prompt))
	for i, tok := range prompt {
		embs[i] = mc.EmbedResidentForTest(tok)
	}
	embed := func(id int) []float32 { return mc.EmbedResidentForTest(id) }
	ctx := context.Background()
	res = map[string]string{}
	hex256 := func(h interface{ Sum([]byte) []byte }) string { return hex.EncodeToString(h.Sum(nil)) }

	// last-row logits, then a greedy continuation through decode: the continuation proves the device state the tail leaves behind (r.x, the KV) is the same.
	rf.Reset()
	h := sha256.New()
	logits, err := rf.PrefillLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("%s PrefillLast: %v", dir, err)
	}
	hashLogits(h, logits)
	pos := len(prompt)
	for range 4 {
		tok := keqvArgmax(logits)
		fmt.Fprintf(h, "tok%d", tok)
		if logits, err = rf.Forward(embed(tok), pos); err != nil {
			t.Fatalf("%s decode after PrefillLast: %v", dir, err)
		}
		hashLogits(h, logits)
		pos++
	}
	res["PrefillLast+decode"] = hex256(h)

	rf.Reset()
	rows, err := rf.PrefillLastN(embs, 0)
	if err != nil {
		t.Fatalf("%s PrefillLastN: %v", dir, err)
	}
	h = sha256.New()
	for _, r := range rows {
		hashLogits(h, r)
	}
	res["PrefillLastN"] = hex256(h)

	rf.Reset()
	ids, err := rf.PrefillLastNArgmax(embs, 0)
	if err != nil {
		t.Fatalf("%s PrefillLastNArgmax: %v", dir, err)
	}
	res["PrefillLastNArgmax"] = fmt.Sprint(ids)

	rf.Reset()
	hid, err := rf.HiddenLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("%s HiddenLast: %v", dir, err)
	}
	h = sha256.New()
	hashLogits(h, hid)
	res["HiddenLast"] = hex256(h)

	rf.Reset()
	resid, err := rf.ResidualAll(ctx, embs, 0)
	if err != nil {
		t.Fatalf("%s ResidualAll: %v", dir, err)
	}
	h = sha256.New()
	for _, r := range resid {
		hashLogits(h, r)
	}
	res["ResidualAll"] = hex256(h)
	return res, true
}

func TestPrefillTails_bitIdenticalToTheRecordedBaseline(t *testing.T) {
	fixtures := []struct {
		dir      string
		moeCache bool
	}{{"gemma4-dense-scaled", false}, {"gemma4-moe-scaled", false}, {"qwen3moe-tiny-k3", true}}
	dev, drv := gpuIdentity(t)
	record := os.Getenv("GOINFER_TAILS_RECORD") != ""
	var base tailsBaseline
	haveBase := false
	if b, err := os.ReadFile(tailsBaselinePath); err == nil && !record {
		if err := json.Unmarshal(b, &base); err != nil {
			t.Fatalf("parse %s: %v", tailsBaselinePath, err)
		}
		haveBase = true
	}
	out := tailsBaseline{Device: dev, Driver: drv, Fixture: map[string]map[string]string{},
		Note: "Hashes of every batched-prefill tail mode's output (PrefillLast plus a greedy decode continuation, PrefillLastN, PrefillLastNArgmax, HiddenLast, ResidualAll), recorded BEFORE R-24 stopped downloading the full residual on the tails that do not read it. Device-specific: compared only on this GPU and driver."}
	if c, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output(); err == nil {
		out.Commit = strings.TrimSpace(string(c))
	}
	ran, compared := 0, 0
	for _, fx := range fixtures {
		got, ok := tailsRun(t, "../testdata/"+fx.dir, fx.moeCache)
		if !ok {
			t.Logf("%s: not CUDA-resident or no batched prefill here, so it is not a gate", fx.dir)
			continue
		}
		ran++
		out.Fixture[fx.dir] = got
		for tail, hsh := range got {
			t.Logf("%s %-20s %.16s", fx.dir, tail, hsh)
		}
		if haveBase && base.Device == dev && base.Driver == drv {
			for tail, hsh := range got {
				want, has := base.Fixture[fx.dir][tail]
				if !has {
					t.Errorf("%s %s: no baseline recorded", fx.dir, tail)
					continue
				}
				compared++
				if hsh != want {
					t.Errorf("%s %s DIFFERS from the pre-change baseline recorded at %s: %s vs %s", fx.dir, tail, base.Commit, hsh, want)
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no fixture ran with a batched prefill: nothing was checked")
	}
	switch {
	case record:
		b, _ := json.MarshalIndent(out, "", " ")
		if err := os.WriteFile(tailsBaselinePath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("recorded %s on %s, driver %s, from %d fixtures", tailsBaselinePath, dev, drv, ran)
	case !haveBase:
		t.Errorf("no %s: record it from the pre-change code with GOINFER_TAILS_RECORD=1", tailsBaselinePath)
	case base.Device != dev || base.Driver != drv:
		t.Logf("baseline is for %s / %s, this is %s / %s: the bit comparison was NOT made", base.Device, base.Driver, dev, drv)
	case compared == 0:
		t.Fatal("the baseline matched this device but nothing was compared")
	default:
		t.Logf("compared %d tail outputs with the baseline recorded at %s: bit-identical", compared, base.Commit)
	}
}
