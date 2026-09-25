//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// S6's byte-identity gate over every resident-parity fixture, not just the two real checkpoints
// TestWeightAlias_logitsByteIdentical is pointed at. Each tiny fixture is transcoded to a metal-target
// .giw (weights format v14: fused q|k|v and gate|up groups, kind-7 singles, f16 scales), loaded twice
// with GOINFER_METAL_ALIAS off and on via Options.Knobs, and driven through the same tokens; every
// logit must match bit for bit. A fixture Metal does not build resident is skipped (named in the log),
// and one whose shapes leave nothing aliasable (cols not a multiple of 32) is compared anyway — the
// copy path must then be the only path — but at least one fixture must alias fused groups, kind-7
// singles, f16 scales and the int8 LM head, or this gate would be copy-vs-copy everywhere.
func TestWeightAlias_fixturesByteIdentical(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dirs, _ := filepath.Glob("../testdata/*/model.safetensors")
	if len(dirs) == 0 {
		t.Skip("no safetensors fixtures under testdata/")
	}
	type seen struct{ groups, singles, f16, int8 bool }
	var cover seen
	var ran, skipped []string
	for _, st := range dirs {
		dir := filepath.Dir(st)
		name := filepath.Base(dir)
		t.Run(name, func(t *testing.T) {
			giw := filepath.Join(t.TempDir(), name+".int4.metal.giw")
			if err := prequant.Transcode(context.Background(), dir, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
				skipped = append(skipped, name+" (transcode: "+firstLine(err)+")")
				t.Skipf("transcode: %v", err)
			}
			off, ok := aliasArm(t, giw, "0")
			if !ok {
				skipped = append(skipped, name+" (not resident on Metal)")
				t.Skip("Metal does not build this fixture resident")
			}
			on, _ := aliasArm(t, giw, "1")
			if off.a.tensors != 0 || off.a.int8Tensors != 0 {
				t.Fatalf("alias=0 arm aliased %d int4 / %d int8 tensors — the knob is not off", off.a.tensors, off.a.int8Tensors)
			}
			a := on.a
			t.Logf("aliased: %d int4 tensors (%d fused groups, %.2f MB nibbles), %.2f MB f16 scales, %d int8 (%.2f MB); copied %d, declined %d, non-adjacent groups %d",
				a.tensors, a.groups, mb(a.aliased), mb(a.scaleBytes), a.int8Tensors, mb(a.int8Bytes), a.copied, a.declined, a.nonAdjacent)
			if a.declined != 0 {
				t.Errorf("%d tensors declined as unaligned in a freshly written v14 file — the writer's alignment regressed", a.declined)
			}
			cover.groups = cover.groups || a.groups > 0
			cover.singles = cover.singles || a.tensors > a.groups
			cover.f16 = cover.f16 || a.scaleBytes > 0
			cover.int8 = cover.int8 || a.int8Tensors > 0
			requireSameBits(t, on.logits, off.logits)
			ran = append(ran, name)
		})
	}
	t.Logf("compared %d fixtures: %s", len(ran), strings.Join(ran, " "))
	t.Logf("skipped %d: %s", len(skipped), strings.Join(skipped, "; "))
	if len(ran) == 0 {
		t.Fatal("no fixture was built resident on Metal — the gate compared nothing")
	}
	if !cover.groups || !cover.singles || !cover.f16 || !cover.int8 {
		t.Errorf("coverage hole across all fixtures: fused groups=%v kind-7 singles=%v f16 scales=%v int8 LM head=%v — "+
			"a layout no fixture exercises is a layout this gate does not vouch for", cover.groups, cover.singles, cover.f16, cover.int8)
	}
}

// S6's Close-ordering gate. A no-copy buffer points INTO the .giw mapping, so the resident must release
// every such buffer before the model unmaps the file (Model.Close: resident first, then munmap) — the
// other order leaves the GPU a pointer into unmapped pages. Aliasing-on load/Close cycles must therefore
// (a) not fault, (b) leave the device's allocated size where it started, and (c) leave no mapping of the
// file behind — the region vmmap lists while the model is open has to be gone after Close, every cycle.
// The existing close_leak tests load a .gguf, which never aliases, so they cannot see any of this.
func TestWeightAlias_closeCycles(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	vm, err := exec.LookPath("vmmap")
	if err != nil {
		t.Skip("vmmap not available")
	}
	const fixture = "../testdata/gemma4-dense-scaled" // the largest aliasing fixture (~53 MB of nibbles)
	if _, err := os.Stat(filepath.Join(fixture, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", fixture)
	}
	tmp, err := filepath.EvalSymlinks(t.TempDir()) // vmmap prints the resolved path (/private/var/...)
	if err != nil {
		t.Fatal(err)
	}
	giw := filepath.Join(tmp, "alias-cycles.int4.metal.giw")
	if err := prequant.Transcode(context.Background(), fixture, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("transcode: %v", err)
	}
	regions := func() int {
		out, err := exec.Command(vm, "-interleaved", fmt.Sprint(os.Getpid())).Output()
		if err != nil {
			t.Skipf("vmmap: %v", err)
		}
		n := 0
		for _, l := range strings.Split(string(out), "\n") {
			if strings.Contains(l, giw) {
				n++
			}
		}
		return n
	}
	var openSize uint64
	cycle := func(i int, checkOpen bool) {
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4",
			Knobs: &decoder.Knobs{"GOINFER_METAL_ALIAS": "1"}})
		if err != nil {
			t.Fatalf("cycle %d: load: %v", i, err)
		}
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("cycle %d: not resident: %s", i, m.ResidentDecline())
		}
		if a := rf.(*metalResident).r.alias; a == nil || a.tensors == 0 {
			t.Fatalf("cycle %d: nothing aliased — this would be a copy-path cycle", i)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		for p := range 6 {
			if _, err := rf.Forward(m.EmbedResidentForTest((p*97+3)%vocab), p); err != nil {
				t.Fatalf("cycle %d: forward[%d]: %v", i, p, err)
			}
		}
		if checkOpen {
			openSize = d.CurrentAllocatedSize()
			if n := regions(); n == 0 {
				t.Fatalf("vmmap lists no region for %s while the model is open — the after-Close check below could not see a leftover", giw)
			}
		}
		if err := m.Close(); err != nil {
			t.Fatalf("cycle %d: Close: %v", i, err)
		}
		if n := regions(); n != 0 {
			t.Errorf("cycle %d: %d region(s) of %s still mapped after Close", i, n, giw)
		}
	}
	idle := d.CurrentAllocatedSize()
	cycle(0, true) // warm: pipelines, library compile
	base := d.CurrentAllocatedSize()
	t.Logf("device CurrentAllocatedSize: idle %d, with the model open %d, after its Close %d bytes", idle, openSize, base)
	if openSize <= base {
		t.Fatalf("CurrentAllocatedSize did not rise while a model was open (%d → %d) — this handle cannot see the resident's buffers, so the flatness check below would be blind", base, openSize)
	}
	const cycles = 5
	for i := 1; i <= cycles; i++ {
		cycle(i, false)
	}
	end := d.CurrentAllocatedSize()
	t.Logf("device CurrentAllocatedSize %d → %d bytes over %d aliasing-on load/Close cycles", base, end, cycles)
	if end > base+8<<20 {
		t.Errorf("device allocated size grew %d → %d bytes over %d cycles — a staircase, not a sawtooth", base, end, cycles)
	}
}

// aliasArm loads giw on the Metal backend with GOINFER_METAL_ALIAS=alias, drives 12 tokens and returns
// their logits and a copy of the alias ledger; ok=false when Metal does not build the model resident.
// 12 positions, not 1: softmax over a single key is 1.0 at any scale and hides attention defects.
func aliasArm(t *testing.T, giw, alias string) (aliasRun, bool) {
	t.Helper()
	m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4",
		Knobs: &decoder.Knobs{"GOINFER_METAL_ALIAS": alias}})
	if err != nil {
		t.Fatalf("load (alias=%s): %v", alias, err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		return aliasRun{}, false
	}
	mr, ok := rf.(*metalResident)
	if !ok {
		t.Fatalf("resident forward is %T, not *metalResident", rf)
	}
	var out aliasRun
	if mr.r.alias != nil {
		out.a = *mr.r.alias
		out.summary = mr.r.alias.summary()
	}
	_, _, _, _, _, _, vocab := m.Dims()
	rf.Reset()
	for i := range 12 {
		l, err := rf.Forward(m.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("forward[%d] (alias=%s): %v", i, alias, err)
		}
		out.logits = append(out.logits, append([]float32(nil), l...))
	}
	return out, true
}

type aliasRun struct {
	logits  [][]float32
	a       weightAlias
	summary string
}

// requireSameBits fails for every step whose logits are not bit-identical.
func requireSameBits(t *testing.T, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%d vs %d steps", len(got), len(want))
	}
	for s := range want {
		x, y := got[s], want[s]
		if len(x) != len(y) {
			t.Fatalf("step %d: %d vs %d logits", s, len(x), len(y))
		}
		diff, maxAbs := 0, 0.0
		for i := range x {
			if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
				diff++
				maxAbs = math.Max(maxAbs, math.Abs(float64(x[i]-y[i])))
			}
		}
		if diff != 0 {
			t.Errorf("step %d: aliased logits differ from copied in %d/%d positions (max |diff| %g)", s, diff, len(x), maxAbs)
		}
	}
}

// An older bundle — here a v12 file, what every non-metal target still writes and what a metal sidecar
// was before v13/v14 — must still load with aliasing on: its 16-aligned singles alias, its fused groups
// and f16 scales take the copy path, the logits stay bit-identical, and the banner says why (no f16
// scales) and what fixes it. A current metal-target file of the same model carries no such note, and
// the anonymous figure it reports is the small remainder the format does not cover.
func TestWeightAlias_olderBundleTakesCopyPath(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const fixture = "../testdata/llama-tiny"
	if _, err := os.Stat(filepath.Join(fixture, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", fixture)
	}
	tmp := t.TempDir()
	build := func(target decoder.GIWTarget, name string) string {
		p := filepath.Join(tmp, name)
		if err := prequant.Transcode(context.Background(), fixture, p, "int4", false, target); err != nil {
			t.Fatalf("transcode (%q): %v", target, err)
		}
		return p
	}
	old, cur := build(decoder.GIWTargetNone, "llama-tiny.int4.giw"), build(decoder.GIWTargetMetal, "llama-tiny.int4.metal.giw")

	oldOff, ok := aliasArm(t, old, "0")
	if !ok {
		t.Skip("Metal does not build this fixture resident")
	}
	oldOn, _ := aliasArm(t, old, "1")
	requireSameBits(t, oldOn.logits, oldOff.logits)
	a := oldOn.a
	t.Logf("v12 file: %s", strings.TrimSpace(oldOn.summary))
	if a.tensors == 0 {
		t.Errorf("v12 file aliased no int4 tensor — its singles are 16-aligned and should bind in place")
	}
	if a.scaleBytes != 0 || a.f16Converted == 0 {
		t.Errorf("v12 file: %d f16 scale bytes aliased, %d scale sets converted — want 0 aliased and >0 converted (it has no f16 scales)", a.scaleBytes, a.f16Converted)
	}
	if a.copyB == 0 {
		t.Errorf("v12 file reports 0 bytes copied, but its fused groups and scales took the copy path")
	}
	if !strings.Contains(oldOn.summary, "carries no f16 scales") || !strings.Contains(oldOn.summary, "-target metal") {
		t.Errorf("v12 file: banner has no note saying why less was aliased and how to fix it:\n%s", oldOn.summary)
	}

	curOn, _ := aliasArm(t, cur, "1")
	t.Logf("v14 file: %s", strings.TrimSpace(curOn.summary))
	if strings.Contains(curOn.summary, "note:") {
		t.Errorf("a current metal-target file got the older-bundle note:\n%s", curOn.summary)
	}
	c := curOn.a
	if bound := c.aliased + c.int8Bytes + c.scaleBytes; c.copyB >= a.copyB || c.copyB*20 > bound {
		t.Errorf("v14 file still copies %d bytes (v12: %d; bound in place: %d) — want well under 5%% of what it binds", c.copyB, a.copyB, bound)
	}
}

// Aliasing is ON BY DEFAULT for a .giw-mapped model (S6 shipped 2026-09-24): with the knob unset the build
// aliases, =0 turns it off, and a load that has no .giw mapping (safetensors here) never aliases.
func TestWeightAlias_onByDefault(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const fixture = "../testdata/llama-tiny"
	if _, err := os.Stat(filepath.Join(fixture, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", fixture)
	}
	t.Setenv("GOINFER_METAL_ALIAS", "") // neutralize the shell; the snapshot then sees no setting
	os.Unsetenv("GOINFER_METAL_ALIAS")
	giw := filepath.Join(t.TempDir(), "llama-tiny.int4.metal.giw")
	if err := prequant.Transcode(context.Background(), fixture, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("transcode: %v", err)
	}
	aliasOf := func(path string, knobs *decoder.Knobs) *weightAlias {
		t.Helper()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", Knobs: knobs})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		defer m.Close()
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Skipf("not resident on Metal: %s", m.ResidentDecline())
		}
		return rf.(*metalResident).r.alias
	}
	if a := aliasOf(giw, nil); a == nil || a.tensors == 0 {
		t.Errorf("knob unset: .giw load did not alias (alias=%v) — the default is supposed to be on", a)
	}
	if a := aliasOf(giw, &decoder.Knobs{"GOINFER_METAL_ALIAS": "0"}); a != nil {
		t.Errorf("GOINFER_METAL_ALIAS=0: the build still aliased %d tensors", a.tensors)
	}
	if a := aliasOf(fixture, nil); a != nil {
		t.Errorf("a safetensors load (no .giw mapping) built an aliaser")
	}
}

func mb(b int64) float64 { return float64(b) / (1 << 20) }

func firstLine(err error) string {
	s := err.Error()
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return s
}
