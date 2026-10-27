//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/binary"
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// S6's byte-identity gate over every resident-parity fixture, not just the two real checkpoints
// TestWeightAlias_logitsByteIdentical is pointed at. Each tiny fixture is transcoded to a metal-target
// .giw (fused q|k|v and gate|up groups, single tensors, binary16 scales), loaded twice
// with GOINFER_METAL_ALIAS off and on via Options.Knobs, and driven through the same tokens; every
// logit must match bit for bit. A fixture Metal does not build resident is skipped (named in the log),
// and one whose shapes leave nothing aliasable (cols not a multiple of 32) is compared anyway — the
// copy path must then be the only path — but at least one fixture must alias fused groups, single
// tensors, f16 scales and the int8 LM head, or this gate would be copy-vs-copy everywhere.
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
				t.Errorf("%d tensors declined as unaligned in a freshly written file — the writer's alignment regressed", a.declined)
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
		t.Errorf("coverage hole across all fixtures: fused groups=%v singles=%v f16 scales=%v int8 LM head=%v — "+
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

// A bundle for a target other than Metal must still load with aliasing on. Its single tensors bind in place, nibbles and
// binary16 scales both, because a single tensor's arrays are each one aligned run in any file. Its fused groups take the copy
// path, because only a -target metal file lays a group's members out back to back. The logits stay bit-identical, and the
// banner says why less was aliased and what fixes it. A metal-target file of the same model carries no such note, and the
// anonymous figure it reports is the small remainder the format does not cover.
//
// The name is historical: the non-Metal bundle is written here by this build, at the same format version as the Metal one,
// which the test asserts. docs/code-notes/metal.md#TestWeightAlias_olderBundleTakesCopyPath has the history.
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
	if vo, vc := giwFileVersion(t, old), giwFileVersion(t, cur); vo != vc {
		t.Fatalf("the non-metal bundle is weights format v%d and the metal one v%d: the writer emits one version for every target", vo, vc)
	}

	oldOff, ok := aliasArm(t, old, "0")
	if !ok {
		t.Skip("Metal does not build this fixture resident")
	}
	oldOn, _ := aliasArm(t, old, "1")
	requireSameBits(t, oldOn.logits, oldOff.logits)
	a := oldOn.a
	t.Logf("non-metal bundle (format v%d): %s", giwFileVersion(t, old), strings.TrimSpace(oldOn.summary))
	if a.tensors == 0 {
		t.Errorf("non-metal bundle aliased no int4 tensor — its singles are 16-aligned and should bind in place")
	}
	if a.scaleBytes == 0 {
		t.Errorf("non-metal bundle: no f16 scale bytes bound in place — a single tensor's binary16 scales are one aligned run in any file")
	}
	if a.f16Converted != 0 {
		t.Errorf("non-metal bundle: %d tensors had their nibbles bound in place and their scales copied — a single tensor's scales bind too", a.f16Converted)
	}
	if a.groups != 0 || a.nonAdjacent == 0 {
		t.Errorf("non-metal bundle: %d fused groups bound in place, %d found not adjacent — want 0 and >0 (its members are separate records)", a.groups, a.nonAdjacent)
	}
	if a.copyB == 0 {
		t.Errorf("non-metal bundle reports 0 bytes copied, but its fused groups took the copy path")
	}
	if !strings.Contains(oldOn.summary, "not in the layout Metal binds") || !strings.Contains(oldOn.summary, "-target metal") {
		t.Errorf("non-metal bundle: banner has no note saying why less was aliased and how to fix it:\n%s", oldOn.summary)
	}

	curOn, _ := aliasArm(t, cur, "1")
	t.Logf("metal-target bundle (format v%d): %s", giwFileVersion(t, cur), strings.TrimSpace(curOn.summary))
	if strings.Contains(curOn.summary, "note:") {
		t.Errorf("a metal-target file got the non-metal-bundle note:\n%s", curOn.summary)
	}
	c := curOn.a
	if bound := c.aliased + c.int8Bytes + c.scaleBytes; c.copyB >= a.copyB || c.copyB*20 > bound {
		t.Errorf("metal-target bundle still copies %d bytes (non-metal: %d; bound in place: %d) — want well under 5%% of what it binds", c.copyB, a.copyB, bound)
	}
}

// Aliasing is ON BY DEFAULT for a .giw-mapped model: with the knob unset the build
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

// giwFileVersion reads a .giw's WEIGHTS-format version. The file starts with a bundle header (magic "GINFB", version 3), and the weights blob
// (magic "GINFW", then a little-endian uint32 version: decoder/serialize.go) starts at byte 64 in a v3 bundle. Both magics are checked so a layout
// change fails here by name instead of reading some other four bytes as a version.
func giwFileVersion(t *testing.T, path string) uint32 {
	t.Helper()
	const bundleMagic, blobMagic, blobOffset = "GINFB", "GINFW", 64
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b := make([]byte, blobOffset+len(blobMagic)+4)
	if _, err := io.ReadFull(f, b); err != nil {
		t.Fatalf("%s: short read of the bundle header: %v", path, err)
	}
	if string(b[:len(bundleMagic)]) != bundleMagic || string(b[blobOffset:blobOffset+len(blobMagic)]) != blobMagic {
		t.Fatalf("%s: not a v3 bundle with its weights blob at byte %d (magics %q and %q)", path, blobOffset, b[:len(bundleMagic)], b[blobOffset:blobOffset+len(blobMagic)])
	}
	return binary.LittleEndian.Uint32(b[blobOffset+len(blobMagic):])
}

// TestBuildDemand_registered (G-GF-4): decoder's registry holds this package's report for metal, not decoder's default copy, so the
// agreement test's comparison is the one the fit guard reads. It first shows the check tells two registered reports apart.
func TestBuildDemand_registered(t *testing.T) {
	ptr := func(f decoder.BuildDemandFunc) uintptr { return reflect.ValueOf(f).Pointer() }
	if o := decoder.RegisteredBuildDemandForTest("cuda"); o == nil || ptr(o) == ptr(buildDemand) {
		t.Fatal("the cuda report is missing or is this package's: the identity check cannot tell reports apart")
	}
	if f := decoder.RegisteredBuildDemandForTest("metal"); f == nil || ptr(f) != ptr(buildDemand) {
		t.Fatal("decoder's registry does not hold this package's metal report")
	}
}

// metalI3 loads path on Metal with opts and checks G-MM3c-3 and G-MM3c-4 on it: the device's CurrentAllocatedSize just after the
// resident's device is created, against after the build and three identical requests, is at least the ledger's charged Unified
// bytes U and at most 64 MiB above them, or (if Metal counts no-copy buffers) within the same band above U plus the charged
// FileMapped bytes F; the charged bytes do not grow from one request to the next (I2); after Close the ledger holds no lease. It
// returns F. Run each cell in its own process: the system device is shared, and earlier tests' frees can land during a cell.
func metalI3(t *testing.T, path string, opts decoder.Options) (fileMapped int64) {
	t.Helper()
	const band = 64 << 20
	real := CreateSystemDefaultDevice
	defer func() { CreateSystemDefaultDevice = real }()
	created := map[*Device]uint64{}
	CreateSystemDefaultDevice = func() (*Device, error) {
		d, err := real()
		if err == nil {
			created[d] = d.CurrentAllocatedSize()
		}
		return d, err
	}
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Skipf("no metal: %v", err)
	}
	rf := m.ResidentForwardForTest()
	if rf == nil {
		m.Close()
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	d := rf.(*metalResident).r.d
	l := metalLedgerOf(d)
	var after []int64
	for range 3 {
		ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 8, decoder.SamplingParams{})
		for range ch {
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		after = append(after, l.Usage().Charged)
	}
	if after[1] != after[0] || after[2] != after[0] {
		t.Errorf("the charged bytes grow across identical requests: %v", after)
	}
	base, seen := created[d]
	if !seen {
		t.Fatal("the resident's device was not created through CreateSystemDefaultDevice")
	}
	u := l.Usage()
	measured, un, fm := int64(d.CurrentAllocatedSize())-int64(base), u.ByTier[mem.Unified], u.ByTier[mem.FileMapped]
	t.Logf("measured %d bytes; the ledger charges U=%d on Unified, F=%d on FileMapped (requested %d); measured-U %d, measured-(U+F) %d; by line %v",
		measured, un, fm, u.Requested, measured-un, measured-un-fm, u.ByLine)
	withoutF, withF := measured >= un && measured <= un+band, measured >= un+fm && measured <= un+fm+band
	switch {
	case withoutF && withF:
		t.Logf("both bands hold: F is too small here to tell whether CurrentAllocatedSize counts no-copy buffers")
	case withF:
		t.Logf("CurrentAllocatedSize counts the no-copy buffers (measured is within the band above U+F)")
	case withoutF:
		t.Logf("CurrentAllocatedSize does not count the no-copy buffers (measured is within the band above U)")
	default:
		t.Errorf("measured %d is outside both [U, U+%d] and [U+F, U+F+%d] (U=%d, F=%d)", measured, band, band, un, fm)
	}
	m.Close()
	if u := l.Usage(); u.Leases != 0 || u.Charged != 0 {
		t.Errorf("after Close the ledger holds %d leases, %d bytes, by line %v", u.Leases, u.Charged, u.ByLine)
	}
	return fm
}

// TestLedger_metalAgreesWithTheDevice (G-MM3c-3, G-MM3c-4): metalI3 on llama-tiny, which copies its weights (no aliasing).
func TestLedger_metalAgreesWithTheDevice(t *testing.T) {
	metalI3(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "metal", Quant: "int4"})
}

// TestLedger_metalAgreesWithTheDevice_alias (G-MM3c-4's aliasing cell): metalI3 on gemma4-dense-scaled transcoded to a Metal .giw
// and loaded with GOINFER_METAL_ALIAS, so its weights are no-copy windows (FileMapped); the cell must alias.
func TestLedger_metalAgreesWithTheDevice_alias(t *testing.T) {
	const fixture = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(filepath.Join(fixture, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", fixture)
	}
	giw := filepath.Join(t.TempDir(), "i3-alias.int4.metal.giw")
	if err := prequant.Transcode(context.Background(), fixture, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatal(err)
	}
	if fm := metalI3(t, giw, decoder.Options{Backend: "metal", Quant: "int4", Knobs: &decoder.Knobs{"GOINFER_METAL_ALIAS": "1"}}); fm == 0 {
		t.Error("the aliased load leased nothing on FileMapped: the cell did not alias")
	}
}

// TestLedger_metalUnplannedIsReported (G-MM3c-2's plant): an allocation on the empty line, and a buffer never released, are each
// what the suite's release check reports; a clean device reports nothing.
func TestLedger_metalUnplannedIsReported(t *testing.T) {
	real := gpu.ReleasedForTest
	defer func() { gpu.ReleasedForTest = real }()
	var got *mem.Ledger
	gpu.ReleasedForTest = func(d *Device) { got = metalLedgerOf(d) }
	for _, plant := range []string{"unplanned", "open", "clean"} {
		d, err := CreateSystemDefaultDevice()
		if err != nil {
			t.Skip(err)
		}
		_ = d.NewBufferLenOn(mem.LineKV, 1024)
		switch plant {
		case "unplanned":
			_ = d.NewBufferLenOn("", 1024)
		case "open":
			metalLedgerOf(d).Reserve(mem.Unified, mem.LineScratch, 4096, 4096) // never released
		}
		d.ReleaseAll()
		d.ReleaseObjects()
		b := metalLedgerBreak(got)
		switch {
		case plant == "unplanned" && !strings.Contains(b, "unplanned peak"):
			t.Errorf("an unplanned buffer gives %q", b)
		case plant == "open" && !strings.Contains(b, "1 leases (4096 bytes) open"):
			t.Errorf("an open lease gives %q", b)
		case plant == "clean" && b != "":
			t.Errorf("a clean device gives %q", b)
		}
	}
}
