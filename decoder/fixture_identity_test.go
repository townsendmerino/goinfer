package decoder

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The tiny parity checkpoints are gitignored and regenerated per machine by scripts/pin_*.py, but the
// goldens recorded from them are committed — and the pin scripts do not reproduce the same random weights
// across torch/transformers versions. So a box whose checkpoint was re-pinned elsewhere compares the
// committed golden against DIFFERENT weights, and the parity test reports a meaningless cosine (−0.04 for
// gemma3-vl-tiny) that reads exactly like a forward-pass regression. That kept five decoder tests red on
// the Mac from 2026-09-18 to 2026-09-24, pre-registered as "known unrelated" and never investigated
// (docs/measurements/tiny-fixture-golden-mismatch-2026-09-24.md).
//
// requireFixtureIdentity fails fast, with the actual diagnosis, when a fixture listed in
// testdata/fixture_identity.json does not hold the weights its goldens were recorded from.
//
// The comparison is NUMERIC, not a file hash: the same pin script with the same seed produces
// byte-different checkpoints on arm64 and amd64 (measured: nemotron3nano-tiny and qwen3next-tiny differ in
// 33-36% of elements between the Mac and nobara, by at most 2.4e-7 — float32 ULP noise from torch's normal_
// on the two architectures), and both correctly pass every golden. A different random draw moves a
// tensor's sum by O(sum|x|/sqrt(n)); ULP noise moves it by ~1e-7 of sum|x|. The 1e-4 relative tolerance
// sits three orders of magnitude from each.
const (
	fixtureIdentityPath = "../testdata/fixture_identity.json"
	fixtureIdentityTol  = 1e-4
)

type fixtureIdentity struct {
	Recorded string                `json:"recorded"`
	Pin      string                `json:"pin,omitempty"`
	Tensors  map[string][2]float64 `json:"tensors"` // name -> [sum, sum|x|], float64 over the F32 elements
}

func requireFixtureIdentity(t *testing.T, fixtureDir string) {
	t.Helper()
	msg, err := fixtureIdentityMismatch(fixtureIdentityPath, fixtureDir)
	if err != nil {
		t.Fatal(err)
	}
	if msg != "" {
		t.Fatal(msg)
	}
}

func readFixtureIdentities(manifestPath string) (map[string]fixtureIdentity, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", manifestPath, err)
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		return nil, fmt.Errorf("parse %s: %w", manifestPath, err)
	}
	out := map[string]fixtureIdentity{}
	for name, v := range all {
		if strings.HasPrefix(name, "_") {
			continue
		}
		var fi fixtureIdentity
		if err := json.Unmarshal(v, &fi); err != nil {
			return nil, fmt.Errorf("parse %s[%s]: %w", manifestPath, name, err)
		}
		out[name] = fi
	}
	return out, nil
}

// fixtureIdentityMismatch returns a diagnosis when fixtureDir's checkpoint does not hold the weights
// manifestPath records, "" when it does (or the fixture is not listed), and an error when the manifest or
// checkpoint cannot be read.
func fixtureIdentityMismatch(manifestPath, fixtureDir string) (string, error) {
	all, err := readFixtureIdentities(manifestPath)
	if err != nil {
		return "", err
	}
	name := filepath.Base(fixtureDir)
	want, ok := all[name]
	if !ok {
		return "", nil
	}
	got, err := safetensorsFingerprint(filepath.Join(fixtureDir, "model.safetensors"))
	if err != nil {
		return "", fmt.Errorf("fingerprint %s: %w", fixtureDir, err)
	}
	var bad []string
	for tn, w := range want.Tensors {
		g, ok := got[tn]
		if !ok {
			bad = append(bad, tn+" (missing)")
			continue
		}
		tol := fixtureIdentityTol*w[1] + 1e-9
		if math.Abs(g[0]-w[0]) > tol || math.Abs(g[1]-w[1]) > tol {
			bad = append(bad, fmt.Sprintf("%s (sum %.6g vs %.6g)", tn, g[0], w[0]))
		}
	}
	for tn := range got {
		if _, ok := want.Tensors[tn]; !ok {
			bad = append(bad, tn+" (not in the recorded checkpoint)")
		}
	}
	if len(bad) == 0 {
		return "", nil
	}
	sort.Strings(bad)
	if len(bad) > 3 {
		bad = append(bad[:3], fmt.Sprintf("… %d more", len(bad)-3))
	}
	return fmt.Sprintf("%s/model.safetensors does not hold the weights its committed goldens were recorded from "+
		"(%s; recorded %s).\nThis is a FIXTURE mismatch, not a forward-pass regression: copy the checkpoint from the box "+
		"that recorded it, or re-pin with %s and commit the regenerated goldens together with a regenerated %s "+
		"(GOINFER_FIXTURE_IDENTITY_UPDATE=1 go test ./decoder/ -run TestFixtureIdentity_update).",
		fixtureDir, strings.Join(bad, ", "), want.Recorded, want.Pin, filepath.Base(manifestPath)), nil
}

// safetensorsFingerprint returns, per tensor, [sum, sum|x|] in float64 over its F32 elements. Every tiny
// fixture in the manifest is all-F32; any other dtype is an error rather than a silent skip.
func safetensorsFingerprint(path string) (map[string][2]float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 8 {
		return nil, fmt.Errorf("%s: too short for a safetensors file", path)
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if n > uint64(len(b)-8) {
		return nil, fmt.Errorf("%s: header length %d exceeds file", path, n)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &hdr); err != nil {
		return nil, fmt.Errorf("%s: header: %w", path, err)
	}
	data := b[8+n:]
	out := map[string][2]float64{}
	for name, raw := range hdr {
		if name == "__metadata__" {
			continue
		}
		var ti struct {
			Dtype   string   `json:"dtype"`
			Offsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &ti); err != nil {
			return nil, fmt.Errorf("%s: tensor %s: %w", path, name, err)
		}
		if ti.Dtype != "F32" {
			return nil, fmt.Errorf("%s: tensor %s is %s; the fingerprint reads F32 only", path, name, ti.Dtype)
		}
		lo, hi := ti.Offsets[0], ti.Offsets[1]
		if lo < 0 || hi < lo || hi > int64(len(data)) || (hi-lo)%4 != 0 {
			return nil, fmt.Errorf("%s: tensor %s: bad offsets %v", path, name, ti.Offsets)
		}
		var sum, abs float64
		for i := lo; i < hi; i += 4 {
			v := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[i:])))
			sum += v
			abs += math.Abs(v)
		}
		out[name] = [2]float64{sum, abs}
	}
	return out, nil
}

// TestFixtureIdentity_update regenerates testdata/fixture_identity.json from the fixtures present on this
// machine — only when asked, and only for fixtures already listed (so a new entry is a deliberate edit),
// keeping each entry's recorded/pin notes. Run it right after the fixture's goldens pass here.
func TestFixtureIdentity_update(t *testing.T) {
	if os.Getenv("GOINFER_FIXTURE_IDENTITY_UPDATE") == "" {
		t.Skip("set GOINFER_FIXTURE_IDENTITY_UPDATE=1 to rewrite testdata/fixture_identity.json from local fixtures")
	}
	raw, err := os.ReadFile(fixtureIdentityPath)
	if err != nil {
		t.Fatal(err)
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		t.Fatal(err)
	}
	for name, v := range all {
		if strings.HasPrefix(name, "_") {
			continue
		}
		var fi fixtureIdentity
		if err := json.Unmarshal(v, &fi); err != nil {
			t.Fatal(err)
		}
		fp, err := safetensorsFingerprint(filepath.Join("../testdata", name, "model.safetensors"))
		if err != nil {
			t.Logf("%s: not updated (%v)", name, err)
			continue
		}
		for tn, v := range fp { // 9 significant digits: far finer than the 1e-4 tolerance, half the file size
			fp[tn] = [2]float64{round9(v[0]), round9(v[1])}
		}
		fi.Tensors = fp
		b, err := json.Marshal(fi)
		if err != nil {
			t.Fatal(err)
		}
		all[name] = b
	}
	// One compact line per fixture, sorted: a re-pin shows up as exactly one changed line in a diff.
	keys := make([]string, 0, len(all))
	for k := range all {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out strings.Builder
	out.WriteString("{\n")
	for i, k := range keys {
		kb, _ := json.Marshal(k)
		var compact bytes.Buffer
		if err := json.Compact(&compact, all[k]); err != nil {
			t.Fatal(err)
		}
		out.WriteString(" " + string(kb) + ": " + compact.String())
		if i < len(keys)-1 {
			out.WriteString(",")
		}
		out.WriteString("\n")
	}
	out.WriteString("}\n")
	if err := os.WriteFile(fixtureIdentityPath, []byte(out.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("rewrote %s (%d entries)", fixtureIdentityPath, len(all))
}

func round9(x float64) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'g', 9, 64), 64)
	return v
}

// writeF32Safetensors writes a minimal all-F32 safetensors file.
func writeF32Safetensors(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	hdr := map[string]any{}
	var data []byte
	for _, n := range names {
		lo := len(data)
		for _, v := range tensors[n] {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		hdr[n] = map[string]any{"dtype": "F32", "shape": []int{len(tensors[n])}, "data_offsets": []int{lo, len(data)}}
	}
	h, _ := json.Marshal(hdr)
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	b = append(append(b, h...), data...)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// The check must pass the recorded weights, pass them again with float32 ULP noise (the arm64-vs-amd64 case
// that is NOT a mismatch), flag a different random draw (the case that is), flag a missing or extra tensor,
// and ignore a fixture the manifest does not list.
func TestFixtureIdentityMismatch(t *testing.T) {
	root := t.TempDir()
	rng := func(seed uint64, n int) []float32 {
		out := make([]float32, n)
		x := seed*0x9E3779B97F4A7C15 + 1
		for i := range out {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			out[i] = float32(int64(x%2001)-1000) * 2e-5 // ~[-0.02, 0.02], like an init std of 0.02
		}
		return out
	}
	mk := func(name string, tensors map[string][]float32) string {
		d := filepath.Join(root, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		writeF32Safetensors(t, filepath.Join(d, "model.safetensors"), tensors)
		return d
	}
	recorded := map[string][]float32{"embed": rng(1, 4096), "a_log": rng(2, 8), "norm": {1, 1, 1, 1}}
	dir := mk("listed-tiny", recorded)
	fp, err := safetensorsFingerprint(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(root, "fixture_identity.json")
	mb, _ := json.Marshal(map[string]any{"_doc": "x", "listed-tiny": fixtureIdentity{Recorded: "test", Pin: "scripts/pin_x.py", Tensors: fp}})
	if err := os.WriteFile(manifest, mb, 0o644); err != nil {
		t.Fatal(err)
	}
	check := func(label string, wantFlag bool) {
		t.Helper()
		msg, err := fixtureIdentityMismatch(manifest, dir)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		if flagged := msg != ""; flagged != wantFlag {
			t.Errorf("%s: flagged=%v, want %v (msg %q)", label, flagged, wantFlag, msg)
		}
		if wantFlag && !strings.Contains(msg, "FIXTURE mismatch") {
			t.Errorf("%s: message does not name the cause: %q", label, msg)
		}
	}
	check("the recorded weights", false)

	ulp := map[string][]float32{}
	for n, v := range recorded {
		w := append([]float32(nil), v...)
		for i := range w {
			if i%3 == 0 && w[i] != 1 {
				w[i] = math.Nextafter32(w[i], 1) // 1-ULP noise on a third of the elements, as between arm64 and amd64
			}
		}
		ulp[n] = w
	}
	writeF32Safetensors(t, filepath.Join(dir, "model.safetensors"), ulp)
	check("the same weights with float32 ULP noise", false)

	redraw := map[string][]float32{"embed": rng(7, 4096), "a_log": recorded["a_log"], "norm": recorded["norm"]}
	writeF32Safetensors(t, filepath.Join(dir, "model.safetensors"), redraw)
	check("a different random draw of one tensor", true)

	writeF32Safetensors(t, filepath.Join(dir, "model.safetensors"), map[string][]float32{"embed": recorded["embed"], "a_log": recorded["a_log"]})
	check("a missing tensor", true)

	writeF32Safetensors(t, filepath.Join(dir, "model.safetensors"), map[string][]float32{"embed": recorded["embed"], "a_log": recorded["a_log"], "norm": recorded["norm"], "extra": {0}})
	check("an extra tensor", true)

	if msg, err := fixtureIdentityMismatch(manifest, mk("unlisted-tiny", recorded)); err != nil || msg != "" {
		t.Errorf("an unlisted fixture was checked: msg=%q err=%v", msg, err)
	}
}

// The committed manifest parses, and every entry has a fingerprint.
func TestFixtureIdentity_manifestIsComplete(t *testing.T) {
	all, err := readFixtureIdentities(fixtureIdentityPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"gemma3-vl-tiny", "bailing_hybrid-tiny"} {
		if _, ok := all[n]; !ok {
			t.Errorf("%s missing from %s", n, fixtureIdentityPath)
		}
	}
	for n, fi := range all {
		if len(fi.Tensors) == 0 {
			t.Errorf("%s: no tensor fingerprints", n)
		}
	}
}
