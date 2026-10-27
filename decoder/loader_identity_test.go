package decoder

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

const loaderIdentityPath = "../testdata/loader_identity.json"

// loaderQuants is parseQuant's modes, by the names Options.Quant takes.
var loaderQuants = []string{"f32", "int8", "int8int8", "int4", "int4mix", "q4k", "bf16", "f16"}

// loaderFlags is the backend flag sets Load passes loadWeights (wantsCanonicalInt4, wantsRow4Fallback): the CPU backend's
// (no canonical int4 kept, row4 allowed), a device backend's other than Metal (canonical kept), and Metal's (canonical
// kept, no row4). A mode is a quant name, then the flag set's suffix.
var loaderFlags = []struct {
	suffix                  string
	needCanonical, skipRow4 bool
}{{"", false, false}, {"/canonical", true, false}, {"/canonical/norow4", true, true}}

// loaderModes is every quant under every flag set.
var loaderModes = func() []string {
	var out []string
	for _, q := range loaderQuants {
		for _, f := range loaderFlags {
			out = append(out, q+f.suffix)
		}
	}
	return out
}()

// loaderFixtures lists every committed loader fixture, relative to the repository root: each testdata/*/ holding a
// config.json and a model.safetensors, and each testdata/*.gguf, as git ls-files names them.
func loaderFixtures(t *testing.T) []string {
	t.Helper()
	ls, err := exec.Command("git", "ls-files", "../testdata").Output()
	if err != nil {
		t.Fatalf("git ls-files: %v", err)
	}
	files := map[string]bool{}
	for _, f := range strings.Split(strings.TrimSpace(string(ls)), "\n") {
		files["testdata/"+strings.TrimPrefix(f, "../testdata/")] = true
	}
	var out []string
	for f := range files {
		dir, base := path.Split(f)
		switch {
		case dir == "testdata/" && strings.HasSuffix(base, ".gguf"):
			out = append(out, f)
		case base == "model.safetensors" && strings.Count(f, "/") == 2 && files[dir+"config.json"]:
			out = append(out, strings.TrimSuffix(dir, "/"))
		}
	}
	if len(out) == 0 {
		t.Fatal("no loader fixture found: the walk covered nothing")
	}
	sort.Strings(out)
	return out
}

// loadForIdentity loads p in mode (loaderModes) with that mode's backend flags.
func loadForIdentity(p, mode string) (*Weights, error) {
	name, suffix, _ := strings.Cut(mode, "/")
	if suffix != "" {
		suffix = "/" + suffix
	}
	q, err := parseQuant(name)
	if err != nil {
		return nil, err
	}
	if err := checkHalf16Load(q, p, "cpu"); err != nil {
		return nil, err
	}
	for _, f := range loaderFlags {
		if f.suffix == suffix {
			return loadWeights(p, q, false, f.needCanonical, f.skipRow4, nil, nil)
		}
	}
	return nil, fmt.Errorf("no flag set %q", suffix)
}

// TestLoaderIdentity_table is SB4's LI1 and the table check: every fixture and mode that loads, loaded twice, gives
// equal identities, and their digests equal this GOARCH's entry in testdata/loader_identity.json (-update rewrites it).
// A pair that loads with no entry, or an entry that no longer loads, fails. The table is per GOARCH because the int4
// repacks are.
func TestLoaderIdentity_table(t *testing.T) {
	got := map[string]map[string]string{}
	pairs := 0
	for _, fx := range loaderFixtures(t) {
		for _, mode := range loaderModes {
			w, err := loadForIdentity(filepath.Join("..", fx), mode)
			if err != nil {
				continue
			}
			ids := loaderIdentity(w)
			w2, err := loadForIdentity(filepath.Join("..", fx), mode)
			if err != nil {
				t.Fatalf("%s %s: loaded once, then: %v", fx, mode, err)
			}
			if d := loaderIdentityDiff(ids, loaderIdentity(w2)); len(d) > 0 {
				t.Errorf("%s %s: two loads differ in %v", fx, mode, d)
			}
			if got[fx] == nil {
				got[fx] = map[string]string{}
			}
			got[fx][mode] = loaderIdentityDigest(ids)
			pairs++
		}
	}
	table := map[string]map[string]map[string]string{}
	if raw, err := os.ReadFile(loaderIdentityPath); err == nil {
		if err := json.Unmarshal(raw, &table); err != nil {
			t.Fatal(err)
		}
	} else if !*updateMatrix {
		t.Fatalf("read %s: %v (run with -update to create it)", loaderIdentityPath, err)
	}
	arch := runtime.GOARCH
	if *updateMatrix {
		table[arch] = got
		b, err := json.MarshalIndent(table, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(loaderIdentityPath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d %s fixture-mode digests to %s", pairs, arch, loaderIdentityPath)
		return
	}
	want, ok := table[arch]
	if !ok {
		t.Skipf("%s has no %s entry, so the table is NOT checked here (LI1's two-load check ran: %d pairs); run -update on %s", loaderIdentityPath, arch, pairs, arch)
	}
	var diffs []string
	for fx, modes := range got {
		for mode, d := range modes {
			if want[fx][mode] != d {
				diffs = append(diffs, fx+" "+mode)
			}
		}
	}
	for fx, modes := range want {
		for mode := range modes {
			if _, ok := got[fx][mode]; !ok {
				diffs = append(diffs, fx+" "+mode+" (no longer loads)")
			}
		}
	}
	sort.Strings(diffs)
	if len(diffs) > 0 {
		t.Errorf("%s loader identities changed for %d fixture-mode pair(s); if the change is meant, run -update and say which tensors moved:\n  %s",
			arch, len(diffs), strings.Join(diffs, "\n  "))
	}
	t.Logf("%d fixture-mode pairs (%s)", pairs, arch)
}

// TestLoaderIdentity_giwRoundTrip is SB4's LI3(a): for every fixture and every mode the .giw holds, the weights read back
// from their .giw (a reader that shares no read path with the safetensors and GGUF loaders) have the direct load's
// identity. bf16 and f16 are left out: the .giw has no form for them, and the entry points refuse them before writing
// one (stream_dir.go, sidecar.go), while SerializeWeights itself writes a blob its reader then refuses. Where the CPU
// flags build an int4 tensor repacked-only (row4, off amd64), the writer refuses it by design; that refusal is accepted
// only in a CPU-flag mode, and the same quant loaded with the canonical copy kept must then round-trip in its place.
func TestLoaderIdentity_giwRoundTrip(t *testing.T) {
	checked := map[string]bool{}
	var refused []string
	for _, fx := range loaderFixtures(t) {
		for _, mode := range loaderModes {
			if q, _ := parseQuant(strings.Split(mode, "/")[0]); q.half16() {
				continue
			}
			w, err := loadForIdentity(filepath.Join("..", fx), mode)
			if err != nil {
				continue
			}
			b, err := SerializeWeights(w, "sb4")
			if err != nil {
				if strings.Contains(mode, "/") || !strings.Contains(err.Error(), "holds only its row4 layout") {
					t.Errorf("%s %s: serialize: %v", fx, mode, err)
				}
				refused = append(refused, fx+" "+mode)
				continue
			}
			w2, err := LoadSerializedWeights(b)
			if err != nil {
				t.Errorf("%s %s: read back: %v", fx, mode, err)
				continue
			}
			if d := loaderIdentityDiff(loaderIdentity(w), loaderIdentity(w2)); len(d) > 0 {
				t.Errorf("%s %s: the .giw round trip moved %v", fx, mode, d)
			}
			checked[fx+" "+mode] = true
		}
	}
	for _, r := range refused {
		if !checked[r+"/canonical"] {
			t.Errorf("%s: refused as row4-only, and its /canonical load did not round-trip in its place", r)
		}
	}
	if len(checked) == 0 {
		t.Fatal("no fixture round-tripped: nothing compared")
	}
	t.Logf("%d fixture-mode pairs round-tripped; %d CPU-flag pairs refused as row4-only, each covered by its /canonical load: %v",
		len(checked), len(refused), refused)
}

// TestLoaderIdentity_oneByte is SB4's LI2: flipping the sign bit of one element of one source tensor moves, in every mode
// that loads, exactly the identity of the tensor loaded from it.
func TestLoaderIdentity_oneByte(t *testing.T) {
	cases := []struct {
		fixture, tensor, path string
		plant                 func(t *testing.T, src, dst, tensor string)
	}{
		{"llama-tiny", "model.layers.0.self_attn.k_proj.weight", "Layers[0].KProj", plantSafetensors},
		{"glm-tiny.gguf", "blk.0.attn_k.weight", "Layers[0].KProj", plantGGUF},
	}
	for _, c := range cases {
		t.Run(c.fixture, func(t *testing.T) {
			src := filepath.Join("../testdata", c.fixture)
			dst := filepath.Join(t.TempDir(), c.fixture)
			c.plant(t, src, dst, c.tensor)
			n := 0
			for _, mode := range loaderQuants {
				a, err := loadForIdentity(src, mode)
				if err != nil {
					continue
				}
				b, err := loadForIdentity(dst, mode)
				if err != nil {
					t.Fatalf("%s: the planted copy does not load: %v", mode, err)
				}
				if d := loaderIdentityDiff(loaderIdentity(a), loaderIdentity(b)); !slices.Equal(d, []string{c.path}) {
					t.Errorf("%s: one flipped byte in %s moved %v, want [%s]", mode, c.tensor, d, c.path)
				}
				n++
			}
			if n == 0 {
				t.Fatal("no mode loaded: nothing compared")
			}
			t.Logf("%d modes", n)
		})
	}
}

// plantSafetensors copies the fixture directory src to dst and flips the sign of tensor's first nonzero f32 element.
func plantSafetensors(t *testing.T, src, dst, tensor string) {
	t.Helper()
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "model.safetensors"} {
		b, err := os.ReadFile(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		if f == "model.safetensors" {
			n := binary.LittleEndian.Uint64(b[:8])
			var hdr map[string]json.RawMessage
			if err := json.Unmarshal(b[8:8+n], &hdr); err != nil {
				t.Fatal(err)
			}
			var info struct {
				DType   string   `json:"dtype"`
				Offsets [2]int64 `json:"data_offsets"`
			}
			if err := json.Unmarshal(hdr[tensor], &info); err != nil {
				t.Fatalf("%s: %v", tensor, err)
			}
			if info.DType != "F32" {
				t.Fatalf("%s is %s; the plant expects F32", tensor, info.DType)
			}
			flipFirstNonzeroF32(t, b[8+int64(n)+info.Offsets[0]:8+int64(n)+info.Offsets[1]])
		}
		if err := os.WriteFile(filepath.Join(dst, f), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// plantGGUF copies the GGUF src to dst and flips the sign of tensor's first nonzero element. The tensor must be stored
// as F32; its bytes are found by searching the file for its values.
func plantGGUF(t *testing.T, src, dst, tensor string) {
	t.Helper()
	g, err := embed.OpenGGUF(src)
	if err != nil {
		t.Fatal(err)
	}
	_, vals, err := g.Tensor(tensor)
	g.Close()
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(want[4*i:], math.Float32bits(v))
	}
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	at := bytes.Index(b, want)
	if at < 0 || bytes.Contains(b[at+1:], want) {
		t.Fatalf("%s's F32 bytes are not in %s exactly once", tensor, src)
	}
	flipFirstNonzeroF32(t, b[at:at+len(want)])
	if err := os.WriteFile(dst, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func flipFirstNonzeroF32(t *testing.T, data []byte) {
	t.Helper()
	for i := 0; i+4 <= len(data); i += 4 {
		if binary.LittleEndian.Uint32(data[i:])&0x7fffffff != 0 {
			data[i+3] ^= 0x80
			return
		}
	}
	t.Fatal("tensor is all zeros: no element to flip")
}

// walkValue is the walk loaderIdentity makes, over any struct v points to.
func walkValue(v any) []tensorIdentity {
	rv := reflect.ValueOf(v).Elem()
	checkIdentType("x", rv.Type(), map[reflect.Type]bool{})
	iw := &identWalker{}
	iw.walk("x", rv)
	return iw.out
}

// TestLoaderIdentity_walkTotal is SB4's LI5: a field of a kind the walk does not know panics instead of being skipped,
// even when it is nil; every field loaderSkip names exists on Weights, so a rename cannot leave a skip entry that hides
// nothing; and a slice's length is part of its identity, zeros included.
func TestLoaderIdentity_walkTotal(t *testing.T) {
	for _, v := range []any{
		&struct{ M map[string][]float32 }{M: map[string][]float32{"a": {1}}},
		&struct{ F func() }{F: func() {}},
		&struct{ I any }{I: 1},
		&struct{ P *float32 }{},
		&struct{ S []map[string]int }{},
		&struct{ S []func() }{},
		&struct{ S []struct{ I any } }{},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%T: the walk skipped a field it does not know", v)
				}
			}()
			walkValue(v)
		}()
	}
	type ints struct{ S []int }
	type bools struct{ S []bool }
	type arr struct{ A [2]float32 }
	type nested struct{ S []struct{ X float32 } }
	for _, pair := range [][2]any{
		{&ints{[]int{0, 0}}, &ints{}},
		{&ints{[]int{1, 0}}, &ints{[]int{1}}},
		{&ints{[]int{}}, &ints{}},
		{&bools{[]bool{false}}, &bools{[]bool{false, false}}},
		{&arr{[2]float32{0, 1}}, &arr{[2]float32{1, 0}}},
		{&nested{make([]struct{ X float32 }, 1)}, &nested{make([]struct{ X float32 }, 2)}},
	} {
		if a, b := walkValue(pair[0]), walkValue(pair[1]); loaderIdentityDigest(a) == loaderIdentityDigest(b) {
			t.Errorf("%+v and %+v have one identity", pair[0], pair[1])
		}
	}
	wt := reflect.TypeOf(Weights{})
	for name := range loaderSkip {
		if _, ok := wt.FieldByName(name); !ok {
			t.Errorf("loaderSkip names %q, which Weights does not have", name)
		}
	}
}

// TestLoaderIdentity_kquant: a k-quant source entry is part of the identity of the WeightMat whose int8 copy it keys, so
// changing it moves exactly that tensor, and an entry that keys no walked tensor is listed. No committed fixture holds a
// k-quant tensor (the tiny GGUFs are F32, and Q4_K needs rows of 256), so this is the only check of the path.
func TestLoaderIdentity_kquant(t *testing.T) {
	mk := func(src kquantSrc, stray bool) []tensorIdentity {
		w := &Weights{Embed: linalg.QuantizeInt8(make([]float32, 4*256), 4, 256, false), LMHead: linalg.QuantizeInt8(make([]float32, 4*256), 4, 256, false)}
		q8, _, _, _ := w.Embed.Int8()
		w.kquant = map[*int8]kquantSrc{unsafe.SliceData(q8): src}
		if stray {
			w.kquant[new(int8)] = src
		}
		return loaderIdentity(w)
	}
	base := mk(kquantSrc{tensor: "token_embd.weight", kind: "q5_k"}, false)
	if d := loaderIdentityDiff(base, mk(kquantSrc{tensor: "token_embd.weight", kind: "q6_k"}, false)); !slices.Equal(d, []string{"Embed"}) {
		t.Errorf("a changed k-quant source moved %v, want [Embed]", d)
	}
	if d := loaderIdentityDiff(base, mk(kquantSrc{tensor: "token_embd.weight", kind: "q5_k"}, true)); !slices.Equal(d, []string{"kquant"}) {
		t.Errorf("a stray k-quant entry moved %v, want [kquant]", d)
	}
}
