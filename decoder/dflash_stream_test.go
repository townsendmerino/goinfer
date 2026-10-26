package decoder

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// repadSafetensors rewrites a safetensors file with its JSON header padded by trailing spaces so its data starts at an odd byte
// offset (odd) or a multiple of 8, as z-lab's Qwen3.5-9B drafter's odd 7,433-byte header leaves it.
func repadSafetensors(t *testing.T, path string, odd bool) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := int(binary.LittleEndian.Uint64(raw))
	hdr, blob := append([]byte(nil), raw[8:8+n]...), raw[8+n:]
	for (8+len(hdr))%2 == 1 != odd || (!odd && (8+len(hdr))%8 != 0) {
		hdr = append(hdr, ' ')
	}
	out := append(binary.LittleEndian.AppendUint64(nil, uint64(len(hdr))), hdr...)
	if err := os.WriteFile(path, append(out, blob...), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestOpenDFlashDrafter_matchesLoad: a drafter opened to stream (OpenDFlashDrafter) reads every matrix row for row, bit for bit,
// as LoadDFlashDrafter holds it, for each dtype the reader decodes (BF16, F16, F32) and whether the file's data is aligned or at
// an odd offset; it holds no mapping; its norms, device-byte estimate and K/V size are the held drafter's; it refuses host
// drafting and its matrix accessors (ErrDrafterStreamed); and a read after Close fails.
func TestOpenDFlashDrafter_matchesLoad(t *testing.T) {
	for _, c := range []struct {
		dtype string
		odd   bool
	}{{"BF16", false}, {"BF16", true}, {"F16", false}, {"F16", true}, {"F32", false}, {"F32", true}} {
		odd := c.odd
		t.Logf("dtype %s, odd offset %v", c.dtype, odd)
		dir := writeTinyDFlashAs(t, false, c.dtype)
		repadSafetensors(t, filepath.Join(dir, "model.safetensors"), odd)
		held, err := LoadDFlashDrafter(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer held.Close()
		if ten, err := held.st.Tensor("fc.weight"); err != nil {
			t.Fatal(err)
		} else if ten.DType != c.dtype {
			t.Fatalf("the fixture's dtype is %s, want %s", ten.DType, c.dtype)
		} else if _, err := ten.Uint16s(); c.dtype != "F32" && (err != nil) != odd {
			t.Fatalf("odd=%v: the fixture's payload alignment is not what the case needs (Uint16s: %v)", odd, err)
		}
		s, err := OpenDFlashDrafter(dir)
		if err != nil {
			t.Fatal(err)
		}
		if !s.Streamed() || held.Streamed() {
			t.Fatalf("odd=%v: Streamed is %v opened and %v loaded", odd, s.Streamed(), held.Streamed())
		}
		if s.st != nil {
			t.Errorf("%s odd=%v: the opened drafter still holds its mapping", c.dtype, odd)
		}
		same := func(what string, m *linalg.WeightMat, r DrafterRows) {
			if r.Rows != m.Rows() || r.Cols != m.Cols() {
				t.Fatalf("odd=%v, %s: rows reader %dx%d, held %dx%d", odd, what, r.Rows, r.Cols, m.Rows(), m.Cols())
			}
			got, want := make([]float32, r.Rows*r.Cols), make([]float32, r.Cols)
			for r0 := 0; r0 < r.Rows; r0 += 3 { // blocks of 3 rows, the last one short
				n := min(3, r.Rows-r0)
				if err := r.Read(r0, n, got[r0*r.Cols:(r0+n)*r.Cols]); err != nil {
					t.Fatalf("odd=%v, %s: read: %v", odd, what, err)
				}
			}
			for i := range r.Rows {
				m.Row(i, want)
				for j, w := range want {
					if math.Float32bits(got[i*r.Cols+j]) != math.Float32bits(w) {
						t.Fatalf("odd=%v, %s: row %d col %d: %v, held %v", odd, what, i, j, got[i*r.Cols+j], w)
					}
				}
			}
		}
		same("fc", held.DrafterFC(), s.DrafterFCRows())
		if !slices.Equal(s.DrafterHiddenNorm(), held.DrafterHiddenNorm()) || !slices.Equal(s.DrafterFinalNorm(), held.DrafterFinalNorm()) {
			t.Errorf("odd=%v: the trunk norms differ", odd)
		}
		for i := range held.DrafterGeometry().Layers {
			h, r := held.DrafterLayer(i), s.DrafterLayerRows(i)
			for j, m := range [7]*linalg.WeightMat{h.Q, h.K, h.V, h.O, h.Gate, h.Up, h.Down} {
				same([7]string{"q", "k", "v", "o", "gate", "up", "down"}[j], m, [7]DrafterRows{r.Q, r.K, r.V, r.O, r.Gate, r.Up, r.Down}[j])
			}
			if !slices.Equal(r.QNorm, h.QNorm) || !slices.Equal(r.KNorm, h.KNorm) || !slices.Equal(r.InputNorm, h.InputNorm) || !slices.Equal(r.PostAttnNorm, h.PostAttnNorm) {
				t.Errorf("odd=%v, layer %d: the norms differ", odd, i)
			}
		}
		if a, b := DrafterResidentBytesEstimate(s), DrafterResidentBytesEstimate(held); a != b || a == 0 {
			t.Errorf("odd=%v: device-byte estimate %d opened, %d loaded", odd, a, b)
		}
		if a, b := DrafterKVBytesPerPosition(s), DrafterKVBytesPerPosition(held); a != b {
			t.Errorf("odd=%v: K/V bytes per position %d opened, %d loaded", odd, a, b)
		}
		row, be := make([]float32, s.hidden), &cpuBackend{}
		if _, err := s.DraftBlock(be, [][]float32{row}, [][]float32{row}); !errors.Is(err, ErrDrafterStreamed) {
			t.Errorf("odd=%v: DraftBlock on an opened drafter: %v, want ErrDrafterStreamed", odd, err)
		}
		if _, err := s.FuseContext(be, [][]float32{make([]float32, 2*s.hidden)}); !errors.Is(err, ErrDrafterStreamed) {
			t.Errorf("odd=%v: FuseContext on an opened drafter: %v, want ErrDrafterStreamed", odd, err)
		}
		ctx := s.NewContext()
		if _, err := s.DraftBlockCtx(be, ctx, [][]float32{row}); !errors.Is(err, ErrDrafterStreamed) {
			t.Errorf("odd=%v: DraftBlockCtx on an opened drafter: %v, want ErrDrafterStreamed", odd, err)
		}
		if fc := s.DrafterFCRows(); fc.Read(0, 2, make([]float32, fc.Cols)) == nil {
			t.Errorf("odd=%v: a read into a destination one row short succeeded", odd)
		}
		for name, f := range map[string]func(){
			"DrafterFC":     func() { s.DrafterFC() },
			"DrafterLayer":  func() { s.DrafterLayer(0) },
			"ExtendContext": func() { s.ExtendContext(be, ctx, [][]float32{row}) },
		} {
			func() {
				defer func() {
					if r := recover(); r != ErrDrafterStreamed {
						t.Errorf("odd=%v: %s on an opened drafter: recovered %v, want ErrDrafterStreamed", odd, name, r)
					}
				}()
				f()
			}()
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("odd=%v: DrafterLayerRows on a held drafter did not panic", odd)
				}
			}()
			held.DrafterLayerRows(0)
		}()
		fc := s.DrafterFCRows()
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if err := fc.Read(0, 1, make([]float32, fc.Cols)); err == nil {
			t.Errorf("odd=%v: a read after Close succeeded", odd)
		}
	}
}

// TestNewBlockSpec_refusesAStreamedDrafterOnAHostThatCannotStream: a resident that hosts drafters but does not read row sources
// (no StreamingDrafterHost) refuses a drafter opened to stream before anything is read, rather than reaching its matrix
// accessors; the same drafter loaded with its weights held gets past that check.
func TestNewBlockSpec_refusesAStreamedDrafterOnAHostThatCannotStream(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	m.resident = &blockspecStubHost{}
	dir := writeTinyDFlash(t, false)
	s, err := OpenDFlashDrafter(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := m.NewBlockSpec(s, s.TargetLayerIDs()); err == nil || !strings.Contains(err.Error(), "opened to stream") {
		t.Errorf("a streamed drafter on a host that cannot stream: %v, want the refusal naming it", err)
	}
	held, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	if _, err := m.NewBlockSpec(held, held.TargetLayerIDs()); err == nil || strings.Contains(err.Error(), "opened to stream") {
		t.Errorf("control: a held drafter on the same host: %v, want the stub host's own refusal", err)
	}
}
