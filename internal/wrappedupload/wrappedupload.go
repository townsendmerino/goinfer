//go:build goinfer_testhooks

// Package wrappedupload tests the resident upload bridge on a sliding-window model, on whichever backend the caller
// names. An upload past the sliding window holds a sliding layer's last rows only, so a slot serves a prefix only if
// the window its next token attends to starts at or past those rows: kvmgr.Slots.SetFloor records that bound. Every
// resident backend shares the bridge and the slot table, so each runs these same tests.
package wrappedupload

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

const (
	promptRows = 256
	shortRows  = 128 // well below any floor promptRows gives

	// gapBar separates a read of the rows it should from one missing a row. A served prefix reads under it against a
	// cold prefill, one window row short reads over it.
	gapBar = 1e-3
)

// fixtures: gemma3-vl-tiny mixes sliding and global layers, mistral-tiny-window has sliding layers only.
var fixtures = []string{"gemma3-vl-tiny", "mistral-tiny-window"}

// skipShort skips under -short: each test loads the fixtures ten times, too slow for a CI runner's GPU inside its step's timeout.
// With no device, decoder's TestResidentUploadPrefill_wrappedFloor and kvmgr's TestSlots_floor read the floor.
func skipShort(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("device tests of the wrapped upload: run without -short on a real device")
	}
}

func options(backend string) decoder.Options {
	return decoder.Options{Backend: backend, Quant: "int4", KVPrecision: "f32", ResidentContext: 512}
}

// fixtureDir is fixture with room for the prompt: a directory linking to its files, with max_position_embeddings
// raised in its config. The caller's working directory is a package directory one below the repository root.
func fixtureDir(t *testing.T, fixture string) string {
	t.Helper()
	src, err := filepath.Abs(filepath.Join("..", "testdata", fixture))
	if err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for _, e := range ents {
		from, to := filepath.Join(src, e.Name()), filepath.Join(dir, e.Name())
		if e.Name() != "config.json" {
			if err := os.Symlink(from, to); err != nil {
				t.Fatal(err)
			}
			continue
		}
		b, err := os.ReadFile(from)
		if err != nil {
			t.Fatal(err)
		}
		out := regexp.MustCompile(`("max_position_embeddings":\s*)\d+`).ReplaceAll(b, []byte("${1}4096"))
		if err := os.WriteFile(to, out, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func load(t *testing.T, backend, fixture string) *decoder.Model {
	t.Helper()
	m, err := decoder.Load(fixtureDir(t, fixture), options(backend))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if m.ResidentForwardForTest() == nil {
		t.Fatalf("no resident on %s: %s", backend, m.ResidentDecline())
	}
	return m
}

// uploaded is a model holding a prompt computed on the CPU past the sliding window and uploaded through the bridge,
// and the shortest prefix that upload covers, worked out here from the rings rather than read from the decoder.
func uploaded(t *testing.T, backend, fixture string) (m *decoder.Model, ids []int, floor int) {
	t.Helper()
	m = load(t, backend, fixture)
	_, layers, _, _, _, _, vocab := m.Dims()
	ids = make([]int, promptRows)
	for i := range ids {
		ids[i] = (i*31 + 7) % vocab
	}
	cache := m.NewCache(promptRows + 8)
	for _, id := range ids {
		if _, err := m.ForwardForTest(id, cache); err != nil {
			t.Fatal(err)
		}
	}
	for l := range layers {
		if _, _, base := cache.LayerKVForTest(l); base > 0 {
			floor = max(floor, base+m.SlidingWindowResident()-1)
		}
	}
	if floor == 0 {
		t.Fatal("no layer's ring wrapped: the test reads nothing")
	}
	if err := m.ResidentUploadPrefillForTest(cache); err != nil {
		t.Fatal(err)
	}
	return m, ids, floor
}

func embed(m *decoder.Model, ids []int) [][]float32 {
	e := make([][]float32, len(ids))
	for i, id := range ids {
		e[i] = slices.Clone(m.EmbedResidentForTest(id))
	}
	return e
}

func gap(a, b []float32) float64 {
	worst := 0.0
	for i := range a {
		worst = max(worst, math.Abs(float64(a[i]-b[i])))
	}
	return worst
}

// Rows reads the device directly: past the prompt's end and at the floor a forward matches a cold prefill, and one
// row short of the floor it does not, because it attends to a row nothing uploaded. So the floor is exact, and it is
// the decoder's grant that keeps a request off those rows.
func Rows(t *testing.T, backend string) {
	skipShort(t)
	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			cold := load(t, backend, fixture)
			coldLast := func(ids []int) []float32 {
				t.Helper()
				out, err := cold.ResidentForwardForTest().ForwardN(embed(cold, ids), 0)
				if err != nil {
					t.Fatal(err)
				}
				return out[len(out)-1]
			}
			// Each read writes the device's rows, so each has its own upload.
			read := func(pos func(floor int) int, wantOver bool, name string) {
				t.Helper()
				m, ids, floor := uploaded(t, backend, fixture)
				p := pos(floor)
				tok := ids[0]
				if p < len(ids) {
					tok = ids[p]
				}
				out, err := m.ResidentForwardForTest().Forward(embed(m, []int{tok})[0], p)
				if err != nil {
					t.Fatal(err)
				}
				got := gap(out, coldLast(append(slices.Clone(ids[:p]), tok)))
				t.Logf("%s: %s, position %d of floor %d: worst logit gap against a cold prefill %.5f", fixture, name, p, floor, got)
				if over := got > gapBar; over != wantOver {
					t.Errorf("%s: gap %.5f, want over %g: %v", name, got, gapBar, wantOver)
				}
			}
			read(func(int) int { return promptRows }, false, "the prompt's end (control)")
			read(func(f int) int { return f }, false, "the floor")
			read(func(f int) int { return f - 1 }, true, "one row short of the floor")
			read(func(int) int { return shortRows }, true, "a much shorter prefix")
		})
	}
}

// Reuse: the decoder grants a slot holding a wrapped upload exactly the prefixes it covers, and each reply equals a
// cold prefill's. The replies are short and the weights random, so the grant is the sharper reading; Rows reads the
// logits.
func Reuse(t *testing.T, backend string) {
	skipShort(t)
	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			cold := load(t, backend, fixture)
			read := func(name string, prefix func(floor int) int, want func(floor int) int) {
				t.Helper()
				m, ids, floor := uploaded(t, backend, fixture)
				m.ResidentCommitIDsForTest(ids)
				_, _, _, _, _, _, vocab := m.Dims()
				prompt := slices.Clone(ids[:prefix(floor)])
				for i := range 8 {
					prompt = append(prompt, (i*17+3)%vocab)
				}
				got, g := generate(t, m, prompt)
				cold.ForgetResidentForTest()
				wantReply, _ := generate(t, cold, prompt)
				t.Logf("%s: %s: reused %d of %d (floor %d); reply %v, cold %v", fixture, name, g.PrefillReused, len(prompt), floor, got, wantReply)
				if g.PrefillReused != want(floor) {
					t.Errorf("%s: the decoder granted %d rows, want %d", name, g.PrefillReused, want(floor))
				}
				if !slices.Equal(got, wantReply) {
					t.Errorf("%s: the reply %v differs from a cold prefill's %v", name, got, wantReply)
				}
			}
			zero := func(int) int { return 0 }
			same := func(f int) int { return f }
			read("a much shorter prefix", func(int) int { return shortRows }, zero)
			read("one row short of the floor", func(f int) int { return f - 1 }, zero)
			read("the floor", same, same)
			read("the whole prompt", func(int) int { return promptRows }, func(int) int { return promptRows })
		})
	}
}

// Turns follows one slot across requests through Generate: a continuation past the floor keeps it, since those rows stay missing, so
// a shorter prefix after it is not granted; and a cold prefill writes from position 0, so a shorter prefix of what it wrote is.
func Turns(t *testing.T, backend string) {
	skipShort(t)
	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			cold := load(t, backend, fixture)
			m, ids, floor := uploaded(t, backend, fixture)
			m.ResidentCommitIDsForTest(ids)
			_, _, _, _, _, _, vocab := m.Dims()
			tail := func(seed int) []int {
				out := make([]int, 8)
				for i := range out {
					out[i] = (i*17 + seed) % vocab
				}
				return out
			}
			turn := func(name string, prompt []int, want int) {
				t.Helper()
				got, g := generate(t, m, prompt)
				cold.ForgetResidentForTest()
				wantReply, _ := generate(t, cold, prompt)
				t.Logf("%s: %s: reused %d of %d (floor %d); reply %v, cold %v", fixture, name, g.PrefillReused, len(prompt), floor, got, wantReply)
				if g.PrefillReused != want {
					t.Errorf("%s: the decoder granted %d rows, want %d", name, g.PrefillReused, want)
				}
				if !slices.Equal(got, wantReply) {
					t.Errorf("%s: the reply %v differs from a cold prefill's %v", name, got, wantReply)
				}
			}
			turn("a continuation past the floor", append(slices.Clone(ids), tail(3)...), promptRows)
			coldPrompt := append(slices.Clone(ids[:shortRows]), tail(5)...)
			turn("then a shorter prefix", coldPrompt, 0)
			turn("then a shorter prefix of the cold prefill", append(slices.Clone(coldPrompt[:shortRows/2]), tail(7)...), shortRows/2)
		})
	}
}

func generate(t *testing.T, m *decoder.Model, prompt []int) ([]int, *decoder.Generation) {
	t.Helper()
	ch, g := m.Generate(context.Background(), prompt, 8, decoder.SamplingParams{})
	var toks []int
	for tok := range ch {
		toks = append(toks, tok)
	}
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	return toks, g
}
