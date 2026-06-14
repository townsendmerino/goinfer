package decoder

import (
	"container/list"
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestExpertPager_lruEviction gates the pager's bookkeeping with no model: an LRU
// over experts capped by a byte budget, releasing (MADV_DONTNEED) the tail and
// faulting (MADV_WILLNEED) misses. The advise hook is captured so we can assert
// exactly which experts were faulted/released and in what order.
func TestExpertPager_lruEviction(t *testing.T) {
	experts := []*expertWeights{{}, {}, {}, {}}
	id := map[*byte]int{}
	p := &expertPager{
		budget: 25, // 10 bytes/expert ⇒ holds 2, the 3rd resident expert evicts the LRU tail
		ranges: map[*expertWeights][][]byte{},
		bytes:  map[*expertWeights]int64{},
		lru:    list.New(),
		pos:    map[*expertWeights]*list.Element{},
	}
	for i, ex := range experts {
		b := []byte{byte(i)} // a unique 1-byte span standing in for the expert's pages
		p.ranges[ex] = [][]byte{b}
		p.bytes[ex] = 10
		id[&b[0]] = i
	}
	var log []ev
	p.advise = func(b []byte, willNeed bool) error {
		log = append(log, ev{id[&b[0]], willNeed})
		return nil
	}

	p.touch(experts[0]) // miss → WILLNEED 0; resident {0}
	p.touch(experts[1]) // miss → WILLNEED 1; resident {1,0} = 20B
	p.touch(experts[2]) // miss → WILLNEED 2; 30>25 → evict tail 0 (DONTNEED 0); resident {2,1}

	want := []ev{{0, true}, {1, true}, {2, true}, {0, false}}
	if !equalEv(log, want) {
		t.Fatalf("phase 1 advise log = %v, want %v", log, want)
	}
	if h, m := p.stats(); h != 0 || m != 3 {
		t.Fatalf("phase 1 stats = (%d hits, %d misses), want (0, 3)", h, m)
	}

	log = nil
	p.touch(experts[1]) // hit (resident) — no advise
	p.touch(experts[0]) // miss → WILLNEED 0; 30>25 → evict tail (now expert 2) (DONTNEED 2)
	want = []ev{{0, true}, {2, false}}
	if !equalEv(log, want) {
		t.Fatalf("phase 2 advise log = %v, want %v", log, want)
	}
	if h, m := p.stats(); h != 1 || m != 4 {
		t.Fatalf("phase 2 stats = (%d hits, %d misses), want (1, 4)", h, m)
	}
}

type ev struct {
	expert   int
	willNeed bool
}

func equalEv(a, b []ev) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestMadvise_dontneedRefaultsIntact is the always-on proof of the property the
// whole expert pager rests on: MADV_DONTNEED on a read-only, file-backed mapping
// cannot corrupt data — the released pages re-fault from the file unchanged. We
// map a multi-page file, checksum it, MADV_DONTNEED the whole span, read it back,
// and require byte-identical contents (then MADV_WILLNEED + read again). If this
// holds, evicting an in-use expert is always safe regardless of timing.
func TestMadvise_dontneedRefaultsIntact(t *testing.T) {
	page := os.Getpagesize()
	want := make([]byte, 3*page)
	for i := range want {
		want[i] = byte((i*1103515245 + 12345) >> 7) // deterministic, non-trivial pattern
	}
	path := filepath.Join(t.TempDir(), "mapping.bin")
	if err := os.WriteFile(path, want, 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := mmapReadOnly(path)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer munmap(m)
	if !bytesEqual(m, want) {
		t.Fatal("mapping differs from file before any advice")
	}
	if err := madviseBytes(m, false); err != nil { // DONTNEED: release all pages
		t.Fatalf("MADV_DONTNEED: %v", err)
	}
	if !bytesEqual(m, want) { // touching re-faults from the file
		t.Fatal("data changed after MADV_DONTNEED + re-fault — eviction is NOT lossless")
	}
	if err := madviseBytes(m, true); err != nil { // WILLNEED: hint back in
		t.Fatalf("MADV_WILLNEED: %v", err)
	}
	if !bytesEqual(m, want) {
		t.Fatal("data changed after MADV_WILLNEED")
	}
}

// TestExpertPaging_bitExact is the end-to-end correctness gate for idea #2: paging
// experts out of the read-only mapping must not change a single output token. It
// loads a real large-expert MoE .giw (GOINFER_MOE_GIW) twice — fully resident and
// with a 1-expert budget (so every token evicts an in-use expert and re-faults it)
// — and asserts the greedy decodes are byte-identical and that eviction actually
// happened (misses > 0). Skipped without the asset (no small MoE .giw round-trips
// on this box — the tiny hybrid fixture's DeltaNet weights aren't .giw-serializable
// and its experts are sub-page anyway). The page-granular safety it relies on is
// proven model-free by TestMadvise_dontneedRefaultsIntact; the LRU policy by
// TestExpertPager_lruEviction.
func TestExpertPaging_bitExact(t *testing.T) {
	giwPath := os.Getenv("GOINFER_MOE_GIW")
	if giwPath == "" {
		t.Skip("set GOINFER_MOE_GIW=<large-expert MoE .giw> for the end-to-end paging gate")
	}

	full, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("load full .giw: %v", err)
	}
	defer full.Close()

	// WeightCacheBytes 1 clamps to a single expert: with top-k > 1 every token
	// touches more experts than fit, forcing evict-then-reuse (the re-fault path).
	paged, err := Load(giwPath, Options{StreamWeights: true, WeightCacheBytes: 1})
	if err != nil {
		t.Fatalf("load paged .giw: %v", err)
	}
	defer paged.Close()
	if paged.pager == nil {
		t.Fatal("paged load built no pager (expected mmap-backed MoE experts)")
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := greedyN(t, full, prompt, 16)
	b := greedyN(t, paged, prompt, 16)
	if len(a) == 0 {
		t.Fatal("no tokens generated")
	}
	if !slicesEqualInt(a, b) {
		t.Fatalf("paging changed the decode:\n full:  %v\n paged: %v", a, b)
	}
	if _, misses := paged.pager.stats(); misses == 0 {
		t.Fatal("pager recorded zero misses — eviction/re-fault path was not exercised")
	} else {
		t.Logf("paged decode byte-identical over %d tokens; pager misses=%d (evict/re-fault proven lossless)", len(a), misses)
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func greedyN(t *testing.T, m *Model, prompt []int, n int) []int {
	t.Helper()
	out, gen := m.Generate(context.Background(), prompt, n, SamplingParams{Temperature: 0})
	var got []int
	for tok := range out {
		got = append(got, tok)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return got
}

func slicesEqualInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
