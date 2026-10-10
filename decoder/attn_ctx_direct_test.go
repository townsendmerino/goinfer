package decoder

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"testing"
)

// attendBatchedHeads's K=1 paths write each head's context straight into ctx (the AV kernels MatmulAVAcc64 /
// MatmulAVAcc64Group overwrite a contiguous destination) instead of into scratch and then copying (the
// per-head `ch`, the grouped `gCtx`, and Arm B's `fullCtx`).
//
// This pins every one of those paths to the ctx bits the scratch-and-copy version produced: a hash of ctx per
// case. The attention kernels accumulate in f64 and are bit-identical across architectures by design (the
// grouped NEON port and its Go fallback included), so one table serves every arch; if a case ever differs by
// architecture, key the table by GOARCH, do not loosen it.
//
//	GOINFER_CTX_RECORD=1 go test ./decoder/ -run TestAttendBatchedHeads_ctxBitsUnchanged -v   # prints the table: only from code whose ctx path is the reference
type ctxCase struct {
	name         string
	nH, nKV, hd  int
	nKeys        int
	workers      int  // pool size: 1 = the serial arm, >1 = the head fan-out (or Arm B for group 6)
	forceGrouped bool // withGroupedKernels, and GOINFER_ATTN_GROUPED=1
}

var ctxCases = []ctxCase{
	{"ungrouped serial, group 4, hd 64", 8, 2, 64, 200, 1, false},
	{"ungrouped head fan-out, group 4, hd 64", 8, 2, 64, 200, 2, false},
	{"ungrouped serial, group 1 (MHA), hd 97", 4, 4, 97, 203, 1, false},
	{"ungrouped serial, group 7, hd 128", 14, 2, 128, 177, 1, false},
	{"grouped Arm A serial, group 6, hd 64", 12, 2, 64, 200, 1, true},
	{"grouped Arm A head fan-out, group 6, hd 64", 12, 2, 64, 200, 2, true},
	{"grouped Arm B many workers, group 6, hd 97 (ragged splits)", 12, 2, 97, 203, maxAttnWorkers, true},
	{"grouped Arm B many workers, group 6, hd 128, 4 kv heads", 24, 4, 128, 301, maxAttnWorkers, true},
}

// ctxWant is filled from GOINFER_CTX_RECORD=1 output, recorded from the scratch-and-copy code (before the AV kernels wrote into ctx).
var ctxWant = map[string]string{
	"ungrouped serial, group 4, hd 64":                           "75ac92b7638d440d079dc39b7cc35c266a81c329cc98285a7100561d91bfd667",
	"ungrouped head fan-out, group 4, hd 64":                     "75ac92b7638d440d079dc39b7cc35c266a81c329cc98285a7100561d91bfd667",
	"ungrouped serial, group 1 (MHA), hd 97":                     "ee1ce75beed529d5ca84a7cb32044873116d8426fc2a2be0de9810d4cfea041e",
	"ungrouped serial, group 7, hd 128":                          "fc47cb4f6f2e8d27fbf2fbf12ae4387eeb699e55ef39e0403513dbb3a72cb296",
	"grouped Arm A serial, group 6, hd 64":                       "1fd4c569ead6750c9dea331a590b9091ffeb4fa4b375d2da43faaa1a6623ca0e",
	"grouped Arm A head fan-out, group 6, hd 64":                 "1fd4c569ead6750c9dea331a590b9091ffeb4fa4b375d2da43faaa1a6623ca0e",
	"grouped Arm B many workers, group 6, hd 97 (ragged splits)": "ad812b3164e92b910e108225c71af68370eeaeade4ae0b2ef1dcc6ed9156e528",
	"grouped Arm B many workers, group 6, hd 128, 4 kv heads":    "ed83d012e079234188ddb616a2943b30852522d67d9cc3773b8127a63a82fa5d",
}

func runCtxCase(t *testing.T, c ctxCase) string {
	t.Helper()
	if c.forceGrouped {
		withGroupedKernels(t)
		t.Setenv("GOINFER_ATTN_GROUPED", "1")
	} else {
		t.Setenv("GOINFER_ATTN_GROUPED", "0")
	}
	arch := &Architecture{NumHeads: c.nH, NumKVHeads: c.nKV, HeadDim: c.hd, AttnScale: 1 / math.Sqrt(float64(c.hd))}
	cache, q := syntheticDecodeCache(t, arch, c.nKeys, int64(c.nH*1000+c.hd))
	hd := c.hd
	group := c.nH / c.nKV
	pool := make([]headWorkerScratch, c.workers)
	for i := range pool {
		pool[i] = headWorkerScratch{
			qh: make([]float32, hd), ch: make([]float32, hd),
			scores:      make([]float32, c.nKeys),
			avAcc:       make([]float64, hd),
			groupScores: make([]float32, c.nKeys*group),
			groupCtx:    make([]float32, hd*group),
			groupAvAcc:  make([]float64, hd*group),
		}
	}
	pool[0].groupScoresCombined = make([]float32, c.nKeys*c.nKV*group)
	ctx := make([]float32, c.nH*hd)
	for i := range ctx { // poison: every element must be written by the kernel, whichever path wrote it
		ctx[i] = float32(math.NaN())
	}
	keys, vals := cache.Keys(0), cache.Vals(0)
	attendBatchedHeads(q, ctx, keys, vals, 0, cache, 0, c.nKeys-1, 1, true, arch, true, pool)
	h := sha256.New()
	var b [4]byte
	for _, v := range ctx {
		if math.IsNaN(float64(v)) {
			t.Fatalf("%s: a ctx element was never written", c.name)
		}
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestAttendBatchedHeads_ctxBitsUnchanged(t *testing.T) {
	record := os.Getenv("GOINFER_CTX_RECORD") != ""
	if len(ctxWant) == 0 && !record {
		t.Fatal("ctxWant is empty: record it from the pre-change code with GOINFER_CTX_RECORD=1")
	}
	for _, c := range ctxCases {
		t.Run(c.name, func(t *testing.T) {
			got := runCtxCase(t, c)
			if record {
				fmt.Printf("\t%q: %q,\n", c.name, got)
				return
			}
			if want := ctxWant[c.name]; got != want {
				t.Fatalf("ctx hash %s != the recorded %s: the attention context changed", got, want)
			}
		})
	}
}
