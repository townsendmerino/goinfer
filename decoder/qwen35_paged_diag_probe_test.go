package decoder

import (
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestQwen35PagedDiag_weightKinds is a throwaway diagnostic (docs/prompts/
// qwen35b-paged-decode-diagnostic.md): size the "f32-scratch handicap" —
// the Phase 0 brief's last open item — at the REAL 35B-A3B checkpoint's
// actual shapes, rather than the 27.8B Qwen3.8 numbers docs/queue-
// performance.md's P12 entry recorded. P12 fixed deltaNetWeights/
// qwenAttnWeights to honor Options.Quant; this confirms that holds for
// this specific streamed prequant .giw and reports the actual byte split.
//
//	GOINFER_STUBPROBE_GIW=~/models/qwen3.5-35b-a3b-int4.giw \
//	go test ./decoder -run TestQwen35PagedDiag_weightKinds -v
func TestQwen35PagedDiag_weightKinds(t *testing.T) {
	path := os.Getenv("GOINFER_STUBPROBE_GIW")
	if path == "" {
		t.Skip("set GOINFER_STUBPROBE_GIW to a real qwen3.5-35B-A3B .giw path")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	weights, _, err := giw.Read(data)
	if err != nil {
		t.Fatalf("bundle read: %v", err)
	}
	w, err := LoadSerializedWeights(weights)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	byteSize := func(kind string, rows, cols int) int64 {
		n := int64(rows) * int64(cols)
		switch kind {
		case "int4":
			return n/2 + int64(rows)*8 // packed nibbles + per-group scale, approx
		case "int8":
			return n + int64(rows)*4
		default: // f32 / native
			return n * 4
		}
	}
	f32VecBytes := func(n int) int64 { return int64(n) * 4 }

	var deltaQuantizable, deltaF32Vecs, qattnQuantizable, qattnF32Norms int64
	var expertBytes, routerBytes, sharedExpertBytes int64
	deltaLayers, qattnLayers := 0, 0
	kinds := map[string]int{}

	for i := range w.Layers {
		l := &w.Layers[i]
		if l.delta != nil {
			deltaLayers++
			d := l.delta
			for name, m := range map[string]*struct {
				rows, cols int
				kind       string
			}{
				"inProjQKV": {d.inProjQKV.Rows(), d.inProjQKV.Cols(), d.inProjQKV.Kind()},
				"inProjZ":   {d.inProjZ.Rows(), d.inProjZ.Cols(), d.inProjZ.Kind()},
				"outProj":   {d.outProj.Rows(), d.outProj.Cols(), d.outProj.Kind()},
			} {
				deltaQuantizable += byteSize(m.kind, m.rows, m.cols)
				kinds["delta."+name+":"+m.kind]++
			}
			deltaF32Vecs += f32VecBytes(len(d.inProjB)) + f32VecBytes(len(d.inProjA)) +
				f32VecBytes(len(d.convW)) + f32VecBytes(len(d.dtBias)) +
				f32VecBytes(len(d.negExpA)) + f32VecBytes(len(d.normW))
		}
		if l.qattn != nil {
			qattnLayers++
			a := l.qattn
			for name, m := range map[string]*struct {
				rows, cols int
				kind       string
			}{
				"qProj": {a.qProj.Rows(), a.qProj.Cols(), a.qProj.Kind()},
				"kProj": {a.kProj.Rows(), a.kProj.Cols(), a.kProj.Kind()},
				"vProj": {a.vProj.Rows(), a.vProj.Cols(), a.vProj.Kind()},
				"oProj": {a.oProj.Rows(), a.oProj.Cols(), a.oProj.Kind()},
			} {
				qattnQuantizable += byteSize(m.kind, m.rows, m.cols)
				kinds["qattn."+name+":"+m.kind]++
			}
			qattnF32Norms += f32VecBytes(len(a.qNorm)) + f32VecBytes(len(a.kNorm))
		}
		if l.Router.Rows() > 0 {
			routerBytes += byteSize(l.Router.Kind(), l.Router.Rows(), l.Router.Cols())
			kinds["router:"+l.Router.Kind()]++
		}
		for e := range l.Experts {
			ex := &l.Experts[e]
			expertBytes += byteSize(ex.Gate.Kind(), ex.Gate.Rows(), ex.Gate.Cols())
			expertBytes += byteSize(ex.Up.Kind(), ex.Up.Rows(), ex.Up.Cols())
			expertBytes += byteSize(ex.Down.Kind(), ex.Down.Rows(), ex.Down.Cols())
			kinds["expert.gate:"+ex.Gate.Kind()]++
		}
		if l.SharedExpert.Gate.Rows() > 0 {
			sharedExpertBytes += byteSize(l.SharedExpert.Gate.Kind(), l.SharedExpert.Gate.Rows(), l.SharedExpert.Gate.Cols())
			sharedExpertBytes += byteSize(l.SharedExpert.Up.Kind(), l.SharedExpert.Up.Rows(), l.SharedExpert.Up.Cols())
			sharedExpertBytes += byteSize(l.SharedExpert.Down.Kind(), l.SharedExpert.Down.Rows(), l.SharedExpert.Down.Cols())
		}
	}

	embedBytes := byteSize(w.Embed.Kind(), w.Embed.Rows(), w.Embed.Cols())
	lmHeadBytes := embedBytes
	if w.LMHead.Rows() > 0 {
		lmHeadBytes = byteSize(w.LMHead.Kind(), w.LMHead.Rows(), w.LMHead.Cols())
	}

	t.Logf("layers: %d delta, %d qattn (of %d total)", deltaLayers, qattnLayers, len(w.Layers))
	t.Logf("kinds seen: %v", kinds)
	t.Logf("DeltaNet quantizable projections (inProjQKV/Z/outProj), ALL layers: %.1f MB", float64(deltaQuantizable)/1e6)
	t.Logf("DeltaNet small f32 vectors (A/B/conv/dtBias/negExpA/normW), ALL layers: %.1f MB", float64(deltaF32Vecs)/1e6)
	t.Logf("qattn quantizable projections (q/k/v/oProj), ALL layers: %.1f MB", float64(qattnQuantizable)/1e6)
	t.Logf("qattn f32 norms (q/kNorm), ALL layers: %.3f MB", float64(qattnF32Norms)/1e6)
	t.Logf("router bytes, ALL layers: %.2f MB", float64(routerBytes)/1e6)
	t.Logf("expert bytes, ALL layers: %.2f GB", float64(expertBytes)/1e9)
	t.Logf("shared-expert bytes, ALL layers: %.2f MB", float64(sharedExpertBytes)/1e6)
	t.Logf("embed bytes: %.1f MB, lmHead bytes: %.1f MB (tied=%v)", float64(embedBytes)/1e6, float64(lmHeadBytes)/1e6, w.LMHead.Rows() == 0)

	total := deltaQuantizable + deltaF32Vecs + qattnQuantizable + qattnF32Norms + routerBytes + expertBytes + sharedExpertBytes + embedBytes
	if w.LMHead.Rows() > 0 {
		total += lmHeadBytes
	}
	fmt.Printf("TOTAL resident-weight bytes (approx): %.2f GB\n", float64(total)/1e9)
}
