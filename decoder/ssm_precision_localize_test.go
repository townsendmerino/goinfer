package decoder

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Precision-localization for the granite-resident quality gap (measurement only,
// docs/ssm-int8-quality.md). All runs use the cpu backend (f32 weights throughout),
// so weight precision is excluded BY CONSTRUCTION — the only variables are the SSM
// compute precision (f64 vs f32, ssmForceF32) and a forced high-precision routing
// override. One model load, three teacher-forced passes vs the f64 reference (R1):
//
//	E1: cpu f32 SSM                — does downcasting the SSM compute to f32 reproduce
//	                                 the ~66% resident drop? (confirms SSM compute is the
//	                                 lever, weights excluded)
//	E2: cpu f32 SSM + f64 routing  — force R1's (f64) expert selection+weights onto the
//	                                 f32-SSM forward. Recovers → router-selection island;
//	                                 doesn't → the whole SSM recurrence needs ~f64.
const ssmLocHeldOut = `The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. ` +
	`It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. ` +
	`Locally nicknamed "La dame de fer", it was constructed as the centerpiece of the 1889 World's Fair. ` +
	`The tower is 330 metres tall, about the same height as an 81-storey building, and the tallest structure in Paris. ` +
	`Its base is square, measuring 125 metres on each side. During its construction, the Eiffel Tower surpassed the ` +
	`Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until ` +
	`the Chrysler Building in New York City was finished in 1930. The tower has three levels for visitors, with ` +
	`restaurants on the first and second levels. Photosynthesis is the process by which green plants convert light ` +
	`energy into chemical energy stored in glucose. Water boils at one hundred degrees Celsius at sea level pressure.`

func TestSSMPrecisionLocalize(t *testing.T) {
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("ssm precision-localization — slow (granite cpu load + 3 forwards); set GOINFER_SSM_QUALITY=1")
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode(ssmLocHeldOut, true)
	if len(ids) > 220 {
		ids = ids[:220]
	}
	N := len(ids) - 1
	t.Logf("held-out: %d tokens scored", N)

	softmax := func(lg []float32) []float64 {
		mx := float64(lg[0])
		for _, v := range lg {
			if float64(v) > mx {
				mx = float64(v)
			}
		}
		out := make([]float64, len(lg))
		var s float64
		for i, v := range lg {
			e := math.Exp(float64(v) - mx)
			out[i] = e
			s += e
		}
		for i := range out {
			out[i] /= s
		}
		return out
	}
	argmax := func(v []float32) int {
		bi, bv := 0, v[0]
		for i, x := range v {
			if x > bv {
				bv, bi = x, i
			}
		}
		return bi
	}
	top5 := func(p []float64) map[int]bool {
		type kv struct {
			i int
			p float64
		}
		a := make([]kv, len(p))
		for i, v := range p {
			a[i] = kv{i, v}
		}
		sort.Slice(a, func(x, y int) bool { return a[x].p > a[y].p })
		m := map[int]bool{}
		for i := 0; i < 5 && i < len(a); i++ {
			m[a[i].i] = true
		}
		return m
	}

	m, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	run := func() ([][]float64, []int) {
		cache := m.NewCache(len(ids) + 1)
		dist := make([][]float64, N)
		arg := make([]int, N)
		for i := range N {
			lg, e := m.ForwardForTest(ids[i], cache)
			if e != nil {
				t.Fatal(e)
			}
			dist[i] = softmax(lg)
			arg[i] = argmax(lg)
		}
		return dist, arg
	}

	// ---- R1: cpu f64 SSM (reference) + record its routing decisions ----
	ssmForceF32 = false
	moeSelTrace = make([][]int, 0, 1<<14)
	moeWtsTrace = make([][]float32, 0, 1<<14)
	r1Dist, r1Arg := run()
	routeIdx, routeWts := moeSelTrace, moeWtsTrace
	moeSelTrace, moeWtsTrace = nil, nil
	r1NLL := 0.0
	for i := range N {
		r1NLL += -math.Log(math.Max(r1Dist[i][ids[i+1]], 1e-12))
	}
	t.Logf("R1 (cpu f64 SSM): perplexity=%.3f; routing calls recorded=%d", math.Exp(r1NLL/float64(N)), len(routeIdx))

	score := func(name string, dist [][]float64, arg []int) {
		agree, kl, t5, nll := 0, 0.0, 0.0, 0.0
		for i := range N {
			if arg[i] == r1Arg[i] {
				agree++
			}
			r1t5, xt5 := top5(r1Dist[i]), top5(dist[i])
			inter := 0
			for kk := range r1t5 {
				if xt5[kk] {
					inter++
				}
			}
			t5 += float64(inter) / 5
			for v, p := range r1Dist[i] {
				if p > 1e-9 {
					kl += p * (math.Log(p) - math.Log(math.Max(dist[i][v], 1e-12)))
				}
			}
			nll += -math.Log(math.Max(dist[i][ids[i+1]], 1e-12))
		}
		t.Logf("%-34s agreement=%.1f%%  meanKL=%.4f  top5=%.1f%%  perplexity=%.3f (R1 %.3f)",
			name, 100*float64(agree)/float64(N), kl/float64(N), 100*t5/float64(N),
			math.Exp(nll/float64(N)), math.Exp(r1NLL/float64(N)))
	}

	// ---- E1: cpu f32 SSM (record routing to count selection divergence) ----
	ssmForceF32 = true
	moeSelTrace = make([][]int, 0, 1<<14)
	moeWtsTrace = make([][]float32, 0, 1<<14)
	e1Dist, e1Arg := run()
	e1Idx := moeSelTrace
	moeSelTrace, moeWtsTrace = nil, nil
	score("E1: cpu f32 SSM", e1Dist, e1Arg)
	flips, tot := 0, 0
	for c := range routeIdx {
		set := map[int]bool{}
		for _, e := range routeIdx[c] {
			set[e] = true
		}
		same := len(e1Idx[c]) == len(routeIdx[c])
		for _, e := range e1Idx[c] {
			if !set[e] {
				same = false
			}
		}
		if !same {
			flips++
		}
		tot++
	}
	t.Logf("E1 vs R1 routing: %d/%d MoE calls picked a different expert set (%.1f%%)",
		flips, tot, 100*float64(flips)/float64(tot))

	// ---- E2: cpu f32 SSM + forced f64 (R1) routing (router-null control) ----
	ssmForceF32 = true
	moeSelOverride, moeWtsOverride, moeOverridePos = routeIdx, routeWts, 0
	e2Dist, e2Arg := run()
	moeSelOverride, moeWtsOverride = nil, nil
	ssmForceF32 = false
	score("E2: cpu f32 SSM + f64 routing", e2Dist, e2Arg)

	// E1 refuted "f32 SSM compute is the lever" (100%). The one thing R2 keeps f32 but
	// the resident quantizes is the int8 MAMBA projections (W8A8 weights + int8 act).
	// D1 isolates that on the otherwise-perfect CPU path; D2 then forces R1's (f64)
	// routing onto it — the REAL router-island test, on the path that reproduces the gap.
	flipCount := func(idx [][]int) (int, int) {
		f, n := 0, 0
		for c := range routeIdx {
			set := map[int]bool{}
			for _, e := range routeIdx[c] {
				set[e] = true
			}
			same := len(idx[c]) == len(routeIdx[c])
			for _, e := range idx[c] {
				if !set[e] {
					same = false
				}
			}
			if !same {
				f++
			}
			n++
		}
		return f, n
	}

	// ---- D1: cpu int8 mamba (W8A8), f64 SSM otherwise ----
	ssmQ8CPU = true
	moeSelTrace = make([][]int, 0, 1<<14)
	moeWtsTrace = make([][]float32, 0, 1<<14)
	d1Dist, d1Arg := run()
	d1Idx := moeSelTrace
	moeSelTrace, moeWtsTrace = nil, nil
	score("D1: cpu int8 mamba (W8A8)", d1Dist, d1Arg)
	if df, dn := flipCount(d1Idx); dn > 0 {
		t.Logf("D1 vs R1 routing: %d/%d MoE calls picked a different expert set (%.1f%%)",
			df, dn, 100*float64(df)/float64(dn))
	}

	// ---- D2: cpu int8 mamba + forced f64 (R1) routing ----
	moeSelOverride, moeWtsOverride, moeOverridePos = routeIdx, routeWts, 0
	d2Dist, d2Arg := run()
	moeSelOverride, moeWtsOverride = nil, nil
	ssmQ8CPU = false
	score("D2: cpu int8 mamba + f64 routing", d2Dist, d2Arg)
}
