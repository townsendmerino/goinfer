//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Phase-A wiring diff (docs/ssm-int8-quality.md). The GPU mamba kernels are proven correct on
// real inputs IN ISOLATION (TestMambaRealInputParity). This test captures the resident's ACTUAL
// per-token layer-0 kernel I/O from the full plan (in_proj output proj, conv, ssm-out y, mixer
// gated) and replays the resident's OWN proj through the CPU reference mixer (mamba2.go steps
// 2-5) with fresh evolving state. Per stage (conv/ssm/gatedNorm):
//   - resident matches CPU-on-resident-proj  → the mixer-in-plan is correct on its proj; the bug
//     is upstream (proj: in_proj GEMV / activation quant) or in attn/MoE.
//   - resident DIVERGES                       → a PLAN/STATE bug (the kernels run differently in
//     the full command buffer than in isolation — barrier/race/state corruption).
func TestMambaResidentCapture(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("resident mamba capture — needs granite model; set GOINFER_SSM_QUALITY=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode(ssmHeldOut, true)
	if len(ids) > 60 {
		ids = ids[:60]
	}

	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load resident: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Fatal("not resident")
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("resident forward is %T, not *residentDecoder", m.ResidentForwardForTest())
	}

	nHeads, headDim, dState, nGroups, dConv, _, _, _, _, gok := m.GraniteResidentParams()
	if !gok {
		t.Fatal("not granite")
	}
	dInner := nHeads * headDim
	gSize := nGroups * dState
	convDim := dInner + 2*gSize
	K := dConv
	projDim := 2*dInner + 2*gSize + nHeads
	repeat := nHeads / nGroups
	eps := float64(m.NormEps())
	_, convW, convB, aLog, dW, dtBias, normW, _ := m.GraniteMambaWeights(0)

	// CPU reference mixer (mamba2.go steps 2-5), fed the resident's OWN proj, fresh state.
	var win [][]float32
	ssm := make([]float32, nHeads*headDim*dState)
	silu := func(x float32) float32 { return x / (1 + float32(math.Exp(-float64(x)))) }
	cpuMixer := func(proj []float32) (conv, y, gated []float32) {
		z := proj[0:dInner]
		xBC := proj[dInner : dInner+convDim]
		dt := proj[projDim-nHeads:]
		conv = make([]float32, convDim)
		for c := range convDim {
			s := convB[c] + convW[c*K+(K-1)]*xBC[c]
			for j := 0; j < K-1; j++ {
				if idx := len(win) - (K - 1) + j; idx >= 0 {
					s += convW[c*K+j] * win[idx][c]
				}
			}
			conv[c] = silu(s)
		}
		win = append(win, append([]float32(nil), xBC...))
		if len(win) > K-1 {
			win = win[len(win)-(K-1):]
		}
		x, B, C := conv[:dInner], conv[dInner:dInner+gSize], conv[dInner+gSize:]
		y = make([]float32, dInner)
		for head := range nHeads {
			g := head / repeat
			A := -math.Exp(float64(aLog[head]))
			dthf := dt[head] + dtBias[head]
			var dth float32
			if dthf > 20 {
				dth = dthf
			} else {
				dth = float32(math.Log1p(math.Exp(float64(dthf))))
			}
			dA := float32(math.Exp(float64(dth) * A))
			for pi := range headDim {
				sBase := (head*headDim + pi) * dState
				dx := dth * x[head*headDim+pi]
				var acc float32
				for n := range dState {
					ssm[sBase+n] = ssm[sBase+n]*dA + dx*B[g*dState+n]
					acc += ssm[sBase+n] * C[g*dState+n]
				}
				y[head*headDim+pi] = acc + dW[head]*x[head*headDim+pi]
			}
		}
		gated = make([]float32, dInner)
		for i := range gated {
			gated[i] = y[i] * silu(z[i])
		}
		var ss float64
		for _, v := range gated {
			ss += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(ss/float64(dInner)+eps))
		for i := range gated {
			gated[i] = normW[i] * gated[i] * inv
		}
		return conv, y, gated
	}

	nLayers := 0
	{
		_, nl, _, _, _, _, _ := m.Dims()
		nLayers = nl
	}
	nMamba := 0
	for i := 0; i < nLayers; i++ {
		if m.GraniteMambaLayer(i) {
			nMamba++
		}
	}

	residentProj := make([][]float32, 0, len(ids))
	worstConv, worstY, worstG := 1.0, 1.0, 1.0
	firstBadConv, firstBadY, firstBadG := -1, -1, -1
	for pos := 0; pos < len(ids); pos++ {
		emb := m.EmbedResidentForTest(ids[pos])
		if _, err := rd.Forward(emb, pos); err != nil {
			t.Fatal(err)
		}
		rProj, rConv, rY, rGated, rerr := rd.runner.ReadMambaCap(projDim, convDim, dInner)
		if rerr != nil {
			t.Fatalf("ReadMambaCap: %v", rerr)
		}
		residentProj = append(residentProj, rProj)
		cConv, cY, cGated := cpuMixer(rProj)
		cc, _ := cosSim(cConv, rConv)
		cy, _ := cosSim(cY, rY)
		cg, mg := cosSim(cGated, rGated)
		if cc < worstConv {
			worstConv = cc
		}
		if cy < worstY {
			worstY = cy
		}
		if cg < worstG {
			worstG = cg
		}
		if cc < 0.999 && firstBadConv < 0 {
			firstBadConv = pos
		}
		if cy < 0.999 && firstBadY < 0 {
			firstBadY = pos
		}
		if cg < 0.999 && firstBadG < 0 {
			firstBadG = pos
		}
		if pos < 4 || cg < 0.999 {
			t.Logf("  pos %2d: conv cos=%.6f  y cos=%.6f  gated cos=%.6f (maxAbs %.4g)", pos, cc, cy, cg, mg)
		}
	}
	t.Logf("RESIDENT-vs-CPU(on resident proj): worst conv=%.6f (firstBad %d), y=%.6f (firstBad %d), gated=%.6f (firstBad %d)",
		worstConv, firstBadConv, worstY, firstBadY, worstG, firstBadG)
	if worstG > 0.999 {
		t.Logf("=> mixer-in-plan CORRECT on the resident's own proj => bug is upstream (proj / attn / MoE), not the mamba kernels")
	} else if firstBadConv >= 0 && firstBadConv <= firstBadY {
		t.Logf("=> CONV stage diverges first (pos %d) => conv kernel/state in the full plan", firstBadConv)
	} else {
		t.Logf("=> SSM/gatedNorm stage diverges => ssm state or gatedNorm in the full plan")
	}

	// Is the resident's PROJ itself right? Compare to mamba2Step's layer-0 proj on the same
	// teacher-forced ids (int8 regime). pos 0 is clean (input = emb·EmbMul, no upstream), so a
	// pos-0 mismatch ⇒ in_proj GEMV / activation-quant bug; a clean pos-0 but growing mismatch
	// ⇒ the residual stream is corrupted upstream (attn / MoE / out_proj), not in_proj.
	os.Unsetenv("GOINFER_SSM_RESIDENT")
	decoder.SetSSMQ8CPU(true)
	var cpuProj [][]float32
	cnt := 0
	decoder.SetMambaCapHook(func(proj, gated []float32) {
		if cnt%nMamba == 0 {
			cpuProj = append(cpuProj, append([]float32(nil), proj...))
		}
		cnt++
	})
	mc, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load cpu ref: %v", err)
	}
	cache := mc.NewCache(len(ids) + 1)
	for pos := 0; pos < len(ids); pos++ {
		if _, err := mc.ForwardForTest(ids[pos], cache); err != nil {
			t.Fatal(err)
		}
	}
	decoder.SetMambaCapHook(nil)
	decoder.SetSSMQ8CPU(false)
	mc.Close()
	worstP, firstBadP := 1.0, -1
	for pos := 0; pos < len(ids) && pos < len(cpuProj); pos++ {
		cp, mp2 := cosSim(cpuProj[pos], residentProj[pos])
		if cp < worstP {
			worstP = cp
		}
		if cp < 0.999 && firstBadP < 0 {
			firstBadP = pos
		}
		if pos < 4 || cp < 0.999 {
			t.Logf("  proj pos %2d: resident-vs-cpu cosine=%.6f maxAbs=%.4g", pos, cp, mp2)
		}
	}
	t.Logf("PROJ resident-vs-mamba2Step: worst=%.6f, first divergence pos=%d", worstP, firstBadP)
	if firstBadP == 0 {
		t.Logf("=> in_proj GEMV / activation-quant is WRONG at pos 0 (clean input) => the bug is the resident in_proj")
	} else if firstBadP > 0 {
		t.Logf("=> proj clean at pos 0, diverges at pos %d => upstream stream corruption (attn / MoE / out_proj), not in_proj", firstBadP)
	} else {
		t.Logf("=> proj matches throughout => in_proj fine; the divergence must be in out_proj / attn / MoE residual adds")
	}
}
