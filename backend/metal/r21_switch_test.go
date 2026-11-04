//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"
	"time"
)

// TestR21_switchCost measures R-21's third item (docs/tasks/task-recompute-audit.md): forwardMultiInto calls stopExec on
// every batched step, which is a no-op unless the one-sequence executor or chain is running, so the cost is paid once
// per switch between one sequence and several. On the 1.5B (mc3LoadCheckpoint) with 2 KV slots filled to 256 tokens,
// 32-step blocks: steady B=2 batched steps, steady one-sequence steps on slot 0, and the two alternating step by step.
// The one-sequence step is the encode-ahead executor (ForwardEmbPipe) or the greedy chain (GreedyChainNext, restarted
// with chainStart after each batched step), the two ways a lone sequence decodes. Per-step medians; a switch pair's
// cost is (alternating one-seq + alternating batched) - (steady one-seq + steady batched). Every block restarts at
// the same positions, so attention depth does not drift between them. Blocks rotate order over the reps. By day,
// in-process, exploratory.
//
//	GOINFER_R21_SWITCH=1 GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestR21_switchCost$' -v ./metal/
func TestR21_switchCost(t *testing.T) {
	if os.Getenv("GOINFER_R21_SWITCH") != "1" {
		t.Skip("set GOINFER_R21_SWITCH=1 (times a real checkpoint)")
	}
	const D, steps = 256, 32
	m, r := mc3LoadCheckpoint(t, 2, 1024)
	defer m.Close()
	if r.batch == nil {
		t.Fatalf("no batched step on this resident: %s", r.batchIneligible())
	}
	mr := &metalResident{r: r}
	chainOK := mr.GreedyChainAvailable()
	ids := make([]int, D)
	for i := range ids {
		ids[i] = (i*7919 + 11) % 20000
	}
	mc3Fill(t, r, 0, ids)
	mc3Fill(t, r, 1, ids)
	emb := mc3Emb(r, 42)
	t0 := time.Now()
	batched := func(posA, posB int) float64 {
		st := time.Now()
		if _, _, err := r.forwardMultiInto([]batchSeq{{slot: 0, pos: posA, emb: emb}, {slot: 1, pos: posB, emb: emb}}, true); err != nil {
			t.Fatal(err)
		}
		return float64(time.Since(st).Microseconds()) / 1e3
	}
	type mode struct {
		name   string
		one    func(pos int, fresh bool) float64 // one step on slot 0; fresh: the previous step was batched
		finish func()
	}
	modes := []mode{{
		name: "executor",
		one: func(pos int, _ bool) float64 {
			st := time.Now()
			if err := r.useKVSlot(0); err != nil {
				t.Fatal(err)
			}
			r.ForwardEmbPipe(emb, pos)
			return float64(time.Since(st).Microseconds()) / 1e3
		},
		finish: r.stopExec,
	}}
	if chainOK {
		modes = append(modes, mode{
			name: "chain",
			one: func(pos int, fresh bool) float64 {
				st := time.Now()
				if fresh {
					if err := r.useKVSlot(0); err != nil {
						t.Fatal(err)
					}
					if err := r.chainStart(42, pos); err != nil {
						t.Fatal(err)
					}
				}
				if _, err := r.chainNext(); err != nil {
					t.Fatal(err)
				}
				return float64(time.Since(st).Microseconds()) / 1e3
			},
			finish: r.stopChain,
		})
	} else {
		t.Logf("the greedy chain is not available on this resident; the executor only")
	}
	reps := auditReps(7)
	for _, md := range modes {
		res := map[string][]float64{}
		run := func(block string) {
			var one, bat []float64
			posA, posB := D, D
			switch block {
			case "steady-batched":
				for range steps {
					bat = append(bat, batched(posA, posB))
					posA, posB = posA+1, posB+1
				}
			case "steady-one":
				for i := range steps {
					one = append(one, md.one(posA, i == 0))
					posA++
				}
				md.finish()
			case "alternating":
				for range steps {
					one = append(one, md.one(posA, true))
					posA++
					bat = append(bat, batched(posA, posB))
					posA, posB = posA+1, posB+1
				}
				md.finish()
			}
			if len(one) > 1 && block == "steady-one" {
				one = one[1:] // the block's first step starts the executor or chain: the steady cost is the rest
			}
			if len(one) > 0 {
				res[block+"/one"] = append(res[block+"/one"], auditMedian(one))
			}
			if len(bat) > 0 {
				res[block+"/batched"] = append(res[block+"/batched"], auditMedian(bat))
			}
		}
		blocks := []string{"steady-batched", "steady-one", "alternating"}
		for rep := range reps + 1 {
			for k := range blocks {
				run(blocks[(k+rep)%len(blocks)])
			}
			if rep == 0 { // warm-up rep: drop it
				for key, v := range res {
					res[key] = v[:len(v)-1]
				}
			}
		}
		sb, so := auditMedian(res["steady-batched/batched"]), auditMedian(res["steady-one/one"])
		ab, ao := auditMedian(res["alternating/batched"]), auditMedian(res["alternating/one"])
		pair := make([]float64, reps)
		for i := range pair {
			pair[i] = res["alternating/one"][i] + res["alternating/batched"][i] - res["steady-one/one"][i] - res["steady-batched/batched"][i]
		}
		auditHB("r21-switch", t0, "%s: steady batched %.3f ms, steady one-seq %.3f ms; alternating batched %.3f ms (+%.3f), one-seq %.3f ms (+%.3f)",
			md.name, sb, so, ab, ab-sb, ao, ao-so)
		auditHB("r21-switch", t0, "RESULT %s: a switch pair (one-seq -> batched -> one-seq) costs median %.3f ms over the steady steps (per rep %s)",
			md.name, auditMedian(pair), auditFmt3(pair))
	}
}
