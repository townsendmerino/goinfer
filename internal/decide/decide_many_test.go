package decide

import (
	"context"
	"math"
	"strings"
	"testing"
)

// DecideMany (D8): the questions about one state go to the model in ONE many-prompt call, and each gets the answer Decide gives it alone. The model here is a
// deterministic function of the prompt's ids, so a distribution can only come out right if the right prompt reached the right question.

// byteTok gives every prompt its own ids (so prompts differ and their outputs can differ), and the verbalizers their fixed single ids.
type byteTok struct{ fakeTok }

func (b byteTok) EncodePlain(s string) ([]int, error) {
	if id, ok := b.ids[s]; ok {
		return []int{id}, nil
	}
	out := make([]int, 0, len(s))
	for i := 0; i < len(s); i++ {
		out = append(out, 40+int(s[i])%40)
	}
	return out, nil
}

func (b byteTok) EncodeChat(s string) ([]int, error) { return b.EncodePlain(s) }

func logitsFor(tok byteTok, ids []int) []float32 {
	sum := 0
	for _, id := range ids {
		sum += id
	}
	l := make([]float32, 128)
	for _, id := range tok.ids {
		l[id] = float32((sum+id)%7) / 3
	}
	return l
}

func manyReqs() []Request {
	return []Request{
		{Kind: KindNoul, State: "the same state", Question: "Is it urgent?", Options: noulOptions},
		{Kind: KindChoice, State: "the same state", Question: "Who handles it?", Options: []string{"billing", "tech", "sales"}},
		{Kind: KindScore, State: "the same state", Question: "How angry?", Options: scoreOptions},
	}
}

func TestDecideMany_oneCallSameAnswers(t *testing.T) {
	tok := byteTok{newFake()}
	var single, many, prompts int
	d, err := New(tok, func(_ context.Context, ids []int) ([]float32, error) { single++; return logitsFor(tok, ids), nil },
		Options{PrefillMany: func(_ context.Context, ps [][]int) ([][]float32, error) {
			many++
			prompts += len(ps)
			out := make([][]float32, len(ps))
			for i, p := range ps {
				out[i] = logitsFor(tok, p)
			}
			return out, nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	reqs := manyReqs()
	got, err := d.DecideMany(context.Background(), reqs)
	if err != nil {
		t.Fatal(err)
	}
	if many != 1 || prompts != 3 || single != 0 {
		t.Fatalf("DecideMany made %d many-calls over %d prompts and %d single calls, want 1, 3 and 0", many, prompts, single)
	}
	for i, r := range reqs {
		want, err := d.Decide(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		if len(got[i].Distribution) != len(want.Distribution) || got[i].Decision != want.Decision || got[i].PromptTokens != want.PromptTokens || got[i].Prefills != 1 {
			t.Fatalf("request %d: %+v, want %+v", i, got[i], want)
		}
		for j := range want.Distribution {
			if math.Abs(got[i].Distribution[j]-want.Distribution[j]) > 1e-12 {
				t.Errorf("request %d option %d: %v, want %v", i, j, got[i].Distribution[j], want.Distribution[j])
			}
		}
	}
	if single != 3 {
		t.Errorf("the three reference Decide calls made %d single calls", single)
	}
}

// Without the hook, with one request, or for a request that asks for several answer orders, each prompt is prefilled alone, as before.
func TestDecideMany_fallbacks(t *testing.T) {
	tok := byteTok{newFake()}
	var single, many int
	prefill := func(_ context.Context, ids []int) ([]float32, error) { single++; return logitsFor(tok, ids), nil }
	hook := func(_ context.Context, ps [][]int) ([][]float32, error) {
		many++
		out := make([][]float32, len(ps))
		for i, p := range ps {
			out[i] = logitsFor(tok, p)
		}
		return out, nil
	}
	plain, _ := New(tok, prefill, Options{})
	if _, err := plain.DecideMany(context.Background(), manyReqs()); err != nil || single != 3 {
		t.Fatalf("no hook: %d single calls, err %v, want 3", single, err)
	}
	single = 0
	d, _ := New(tok, prefill, Options{PrefillMany: hook})
	if _, err := d.DecideMany(context.Background(), manyReqs()[:1]); err != nil || many != 0 || single != 1 {
		t.Fatalf("one request: %d many, %d single, err %v, want 0 and 1", many, single, err)
	}
	single, many = 0, 0
	reqs := manyReqs()
	reqs[1].Permute = 3 // three answer orders: three prefills of its own, not part of the shared call
	got, err := d.DecideMany(context.Background(), reqs)
	if err != nil {
		t.Fatal(err)
	}
	if many != 1 || single != 3 || got[1].Prefills != 3 {
		t.Errorf("a permuted request: %d many-calls, %d single calls, %d prefills for it; want 1, 3 and 3", many, single, got[1].Prefills)
	}
}

func TestDecideMany_errors(t *testing.T) {
	tok := byteTok{newFake()}
	prefill := func(_ context.Context, ids []int) ([]float32, error) { return logitsFor(tok, ids), nil }
	short, _ := New(tok, prefill, Options{PrefillMany: func(_ context.Context, ps [][]int) ([][]float32, error) { return make([][]float32, 1), nil }})
	if _, err := short.DecideMany(context.Background(), manyReqs()); err == nil || !strings.Contains(err.Error(), "returned 1 outputs for 3 prompts") {
		t.Errorf("a hook returning too few outputs: %v", err)
	}
	d, _ := New(tok, prefill, Options{PrefillMany: func(_ context.Context, ps [][]int) ([][]float32, error) {
		t.Fatal("the hook ran before validation")
		return nil, nil
	}})
	bad := manyReqs()
	bad[2].Options = []string{"1", "2"} // a score must start at 0
	if _, err := d.DecideMany(context.Background(), bad); err == nil {
		t.Error("an invalid request in the batch was accepted")
	}
}
