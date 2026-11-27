package systemone

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/internal/clef"
)

func TestParseRequestAndClefAnswer(t *testing.T) {
	items, err := ParseRequest(
		json.RawMessage(`"ticket: refund"`),
		json.RawMessage(`{"q1":{"type":"choice","instructions":"pick","criteria":{"b":"second","a":"first"}}}`),
	)
	if err != nil || len(items) != 1 || len(items[0].Names) != 2 || items[0].Names[0] != "b" {
		t.Fatalf("ParseRequest = %+v, %v", items, err)
	}
	if math.Abs(Confidence(0.8, 3)-0.7) > 1e-9 {
		t.Fatalf("Confidence(0.8, 3) = %v, want 0.7", Confidence(0.8, 3))
	}
	ans, err := ClefAnswer(
		ClefQuestion{Name: "q", Typ: "choice", Keys: []string{"b", "a"}},
		clef.Answer{ID: "q", Options: []string{"a", "b"}, Probs: []float64{0.5, 0.5}},
	)
	if err != nil || ans.(map[string]any)["choice"] != "b" {
		t.Fatalf("ClefAnswer = %+v, %v", ans, err)
	}
}
