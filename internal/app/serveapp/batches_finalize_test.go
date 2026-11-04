package serveapp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"testing"
)

// TestAssembleOpenAIOutput_splitsSuccessAndErrorPreservingOrder: a success line goes to output, an
// error line goes to error — never both, never neither — and each keeps its input custom_id order
// within its own file (task doc's own gate: "the result file's custom_id ordering matches the input").
func TestAssembleOpenAIOutput_splitsSuccessAndErrorPreservingOrder(t *testing.T) {
	b := &batchRecord{
		CustomIDs: []string{"req-1", "req-2", "req-3"},
		Results: []batchLineResult{
			{CustomID: "req-1", StatusCode: 200, Body: map[string]any{"object": "chat.completion"}},
			{CustomID: "req-2", ErrType: "invalid_request_error", ErrMsg: "images are not supported"},
			{CustomID: "req-3", StatusCode: 200, Body: map[string]any{"object": "chat.completion"}},
		},
	}
	out, errs := assembleOpenAIOutput(b)

	var outIDs []string
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		var line struct {
			CustomID string           `json:"custom_id"`
			Response *json.RawMessage `json:"response"`
			Error    *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatalf("output line did not parse: %v", err)
		}
		if line.Response == nil {
			t.Errorf("output line %s has a nil response", line.CustomID)
		}
		if line.Error != nil && string(*line.Error) != "null" {
			t.Errorf("output line %s carries a non-null error: %s", line.CustomID, *line.Error)
		}
		outIDs = append(outIDs, line.CustomID)
	}
	if want := []string{"req-1", "req-3"}; !equalStrings(outIDs, want) {
		t.Errorf("output custom_ids = %v, want %v", outIDs, want)
	}

	var errIDs []string
	sc = bufio.NewScanner(bytes.NewReader(errs))
	for sc.Scan() {
		var line struct {
			CustomID string           `json:"custom_id"`
			Response *json.RawMessage `json:"response"`
			Error    *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatalf("error line did not parse: %v", err)
		}
		if line.Response != nil && string(*line.Response) != "null" {
			t.Errorf("error line %s carries a non-null response: %s", line.CustomID, *line.Response)
		}
		if line.Error == nil {
			t.Errorf("error line %s has a nil error", line.CustomID)
		}
		errIDs = append(errIDs, line.CustomID)
	}
	if want := []string{"req-2"}; !equalStrings(errIDs, want) {
		t.Errorf("error custom_ids = %v, want %v", errIDs, want)
	}
}

func equalStrings(a, b []string) bool {
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
