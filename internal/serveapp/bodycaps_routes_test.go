package serveapp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Route-specific body-cap gates (v0.10.3 pre-tag review of the G1 default).
//
// The derived body cap is a NEW DEFAULT that can reject requests which previously worked, so each
// route's cap has to be justified by what that route actually carries. Two of the four checks had
// real answers:
//
//   - Multimodal is SAFE, and this pins why: the pre-tokenization guard measures only `type:"text"`
//     content parts, so a multi-megabyte base64 image never counts against a context window it does
//     not consume (an image is a few hundred tokens). The byte cap on the chat/messages routes is
//     visionCap = textCap + 32 MiB, so the payload itself fits too. Both halves must hold — if
//     either regresses, VL requests start failing on small-context models.
//
//   - Embeddings was WRONG: /v1/embeddings is served by the encoder, which is not in s.models, so a
//     cap derived from decoder context windows measured the wrong thing. On an embed-only server it
//     collapsed to the 4 MiB text floor and rejected a batch the route's own bounds accept.
func TestBodyCaps_embeddingsIsIndependentOfDecoderContext(t *testing.T) {
	// An embed-only server: no decoder loaded, so the derived TEXT cap falls to its 4 MiB floor.
	s := &server{models: map[string]*loadedModel{}}
	textCap, visionCap, embedCap := s.resolveBodyCaps(0)

	if textCap != maxBodyBytes {
		t.Fatalf("text cap = %d, want the %d floor with no decoder loaded", textCap, maxBodyBytes)
	}
	if embedCap == textCap {
		t.Fatal("embeddings cap tracks the decoder-derived text cap — the regression this test exists " +
			"for: /v1/embeddings is served by the encoder and has nothing to do with a chat model's context")
	}
	if embedCap != maxEmbedBodyBytes {
		t.Errorf("embed cap = %d, want %d", embedCap, maxEmbedBodyBytes)
	}
	// The concrete false rejection: the route's OWN bounds accept 2048 inputs, and a realistic
	// 4 KiB-per-input batch is ~8 MiB — over the 4 MiB text floor, under the embed cap.
	const realisticBatch = int64(2048 * 4 << 10) // ~8 MiB
	if realisticBatch <= textCap {
		t.Fatalf("fixture no longer demonstrates the gap: batch %d <= text cap %d", realisticBatch, textCap)
	}
	if realisticBatch > embedCap {
		t.Errorf("a legal %d-byte batch (2048 inputs × 4 KiB, within maxEmbedInputs/maxEmbedInputBytes) "+
			"exceeds the embeddings cap %d — the route rejects a body its own validator would accept",
			realisticBatch, embedCap)
	}
	if visionCap <= textCap {
		t.Errorf("vision cap %d should exceed text cap %d", visionCap, textCap)
	}
}

// TestBodyCaps_explicitOverrideGovernsEveryRoute: -max-body-bytes is documented as "the" cap, so it
// must not be silently ignored on the route with its own default.
func TestBodyCaps_explicitOverrideGovernsEveryRoute(t *testing.T) {
	s := &server{models: map[string]*loadedModel{}}
	const override = int64(7 << 20)
	textCap, _, embedCap := s.resolveBodyCaps(override)
	if textCap != override {
		t.Errorf("text cap = %d, want the override %d", textCap, override)
	}
	if embedCap != override {
		t.Errorf("embed cap = %d, want the override %d — an explicit -max-body-bytes must govern "+
			"/v1/embeddings too, or the flag silently does not mean what it says", embedCap, override)
	}
}

// TestPromptGuard_ignoresImageBytes is the multimodal half (Step 1a). A VL request carries megabytes
// of base64 image and a short text prompt; only the text may count toward the context window, or a
// small-context VL model rejects every image request.
func TestPromptGuard_ignoresImageBytes(t *testing.T) {
	// ~3 MB of base64 image data — a realistic photo payload — plus a short instruction.
	img := "data:image/png;base64," + strings.Repeat("A", 3<<20)
	parts := []map[string]any{
		{"type": "text", "text": "What is in this image?"},
		{"type": "image_url", "image_url": map[string]any{"url": img}},
	}
	raw, err := json.Marshal(parts)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	msgs := []chatMessage{{Role: "user", Content: raw}}

	got := chatInputBytes(msgs)
	if got > 1024 {
		t.Fatalf("chatInputBytes counted %d bytes for a 3 MiB image + 22-byte prompt — image data is "+
			"being charged against the context window, so a small-context VL model would reject a "+
			"request it can serve", got)
	}
	// And the guard itself must not fire for a small-context model on this request.
	if err := promptByteBudgetError(got, 2048, 16); err != nil {
		t.Errorf("pre-tokenization guard rejected a valid VL request on a 2048-token model: %v", err)
	}
	// Control: the same guard MUST still reject genuinely oversized text, or it is not doing its job.
	if err := promptByteBudgetError(2048*16+1, 2048, 16); err == nil {
		t.Error("guard accepted text that cannot fit the context window — the DoS protection is gone")
	}
}

// TestAnthropicInputBytes_excludesImages is the /v1/messages half of the pre-tokenization guard.
// The guard was added to the Anthropic route in v0.10.3 (it previously ran only on the OpenAI
// routes, so a body under the cap still tokenized in full before rejection). It must bound the same
// thing the OpenAI side bounds — tokenizable TEXT — or it becomes a new way to reject valid vision
// requests on small-context models, trading one defect for a worse one.
func TestAnthropicInputBytes_excludesImages(t *testing.T) {
	img := strings.Repeat("A", 3<<20) // ~3 MB of base64 image data
	blocks, err := json.Marshal([]map[string]any{
		{"type": "text", "text": "Describe this."},
		{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": img}},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req := &anthropicReq{
		System:   rawStr("You are helpful."),
		Messages: []anthropicMessage{{Role: "user", Content: blocks}},
	}

	got := anthropicInputBytes(req)
	if got > 1024 {
		t.Fatalf("anthropicInputBytes counted %d bytes for a 3 MiB image + short text — image data is "+
			"being charged against the context window, so the new guard would reject valid vision "+
			"requests on small-context models", got)
	}
	if err := promptByteBudgetError(got, 2048, 16); err != nil {
		t.Errorf("guard rejected a valid Anthropic vision request on a 2048-token model: %v", err)
	}
	// Control: genuinely oversized TEXT must still be caught on this route.
	big := &anthropicReq{Messages: []anthropicMessage{{Role: "user", Content: rawStr(strings.Repeat("x", 2048*16+1))}}}
	if err := promptByteBudgetError(anthropicInputBytes(big), 2048, 16); err == nil {
		t.Error("guard accepted Anthropic text that cannot fit the context window")
	}
}

// TestEmbeddings413_namesTheRoutesOwnBounds: /v1/embeddings advertises per-DIMENSION limits (2048
// inputs, 1 MiB each) that multiply out to 2 GiB — far past any body cap, so the two can never all
// be satisfied at once. A 413 naming only the body cap leaves a client that respected both
// per-dimension limits unable to tell which number it violated.
func TestEmbeddings413_namesTheRoutesOwnBounds(t *testing.T) {
	note := fmt.Sprintf("this route also limits each request to %d inputs of at most %d bytes each; "+
		"the body cap bounds their total", maxEmbedInputs, maxEmbedInputBytes)
	h := maxBytes(1024, func(http.ResponseWriter, *http.Request) { t.Fatal("handler ran for an over-cap body") }, note)

	req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(""))
	req.ContentLength = 4096
	rec := httptest.NewRecorder()
	h(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{"4096", "1024", "2048", "1048576"} {
		if !strings.Contains(body, want) {
			t.Errorf("413 body does not mention %q — the client cannot tell which of the three limits "+
				"it violated: %s", want, body)
		}
	}
}
