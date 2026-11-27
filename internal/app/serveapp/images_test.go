package serveapp

import (
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

const png1x1b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

func TestLastUserTurn(t *testing.T) {
	turns := []chat.Turn{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}, {Role: "user", Content: "c"}}
	if i := lastUserTurn(turns); i != 2 {
		t.Errorf("lastUserTurn = %d, want 2", i)
	}
	if i := lastUserTurn([]chat.Turn{{Role: "assistant", Content: "x"}}); i != -1 {
		t.Errorf("no user turn should give -1, got %d", i)
	}
}
