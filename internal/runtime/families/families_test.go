package families

import (
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder/config"
)

func TestResolve_unknownModelType(t *testing.T) {
	_, _, err := Resolve(&config.Config{ModelType: "nonexistent_family_xyz"})
	if !errors.Is(err, ErrUnsupportedModel) {
		t.Fatalf("got err=%v, want ErrUnsupportedModel", err)
	}
	if !strings.Contains(KnownModelTypes(), "llama") {
		t.Fatalf("KnownModelTypes missing llama: %q", KnownModelTypes())
	}
}

func TestValidateConfigBounds_rejectsHostileLayerCount(t *testing.T) {
	err := ValidateConfigBounds(&config.Config{NumLayers: maxGGUFLayers + 1})
	if err == nil || !strings.Contains(err.Error(), "ceiling") {
		t.Fatalf("got err=%v, want ceiling error", err)
	}
}
